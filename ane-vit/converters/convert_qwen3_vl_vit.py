#!/usr/bin/env python3
"""Convert Qwen3-VL vision tower to Core ML for ANE.

Qwen3-VL's vision tower expects pre-patchified inputs (each row is one
patch's flattened pixels) plus a `grid_thw` tensor describing the
temporal/height/width patch counts. To keep the Core ML interface simple
(raw image pixels in, hidden out), we wrap the vision module with an
inline patchifier so the traced graph accepts a single 4-D pixel tensor.

Usage:
    uv run --isolated --with torch==2.7.0 --with transformers \
        --with coremltools --with scipy --with numpy \
        python ane-vit-spike/converters/convert_qwen3_vl_vit.py \
            --model Qwen/Qwen3-VL-2B-Instruct \
            --output ane-vit-spike/qwen3-vl-2b-vit.mlpackage \
            --compute-units cpuAndNeuralEngine
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--image-size", type=int, default=448,
                   help="Square pixel side; must be multiple of patch*spatial_merge")
    p.add_argument("--compute-units", default="all",
                   choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"])
    p.add_argument("--precision", default="float16",
                   choices=["float16", "float32"])
    args = p.parse_args()

    import torch
    import coremltools as ct
    from transformers import AutoModel, AutoConfig

    print(f"[convert] loading {args.model} ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    # Find vision tower.
    vt = None
    for name, mod in model.named_modules():
        if name == "visual" or name.endswith(".visual"):
            vt = mod
            break
    if vt is None:
        for name, _ in model.named_modules():
            print(" ", name)
        raise SystemExit("Could not locate Qwen3-VL vision tower (.visual).")
    print(f"[convert] vision tower located at: {type(vt).__name__}")

    # Suppress deepstack outputs for tracing — Core ML's tracer can't handle
    # a (Tensor, List[Tensor]) return. The LLM body normally consumes
    # mid-ViT layer outputs from `deepstack_visual_indexes`; we skip them
    # for the spike. End-to-end fidelity will be slightly off (the LLM
    # will see only the final ViT output) but the perf comparison is
    # apples-to-apples for the inference path itself.
    if hasattr(vt, "deepstack_visual_indexes"):
        vt.deepstack_visual_indexes = []
    if hasattr(vt, "config") and hasattr(vt.config, "deepstack_visual_indexes"):
        vt.config.deepstack_visual_indexes = []
    print("[convert] deepstack_visual_indexes cleared for tracing")

    # Monkey-patch chunked-attention forward → single full-attention call.
    # The stock Qwen3-VL Qwen3VLVisionAttention.forward splits queries/
    # keys/values by `cu_seqlens` (windowed attention) and Python-loops
    # over each chunk, then cats. The split is data-dependent
    # (`lengths.tolist()`) and coremltools can't trace it (it bakes a
    # specific chunk count then collides with mixed dtypes inside the
    # cat op). For square single-image inputs the windowing is degenerate
    # (1 chunk = full attention), so replacing the path with one direct
    # SDPA call is semantically equivalent for our use case.
    import torch.nn.functional as F
    def _full_attention_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                                position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        # apply_rotary_pos_emb_vision is in the parent module; re-import here.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        # Single SDPA call over the whole sequence (no chunking).
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    from transformers.models.qwen3_vl import modeling_qwen3_vl as _mqvl
    n_patched = 0
    for name, mod in vt.named_modules():
        if type(mod).__name__ == "Qwen3VLVisionAttention":
            mod.forward = _full_attention_forward.__get__(mod, type(mod))
            n_patched += 1
    print(f"[convert] patched {n_patched} Qwen3VLVisionAttention.forward → full SDPA (bypass windowed cu_seqlens split)")

    vcfg = config.vision_config
    P = vcfg.patch_size
    T = vcfg.temporal_patch_size
    M = vcfg.spatial_merge_size
    H = W = args.image_size
    if H % (P * M) != 0:
        raise SystemExit(
            f"image_size {H} must be a multiple of patch*merge ({P*M})"
        )
    grid_h = H // P
    grid_w = W // P
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    print(f"[convert] grid_thw=[1,{grid_h},{grid_w}] patch={P} temporal_patch={T} merge={M}")

    # The vision tower forward calls three module-level helpers from
    # transformers.vision_utils that iterate over `grid_thw.tolist()` and
    # build tensors via list-of-lists. coremltools traces those builders
    # as mixed-dtype stack ops and fails. Since input shape is fixed at
    # convert time, precompute their outputs and rebind the helpers in
    # the modeling module's namespace to return the constants.
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _mqv
    with torch.no_grad():
        _bi, _bw = _mqv.get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=vt.num_grid_per_side,
            spatial_merge_size=vt.config.spatial_merge_size,
            kwargs=None,
        )
        _pids = _mqv.get_vision_position_ids(grid_thw, vt.spatial_merge_size, kwargs=None)
        _cu = _mqv.get_vision_cu_seqlens(grid_thw, kwargs=None)
    _bi = _bi.detach().clone()
    _bw = _bw.detach().clone()
    _pids = _pids.detach().clone()
    _cu = _cu.detach().clone() if hasattr(_cu, "detach") else _cu
    vt.register_buffer("_frozen_bilinear_indices", _bi, persistent=False)
    vt.register_buffer("_frozen_bilinear_weights", _bw, persistent=False)
    vt.register_buffer("_frozen_vision_position_ids", _pids, persistent=False)
    vt.register_buffer("_frozen_cu_seqlens", _cu, persistent=False)
    _mqv.get_vision_bilinear_indices_and_weights = (
        lambda grid_thw, num_grid_per_side, spatial_merge_size, kwargs=None: (
            vt._frozen_bilinear_indices, vt._frozen_bilinear_weights
        )
    )
    _mqv.get_vision_position_ids = (
        lambda grid_thw, spatial_merge_size, kwargs=None: vt._frozen_vision_position_ids
    )
    _mqv.get_vision_cu_seqlens = (
        lambda grid_thw, kwargs=None: vt._frozen_cu_seqlens
    )
    print(f"[convert] froze vision_utils helpers: bilinear_indices={tuple(_bi.shape)} "
          f"bilinear_weights={tuple(_bw.shape)} position_ids={tuple(_pids.shape)} "
          f"cu_seqlens={tuple(_cu.shape) if hasattr(_cu, 'shape') else _cu}")

    # Wrap: takes raw [1, 3, H, W] in, returns hidden [(grid_h/M)*(grid_w/M), out_hidden].
    # We must repeat the image temporal_patch times then patchify into the
    # flattened layout the inner module expects:
    #   pixel_values: [num_patches_flat, C * temporal_patch * P * P]
    # where num_patches_flat = T_blocks * grid_h * grid_w and T_blocks = 1
    # (for a still image we duplicate so the temporal axis is full).
    class QwenVLVisionWrap(torch.nn.Module):
        def __init__(self, inner, P, T, grid_h, grid_w):
            super().__init__()
            self.inner = inner
            self.P = P
            self.T = T
            self.grid_h = grid_h
            self.grid_w = grid_w
            self.register_buffer("grid_thw", torch.tensor([[1, grid_h, grid_w]], dtype=torch.long))

        def forward(self, pixels):
            # pixels: [1, 3, H, W]
            grid_h, grid_w, P, T = self.grid_h, self.grid_w, self.P, self.T
            # Tile across temporal axis: [T, 3, H, W]
            pix = pixels.expand(T, -1, -1, -1).contiguous()
            # Reshape using reshape+permute (Core-ML-friendly; unfold is
            # not yet supported by coremltools as of v9.0).
            #   [T, 3, grid_h*P, grid_w*P]
            #     reshape  -> [T, 3, grid_h, P, grid_w, P]
            #     permute  -> [grid_h, grid_w, T, 3, P, P]
            #     reshape  -> [grid_h*grid_w, T*3*P*P]
            x = pix.reshape(T, 3, grid_h, P, grid_w, P)
            x = x.permute(2, 4, 0, 1, 3, 5).contiguous()
            patches = x.reshape(grid_h * grid_w, T * 3 * P * P)
            out = None
            try:
                out = self.inner(patches, self.grid_thw)
            except TypeError:
                out = self.inner(patches, grid_thw=self.grid_thw)
            # Qwen3-VL returns a BaseModelOutputWithPooling dataclass.
            # The tracer can't deal with the dataclass / empty deepstack
            # list — pull out just `last_hidden_state` so the graph
            # output is a single tensor.
            if isinstance(out, (tuple, list)):
                return out[0]
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            return out

    wrapped = QwenVLVisionWrap(vt, P, T, grid_h, grid_w).eval()

    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    print(f"[convert] tracing with input shape {tuple(dummy.shape)} ...")
    try:
        traced = torch.jit.trace(wrapped, dummy, strict=False)
    except Exception as e:
        print(f"[convert] trace failed: {e}")
        raise

    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }
    precision_map = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
    }
    print("[convert] converting to Core ML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixels", shape=(1, 3, H, W), dtype=float)],
        compute_units=cu_map[args.compute_units],
        compute_precision=precision_map[args.precision],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"Qwen3-VL vision tower (image_size={H})"
    mlmodel.save(str(args.output))
    print(f"[convert] saved {args.output}")

    manifest = {
        "model": args.model,
        "image_size": H,
        "patch_size": P,
        "temporal_patch_size": T,
        "spatial_merge_size": M,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "input_name": "pixels",
        "input_shape": [1, 3, H, W],
        "compute_units": args.compute_units,
        "precision": args.precision,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
