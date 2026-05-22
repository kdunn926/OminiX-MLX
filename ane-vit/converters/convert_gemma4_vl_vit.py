#!/usr/bin/env python3
"""Convert Gemma 4 vision tower to Core ML for ANE.

Gemma 4's vision tower (`Gemma4VisionModel`) takes
`(pixel_values, pixel_position_ids)` where `pixel_values` is already
patchified into `[batch, num_patches, 3 * patch_size**2]` and
`pixel_position_ids` is `[batch, num_patches, 2]` carrying (x, y)
patch coords plus `(-1, -1)` for padding patches.

For the Core ML interface we wrap the module so it accepts raw
`[1, 3, H, W]` pixels (no padding), patchifies them inline, builds a
fixed all-valid position id tensor, and returns the pooled soft tokens
as a contiguous `[num_soft_tokens, hidden]` tensor.

Tracing fixes applied:
  - Skip the data-dependent `hidden_states[pooler_mask]` slice (all
    positions valid → mask is all-True, slice is identity).
  - Skip dataclass return (`BaseModelOutputWithPast`) — wrapper returns
    the bare tensor.

Usage:
    uv run --isolated --python 3.10 --with torch --with transformers \
        --with coremltools==9.0 --with pillow --with "numpy<2" \
        --with scipy --with accelerate \
        python ane-vit/converters/convert_gemma4_vl_vit.py \
            --model google/gemma-4-e2b-it \
            --output ane-vit/gemma4-e2b-vit.mlpackage
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-e2b-it")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Square pixel side; must be a multiple of patch_size * pooling_kernel_size",
    )
    p.add_argument(
        "--compute-units",
        default="all",
        choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"],
    )
    p.add_argument(
        "--precision",
        default="float16",
        choices=["float16", "float32"],
    )
    args = p.parse_args()

    import torch
    import coremltools as ct
    from transformers import AutoModelForImageTextToText, AutoConfig

    print(f"[convert] loading {args.model} ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    full = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    )
    full.eval()

    vt = None
    for name, mod in full.named_modules():
        if name == "vision_tower" or name.endswith(".vision_tower"):
            vt = mod
            break
    if vt is None:
        for name, _ in full.named_modules():
            print(" ", name)
        raise SystemExit("Could not locate Gemma4 vision tower (.vision_tower).")
    print(f"[convert] vision tower located at: {type(vt).__name__}")

    vcfg = vt.config
    P = vcfg.patch_size
    K = vcfg.pooling_kernel_size
    H = W = args.image_size
    if H % (P * K) != 0:
        raise SystemExit(
            f"image_size {H} must be a multiple of patch_size*pool ({P*K})"
        )
    grid = H // P                            # patches per side
    num_patches = grid * grid
    soft_tokens = (H // (P * K)) ** 2
    print(
        f"[convert] image={H} patch={P} pool={K} -> {grid}x{grid} patches "
        f"({num_patches} total), {soft_tokens} soft tokens out"
    )

    # Build fixed pixel_position_ids = [(x,y) for y in 0..grid for x in 0..grid].
    # Shape [1, num_patches, 2], no padding.
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.long),
        torch.arange(grid, dtype=torch.long),
        indexing="ij",
    )
    fixed_pos = torch.stack([xs.flatten(), ys.flatten()], dim=-1).unsqueeze(0).contiguous()
    print(f"[convert] fixed pixel_position_ids shape={tuple(fixed_pos.shape)} "
          f"(x_max={int(fixed_pos[..., 0].max())}, y_max={int(fixed_pos[..., 1].max())})")

    # Precompute the patch_embedder's `_position_embeddings` output for
    # the fixed position grid (one_hot + matmul + sum). coremltools'
    # torch frontend doesn't preserve int dtypes through `.clamp(min=0)`
    # → the resulting `one_hot` input is fp32 → MIL rejects. Since
    # positions are fixed, the whole result is a constant. Rebind
    # `_position_embeddings` to return it.
    pe = vt.patch_embedder
    fake_padding = torch.zeros(fixed_pos.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        pos_embed_const = pe._position_embeddings(fixed_pos, fake_padding).detach().clone()
    pe.register_buffer("_frozen_pos_embed", pos_embed_const, persistent=False)
    def _frozen_pe(self, pixel_position_ids, padding_positions):
        return self._frozen_pos_embed
    pe._position_embeddings = _frozen_pe.__get__(pe, type(pe))
    print(f"[convert] froze patch_embedder._position_embeddings "
          f"shape={tuple(pos_embed_const.shape)}")

    # Same story for the pooler: `_avg_pool_by_positions` uses one_hot
    # over int-clamped positions. Precompute its weight matrix once.
    # (Mask output is all-True since no padding.)
    pooler = vt.pooler
    enc_seq_len = num_patches
    pool_out_len = soft_tokens
    k = int((enc_seq_len // pool_out_len) ** 0.5)
    k_squared = k * k
    clamped_positions = fixed_pos.clamp(min=0)
    max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    pool_weights = torch.nn.functional.one_hot(
        kernel_idxs.long(), pool_out_len
    ).float() / k_squared
    pool_mask = ~(pool_weights == 0).all(dim=1)
    pooler.register_buffer("_frozen_pool_weights", pool_weights, persistent=False)
    pooler.register_buffer("_frozen_pool_mask", pool_mask, persistent=False)
    def _frozen_pool(self, hidden_states, pixel_position_ids, length):
        output = self._frozen_pool_weights.transpose(1, 2) @ hidden_states.float()
        return output.to(hidden_states.dtype), self._frozen_pool_mask
    pooler._avg_pool_by_positions = _frozen_pool.__get__(pooler, type(pooler))
    print(f"[convert] froze pooler weights k={k} shape={tuple(pool_weights.shape)}")

    # Replace the encoder.forward — it calls `create_bidirectional_mask`
    # which uses `attention_mask.new_ones(...)` (unimplemented in
    # coremltools). Since we never have padding, the mask is just
    # "attend to everything"; passing None works with all SDPA impls.
    encoder = vt.encoder
    def _enc_forward(self, inputs_embeds, attention_mask=None,
                     pixel_position_ids=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)
    encoder.forward = _enc_forward.__get__(encoder, type(encoder))
    print("[convert] encoder.forward replaced (skip create_bidirectional_mask, mask=None)")

    # Wrapper: accepts raw [1, 3, H, W], patchifies, runs the vision
    # tower's inner ops without dataclass return + without the
    # data-dependent boolean-mask slice.
    class Gemma4VTWrap(torch.nn.Module):
        def __init__(self, inner, P, grid, fixed_pos):
            super().__init__()
            self.inner = inner
            self.P = P
            self.grid = grid
            self.register_buffer("fixed_pos", fixed_pos, persistent=False)

        def forward(self, pixel_values):
            # [1, 3, H, W] -> [1, 3, grid, P, grid, P] -> [1, grid, grid, 3, P, P]
            #              -> [1, grid*grid, 3*P*P]
            B = pixel_values.shape[0]
            x = pixel_values.reshape(B, 3, self.grid, self.P, self.grid, self.P)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B, self.grid * self.grid, 3 * self.P * self.P)

            pos = self.fixed_pos.expand(B, -1, -1)
            # Replicate Gemma4VisionModel.forward, minus dataclass +
            # boolean-mask slice (all valid).
            inner = self.inner
            cfg = inner.config
            pooling_kernel_size = cfg.pooling_kernel_size
            output_length = self.grid * self.grid // (pooling_kernel_size * pooling_kernel_size)

            padding_positions = (pos == -1).all(dim=-1)  # all False
            inputs_embeds = inner.patch_embedder(x, pos, padding_positions)
            enc_out = inner.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=~padding_positions,
                pixel_position_ids=pos,
            )
            hidden_states, _pool_mask = inner.pooler(
                hidden_states=enc_out.last_hidden_state,
                pixel_position_ids=pos,
                padding_positions=padding_positions,
                output_length=output_length,
            )
            # Skip `hidden_states[pool_mask]` — mask is all-True.
            if cfg.standardize:
                hidden_states = (hidden_states - inner.std_bias) * inner.std_scale
            # [1, output_length, hidden]
            return hidden_states

    wrap = Gemma4VTWrap(vt, P, grid, fixed_pos).eval()

    dummy_pixels = torch.rand(1, 3, H, W, dtype=torch.float32)
    print(f"[convert] tracing with input shape {(1, 3, H, W)} ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrap, dummy_pixels, strict=False)

    print("[convert] converting to Core ML ...")
    ct_units = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]
    ct_precision = (
        ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, H, W), dtype=float)],
        compute_units=ct_units,
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(args.output))
    print(f"[convert] saved {args.output}")

    manifest = {
        "model": args.model,
        "image_size": H,
        "patch_size": P,
        "pooling_kernel_size": K,
        "num_patches": num_patches,
        "soft_tokens": soft_tokens,
        "hidden_size": vcfg.hidden_size,
        "output_elements": soft_tokens * vcfg.hidden_size,
        "compute_units": args.compute_units,
        "precision": args.precision,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


if __name__ == "__main__":
    main()
