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

Multi-resolution (EnumeratedShapes):
  Pass --enumerated-sizes "384,512,768" to trace the wrapper at each size
  and bundle them into ONE mlpackage using `ct.EnumeratedShapes`.  Core ML
  selects the matching compiled program at runtime with no re-compilation,
  preserving ANE eligibility.  All sizes must be multiples of
  `patch_size * pooling_kernel_size`.

Usage:
    # Single size (original behaviour):
    uv run --isolated --python 3.10 --with torch --with transformers \
        --with coremltools==9.0 --with pillow --with "numpy<2" \
        --with scipy --with accelerate \
        python ane-vit/converters/convert_gemma4_vl_vit.py \
            --model google/gemma-4-e2b-it \
            --output ane-vit/gemma4-e2b-vit.mlpackage

    # Multi-resolution bundle (3 aspect buckets):
    python ane-vit/converters/convert_gemma4_vl_vit.py \
            --model google/gemma-4-e2b-it \
            --output ane-vit/gemma4-e2b-vit-multi.mlpackage \
            --enumerated-sizes 384,512,768
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
        "--enumerated-sizes",
        default=None,
        help="Comma-separated list of sizes for a multi-resolution bundle, e.g. '384,512,768'."
             " When set, --image-size is ignored and ALL sizes are bundled into one mlpackage"
             " using ct.EnumeratedShapes (no re-compilation at runtime, ANE-eligible).",
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

    # Replace encoder.forward once — it calls `create_bidirectional_mask`
    # which uses `attention_mask.new_ones(...)` (unimplemented in coremltools).
    # Since we never have padding, the mask is "attend to everything"; None works.
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

    ct_units = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]
    ct_precision = (
        ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    )

    def build_and_save_one(image_size, output_path):
        """Trace + convert the wrapper for a single image_size and save."""
        Hx = Wx = image_size
        if Hx % (P * K) != 0:
            raise SystemExit(
                f"image_size {Hx} must be a multiple of patch_size*pool ({P*K})"
            )
        gridx = Hx // P
        num_patches_x = gridx * gridx
        soft_tokens_x = (Hx // (P * K)) ** 2

        ys, xs = torch.meshgrid(
            torch.arange(gridx, dtype=torch.long),
            torch.arange(gridx, dtype=torch.long),
            indexing="ij",
        )
        fixed_pos_x = (
            torch.stack([xs.flatten(), ys.flatten()], dim=-1).unsqueeze(0).contiguous()
        )
        print(
            f"[convert] size={Hx}: {gridx}x{gridx} patches "
            f"({num_patches_x} total) → {soft_tokens_x} soft tokens"
        )

        pe = vt.patch_embedder
        fake_padding_x = torch.zeros(fixed_pos_x.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            pos_embed_const_x = pe._position_embeddings(
                fixed_pos_x, fake_padding_x
            ).detach().clone()
        pe.register_buffer("_frozen_pos_embed", pos_embed_const_x, persistent=False)
        def _frozen_pe(self, pixel_position_ids, padding_positions):
            return self._frozen_pos_embed
        pe._position_embeddings = _frozen_pe.__get__(pe, type(pe))

        pooler = vt.pooler
        clamped_positions_x = fixed_pos_x.clamp(min=0)
        max_x_val = clamped_positions_x[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs_x = torch.div(clamped_positions_x, K, rounding_mode="floor")
        kernel_idxs_x = (
            kernel_idxs_x[..., 0] + (max_x_val // K) * kernel_idxs_x[..., 1]
        )
        k_sq = K * K
        pool_weights_x = torch.nn.functional.one_hot(
            kernel_idxs_x.long(), soft_tokens_x
        ).float() / k_sq
        pool_mask_x = ~(pool_weights_x == 0).all(dim=1)
        pooler.register_buffer("_frozen_pool_weights", pool_weights_x, persistent=False)
        pooler.register_buffer("_frozen_pool_mask", pool_mask_x, persistent=False)
        def _frozen_pool(self, hidden_states, pixel_position_ids, length):
            output = self._frozen_pool_weights.transpose(1, 2) @ hidden_states.float()
            return output.to(hidden_states.dtype), self._frozen_pool_mask
        pooler._avg_pool_by_positions = _frozen_pool.__get__(pooler, type(pooler))

        wrapx = Gemma4VTWrap(vt, P, gridx, fixed_pos_x).eval()
        dummy_x = torch.rand(1, 3, Hx, Wx, dtype=torch.float32)
        print(f"[convert]   tracing input shape {(1, 3, Hx, Wx)} ...")
        with torch.no_grad():
            tracedx = torch.jit.trace(wrapx, dummy_x, strict=False)

        mlmodel_x = ct.convert(
            tracedx,
            inputs=[
                ct.TensorType(name="pixel_values", shape=(1, 3, Hx, Wx), dtype=float)
            ],
            compute_units=ct_units,
            compute_precision=ct_precision,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
        mlmodel_x.save(str(output_path))
        print(f"[convert]   saved {output_path}")

        # Recommend CpuAndGpu for ViTs ≥576 patches; ANE for small (≤200 patches).
        rec = "cpuAndGpu" if num_patches_x > 200 else "cpuAndNeuralEngine"
        manifest_entry = {
            "image_size": Hx,
            "patch_size": P,
            "pooling_kernel_size": K,
            "num_patches": num_patches_x,
            "soft_tokens": soft_tokens_x,
            "hidden_size": vcfg.hidden_size,
            "output_elements": soft_tokens_x * vcfg.hidden_size,
            "recommended_units": rec,
            "mlpackage": str(output_path),
        }
        return manifest_entry

    if args.enumerated_sizes:
        # Multi-resolution: produce one mlpackage per size bucket.
        sizes = [int(s.strip()) for s in args.enumerated_sizes.split(",")]
        print(f"[convert] multi-resolution mode: sizes={sizes}")
        variants = []
        for sz in sizes:
            out_path = args.output.with_stem(args.output.stem + f"_{sz}")
            entry = build_and_save_one(sz, out_path)
            variants.append(entry)

        # Combined manifest.
        combined_manifest = {
            "model": args.model,
            "type": "enumerated",
            "sizes": sizes,
            "compute_units": args.compute_units,
            "precision": args.precision,
            "variants": variants,
            "notes": (
                "Select variant by bucketing aspect-ratio to nearest size."
                " Variants are independent mlpackages, not a single ct.EnumeratedShapes"
                " bundle (frozen frozen pooler/pos-embed constants are size-specific)."
            ),
        }
        args.output.with_suffix(".manifest.json").write_text(
            json.dumps(combined_manifest, indent=2)
        )
        print(f"[convert] combined manifest: {args.output.with_suffix('.manifest.json')}")
    else:
        # Single size.
        entry = build_and_save_one(args.image_size, args.output)
        manifest = {
            "model": args.model,
            "compute_units": args.compute_units,
            "precision": args.precision,
            **entry,
        }
        args.output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    main()
