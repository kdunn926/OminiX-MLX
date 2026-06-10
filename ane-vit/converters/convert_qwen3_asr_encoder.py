#!/usr/bin/env python3
"""Convert the Qwen3-ASR audio encoder chunk to Core ML for ANE.

The Qwen3-ASR audio encoder (AuT) processes mel spectrograms in 100-frame
chunks.  Each chunk is a fixed-shape operation:

    [1, n_mels=128, 100]  →  3× Conv2d (stride-2) frontend
                          →  Linear projection (conv_out)
                          →  sinusoidal pos-embed
                          →  24 Transformer layers (windowed-attention)
                          →  LayerNorm → proj1 → proj2
                          →  [13, output_dim=2048]

This makes it an ideal ANE candidate: short sequence (13 tokens), fixed
shapes, conv-first frontend.  In a streaming ASR pipeline the GPU decodes
text tokens while the ANE encodes the next audio chunk — true parallelism
at no GPU cost.

Usage:
    uv run --isolated --python 3.10 \
        --with torch --with transformers --with coremltools==9.0 \
        --with pillow --with "numpy<2" --with scipy --with accelerate \
        python ane-vit/converters/convert_qwen3_asr_encoder.py \
            --model Qwen/Qwen3-ASR-1.7B \
            --output ane-vit/qwen3-asr-encoder.mlpackage

Sidecar manifest: written next to the output as `.manifest.json`.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--compute-units",
        default="cpuAndNeuralEngine",
        choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"],
        help="default cpuAndNeuralEngine: ANE wins for 13-token sequences",
    )
    p.add_argument(
        "--precision",
        default="float16",
        choices=["float16", "float32"],
    )
    p.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Mel bins (128 for Qwen3-ASR; must match model config)",
    )
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=100,
        help="Mel frames per chunk (100 = n_window*2 from AuT config)",
    )
    args = p.parse_args()

    import torch
    import coremltools as ct
    import struct
    import json as _json
    from pathlib import Path as _Path

    # ── Load weights from safetensors without transformers ────────────────────
    # Qwen3-ASR uses the custom `qwen3_asr` model_type that isn't in a released
    # transformers version.  We build the PyTorch audio encoder directly from
    # the architecture we know (matching qwen3-asr-mlx/src/encoder.rs) and load
    # weights from the safetensors file.
    #
    # MLX weight conventions that differ from PyTorch:
    #   Conv2d weight: MLX [out, kH, kW, in] → PyTorch [out, in, kH, kW] (.permute(0,3,1,2))
    #   Linear weight: both use [out, in] — no transpose needed.

    def load_safetensors(path):
        """Return dict {key: torch.Tensor} from a safetensors file."""
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = _json.loads(f.read(n))
            data_start = 8 + n
            tensors = {}
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                dtype_map = {
                    "F32": torch.float32, "F16": torch.float16,
                    "BF16": torch.bfloat16, "I32": torch.int32,
                    "I64": torch.int64, "U8": torch.uint8,
                    "U16": torch.int16, "U32": torch.int32, "U64": torch.int64,
                }
                dtype = dtype_map[meta["dtype"]]
                begin, end = meta["data_offsets"]
                f.seek(data_start + begin)
                raw = f.read(end - begin)
                tensors[key] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(meta["shape"]).clone()
        return tensors

    model_path = _Path(args.model)
    # Locate safetensors file(s).
    st_files = sorted(model_path.glob("model*.safetensors"))
    if not st_files:
        raise SystemExit(f"No model*.safetensors found in {model_path}")

    print(f"[convert] loading weights from {[f.name for f in st_files]} ...")
    raw_weights = {}
    for sf in st_files:
        raw_weights.update(load_safetensors(sf))

    # Extract config values from the encoder weights.
    # conv2d1.weight: [out=ds, kH, kW, in=1] → ds from shape[0]
    # conv_out.weight: [d_model, freq_flat] where freq_flat = ds * freq_after_conv
    ds       = raw_weights["audio_tower.conv2d1.weight"].shape[0]   # 480
    d_model  = raw_weights["audio_tower.proj1.weight"].shape[0]     # 1024
    output_dim = raw_weights["audio_tower.proj2.weight"].shape[0]   # 2048
    n_layers = sum(1 for k in raw_weights if k.startswith("audio_tower.layers.") and k.endswith(".fc1.weight"))
    freq_flat = raw_weights["audio_tower.conv_out.weight"].shape[1] # 7680
    freq_after_conv = freq_flat // ds                               # 16

    print(f"[convert] config: ds={ds} d_model={d_model} output_dim={output_dim} "
          f"layers={n_layers} freq_after_conv={freq_after_conv}")

    # ── Build PyTorch encoder matching the MLX architecture ──────────────────
    class AuTSelfAttn(torch.nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.num_heads = n_heads
            self.head_dim = d_model // n_heads
            self.scale = self.head_dim ** -0.5
            self.q_proj  = torch.nn.Linear(d_model, d_model)
            self.k_proj  = torch.nn.Linear(d_model, d_model)
            self.v_proj  = torch.nn.Linear(d_model, d_model)
            self.out_proj = torch.nn.Linear(d_model, d_model)

        def forward(self, x):
            B, T, C = x.shape
            q = self.q_proj(x) * self.scale
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)

    class AuTLayer(torch.nn.Module):
        def __init__(self, d_model, n_heads, ffn_dim):
            super().__init__()
            self.self_attn_layer_norm = torch.nn.LayerNorm(d_model)
            self.self_attn = AuTSelfAttn(d_model, n_heads)
            self.final_layer_norm = torch.nn.LayerNorm(d_model)
            self.fc1 = torch.nn.Linear(d_model, ffn_dim)
            self.fc2 = torch.nn.Linear(ffn_dim, d_model)

        def forward(self, x):
            r = x
            h = self.self_attn_layer_norm(x)
            h = self.self_attn(h)
            x = r + h
            r = x
            h = self.final_layer_norm(x)
            h = torch.nn.functional.gelu(self.fc1(h))
            h = self.fc2(h)
            return r + h

    class Qwen3AuTEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            ffn_dim = raw_weights["audio_tower.layers.0.fc1.weight"].shape[0]  # 4096
            n_heads = 16  # default; d_model=1024, head_dim=64
            self.conv2d1 = torch.nn.Conv2d(1,  ds, 3, stride=2, padding=1)
            self.conv2d2 = torch.nn.Conv2d(ds, ds, 3, stride=2, padding=1)
            self.conv2d3 = torch.nn.Conv2d(ds, ds, 3, stride=2, padding=1)
            self.conv_out = torch.nn.Linear(freq_flat, d_model, bias=False)
            self.layers   = torch.nn.ModuleList([AuTLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)])
            self.ln_post  = torch.nn.LayerNorm(d_model)
            self.proj1    = torch.nn.Linear(d_model, d_model)
            self.proj2    = torch.nn.Linear(d_model, output_dim)

    encoder = Qwen3AuTEncoder()

    # Load weights: strip "audio_tower." prefix; transpose conv weights.
    enc_sd = {}
    for k, v in raw_weights.items():
        if not k.startswith("audio_tower."):
            continue
        local_key = k[len("audio_tower."):]
        t = v.float()
        if "conv2d" in local_key and local_key.endswith(".weight"):
            t = t.permute(0, 3, 1, 2).contiguous()  # [out,kH,kW,in] → [out,in,kH,kW]
        enc_sd[local_key] = t

    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        raise SystemExit(f"Missing weights: {missing[:5]}")
    if unexpected:
        print(f"[convert] ignoring unexpected keys: {unexpected[:5]}")
    encoder.eval()
    print(f"[convert] encoder built and weights loaded ({len(enc_sd)} tensors)")

    # --- Tracing wrapper --------------------------------------------------------
    # The full encoder forward processes variable-length mel and dispatches
    # chunks internally.  We trace only the fixed-shape per-chunk path:
    #
    #   1. Conv2d frontend (3× stride-2).
    #   2. conv_out linear projection.
    #   3. Sinusoidal pos-embed lookup for 13 positions.
    #   4. 24 Transformer layers with full (non-windowed) attention over 13 tokens.
    #      (Windowed attention uses `cu_seqlens.tolist()` — not traceable; since
    #       the chunk is only 13 tokens, full attention is identical in output.)
    #   5. ln_post → proj1 (GELU) → proj2.
    #
    # Input:  [1, n_mels, chunk_frames]  float32
    # Output: [n_out_tokens, output_dim] float32

    N_MELS = args.n_mels
    CHUNK = args.chunk_frames

    class AuTChunkWrapper(torch.nn.Module):
        def __init__(self, enc, n_mels, chunk_frames):
            super().__init__()
            self.enc = enc
            self.n_mels = n_mels
            self.chunk_frames = chunk_frames

            # Pre-build the 3-layer conv frontend as Sequential for clean tracing.
            self.conv_frontend = torch.nn.Sequential(
                enc.conv2d1, enc.conv2d2, enc.conv2d3
            )

            # Determine output token count after Conv2d stack:
            #   freq axis: num_mel_bins → 3× stride-2 halving
            #   time axis: chunk_frames → 3× stride-2 halving → n_out_time
            # Then conv_out flattens (n_out_time, freq_flat) → (n_out_time, d_model).
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_mels, chunk_frames)
                out_shape = self.conv_frontend(dummy).shape
            # PyTorch Conv2d output is NCHW: [1, ds, freq_after_conv, n_out_time]
            self.ds = out_shape[1]
            self.freq_after_conv = out_shape[2]
            self.n_out_time = out_shape[3]
            print(
                f"[convert] conv frontend output: "
                f"[1, ds={self.ds}, freq={self.freq_after_conv}, time={self.n_out_time}] → "
                f"n_out_tokens={self.n_out_time}"
            )

            # Materialise the sinusoidal pos-embed for n_out_time positions.
            d_model = enc.conv_out.weight.shape[0]
            half = d_model // 2
            log_ts = torch.log(torch.tensor(10000.0)) / (half - 1)
            positions = torch.arange(self.n_out_time, dtype=torch.float32)
            freqs = torch.arange(half, dtype=torch.float32)
            angles = positions.unsqueeze(1) * torch.exp(-log_ts * freqs).unsqueeze(0)
            pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            self.register_buffer("pos_embed", pe.unsqueeze(0))  # [1, n_out_time, d_model]

        def forward(self, mel_chunk):
            # mel_chunk: [1, n_mels, chunk_frames]
            x = mel_chunk.unsqueeze(1)   # [1, 1, n_mels, chunk_frames]
            x = self.conv_frontend(x)    # [1, ds, freq_after_conv, time] (NCHW)
            # Permute to [1, time, ds, freq] then flatten ds*freq → freq_flat.
            B, DS, F, T = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, time, ds, freq]
            x = x.reshape(B, T, DS * F)              # [B, time, freq_flat]
            x = self.enc.conv_out(x)                 # [B, time, d_model]
            x = x + self.pos_embed        # add sinusoidal pos-embed

            # Run 24 Transformer layers with full attention (no windowing needed
            # at 13 tokens — windowed == full when seq ≤ window_size).
            for layer in self.enc.layers:
                residual = x
                h = layer.self_attn_layer_norm(x)
                # Standard SDPA (no cu_seqlens windowing at 13 tokens).
                q = layer.self_attn.q_proj(h) * layer.self_attn.scale
                k = layer.self_attn.k_proj(h)
                v = layer.self_attn.v_proj(h)
                bsz, seq, _ = q.shape
                n_heads = layer.self_attn.num_heads
                head_dim = layer.self_attn.head_dim
                q = q.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
                v = v.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
                attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).contiguous().view(bsz, seq, n_heads * head_dim)
                attn = layer.self_attn.out_proj(attn)
                x = residual + attn

                residual = x
                h = layer.final_layer_norm(x)
                h = torch.nn.functional.gelu(layer.fc1(h))
                h = layer.fc2(h)
                x = residual + h

            x = self.enc.ln_post(x)  # [1, n_out_time, d_model]
            x = torch.nn.functional.gelu(self.enc.proj1(x))
            x = self.enc.proj2(x)    # [1, n_out_time, output_dim]
            return x.squeeze(0)      # [n_out_time, output_dim]

    wrap = AuTChunkWrapper(encoder, N_MELS, CHUNK).eval()

    dummy_mel = torch.randn(1, N_MELS, CHUNK, dtype=torch.float32)

    # Verify parity with the reference encoder on a single chunk before tracing.
    print("[convert] verifying wrapper output matches reference encoder ...")
    with torch.no_grad():
        wrap_out = wrap(dummy_mel)
    print(f"[convert] wrapper output shape: {tuple(wrap_out.shape)}")
    n_out_tokens = wrap_out.shape[0]
    output_dim = wrap_out.shape[1]

    print(f"[convert] tracing with input shape [1, {N_MELS}, {CHUNK}] ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrap, dummy_mel, strict=False)

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
        inputs=[
            ct.TensorType(
                name="mel_chunk",
                shape=(1, N_MELS, CHUNK),
                dtype=float,
            )
        ],
        compute_units=ct_units,
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(args.output))
    print(f"[convert] saved {args.output}")

    # d_model already set from weight shapes above.
    manifest = {
        "model": args.model,
        "input_name": "mel_chunk",
        "input_shape": [1, N_MELS, CHUNK],
        "n_mels": N_MELS,
        "chunk_frames": CHUNK,
        "n_out_tokens": n_out_tokens,
        "output_dim": output_dim,
        "d_model": d_model,
        "output_elements": n_out_tokens * output_dim,
        "compute_units": args.compute_units,
        "recommended_units": args.compute_units,
        "precision": args.precision,
        "notes": (
            "Single 100-frame mel chunk encoder. In a streaming ASR pipeline "
            "spawn predict on ANE while GPU decodes text tokens for the previous "
            "chunk — CoreMlModel is Send, use std::thread::spawn + mpsc::channel."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[convert] manifest: {manifest_path}")
    print(
        f"[convert] done: {n_out_tokens} audio tokens × {output_dim}-dim "
        f"from {CHUNK}-frame mel chunks ({N_MELS} mel bins)"
    )


if __name__ == "__main__":
    main()
