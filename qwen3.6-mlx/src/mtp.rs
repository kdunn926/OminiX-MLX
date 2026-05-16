//! Multi-Token Prediction (MTP) head for Qwen3.6.
//!
//! Qwen3.6 declares `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings`
//! in its config. The MTPLX-Optimized-Speed checkpoint ships the MTP weights
//! as a sidecar `mtp.safetensors` file. Most stock `mlx-community` releases
//! strip the MTP weights — see `mlx_lm.models.qwen3_5.Qwen3_5MoEModel.sanitize`,
//! which drops every key containing `"mtp."` — so most loads return `None`.
//!
//! Actual layout (MTPLX-Optimized-Speed sidecar):
//!
//! ```text
//!   mtp.fc.weight                          [H, 2H]   bf16  (plain Linear, not quantized)
//!   mtp.pre_fc_norm_embedding.weight       [H]       bf16
//!   mtp.pre_fc_norm_hidden.weight          [H]       bf16
//!   mtp.norm.weight                        [H]       bf16  (post-block RMS)
//!   mtp.layers.<i>.input_layernorm.weight                  (standard transformer block)
//!   mtp.layers.<i>.post_attention_layernorm.weight
//!   mtp.layers.<i>.self_attn.{q,k,v,o}_proj.{weight,scales,biases}  INT4 quantized
//!   mtp.layers.<i>.self_attn.{q,k}_norm.weight
//!   mtp.layers.<i>.mlp.{gate,up,down}_proj.{weight,scales,biases}   INT4 quantized
//! ```
//!
//! Forward semantics:
//!
//! ```text
//!   h_norm = pre_fc_norm_hidden(host_hidden)
//!   e_norm = pre_fc_norm_embedding(prev_token_emb)
//!   m      = fc(concat([h_norm, e_norm], -1))      // 2H -> H
//!   for layer in layers: m = layer.forward(m, mask=None, cache=fresh)
//!   out    = norm(m)
//!   return out                                       // [B, T, H]
//! ```
//!
//! The caller (mtplx-mlx) applies the host model's `apply_lm_head(...)` to
//! project `out` to vocabulary logits. There is no `shared_head` in the
//! sidecar — the host's tied embed / lm_head is reused.
//!
//! For backward compatibility, key probing also recognises legacy layouts
//! (`mtp_layers`, `language_model.model.mtp.*`, etc.) but currently only the
//! flat top-level `mtp.*` layout produced by the MTPLX sidecar is wired into
//! the loader.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    module::{Module, Param},
    nn,
    ops::concatenate_axis,
    Array,
};
use mlx_rs_core::cache::KVCache;

use crate::cache::HybridCache;
use crate::config::ModelArgs;
use crate::model::{
    get_weight, load_rms_norm, load_transformer_block, TransformerBlock,
};

/// MTP head: pre-FC adapters → fc(2H→H) → N transformer blocks → final norm.
///
/// Returns hidden state `[B, T, H]`. The caller projects to logits via the
/// host model's `apply_lm_head(...)`.
pub struct MtpHead {
    /// Number of MTP transformer layers (matches `mtp_num_hidden_layers`).
    pub num_layers: i32,
    /// Hidden size (matches the host model).
    pub hidden_size: i32,
    /// Whether this head expects dedicated token embeddings.
    pub use_dedicated_embeddings: bool,
    /// Names of weight keys that were detected in the checkpoint.
    pub detected_weight_keys: Vec<String>,
    /// `fc`: plain BF16 Linear projecting `[B, T, 2H] -> [B, T, H]`.
    pub fc: nn::Linear,
    /// RMSNorm applied to the previous-token embedding before concat.
    pub pre_fc_norm_embedding: nn::RmsNorm,
    /// RMSNorm applied to the host hidden state before concat.
    pub pre_fc_norm_hidden: nn::RmsNorm,
    /// Final RMSNorm applied to the block stack output (before LM head).
    pub norm: nn::RmsNorm,
    /// MTP transformer blocks. Empty in stub mode.
    pub layers: Vec<TransformerBlock>,
}

impl MtpHead {
    /// True when no real MTP weights were materialised.
    pub fn is_stub(&self) -> bool {
        self.layers.is_empty()
    }

    /// Draft the next-token hidden state given the last host hidden state
    /// and the previously-emitted token's embedding.
    ///
    /// * `host_hidden`: `[B, T, H]` post-norm hidden state from the host
    ///   model. Typically T=1 during decode.
    /// * `prev_token_emb`: `[B, T, H]` embedding of the previously-committed
    ///   token from the host's embed table.
    ///
    /// Returns `[B, T, H]` hidden state. The caller must apply the host LM
    /// head (via `Model::apply_lm_head`) to produce logits.
    pub fn forward(
        &mut self,
        host_hidden: &Array,
        prev_token_emb: &Array,
    ) -> Result<Array, Exception> {
        if self.layers.is_empty() {
            return Err(Exception::custom(
                "MtpHead::forward called on a stub head — no MTP weights were loaded.",
            ));
        }

        let h_norm = self.pre_fc_norm_hidden.forward(host_hidden)?;
        let e_norm = self.pre_fc_norm_embedding.forward(prev_token_emb)?;
        let cat = concatenate_axis(&[&h_norm, &e_norm], -1)?;
        let mut m = self.fc.forward(&cat)?;

        // Each MTP block runs over the input positions with a fresh KV cache
        // and no explicit mask (single-position decode is the typical case).
        for layer in self.layers.iter_mut() {
            let mut cache = vec![HybridCache::KV(KVCache::new())];
            let cache_slot = &mut cache[0];
            m = layer.forward(&m, None, cache_slot)?;
        }

        self.norm.forward(&m)
    }
}

/// True when a weight key belongs to the MTP block. Matches the suffix
/// conventions used across the Qwen3.5 / Qwen3-Next / DeepSeek MTP forks
/// and the MTPLX-Optimized-Speed sidecar:
///   * `mtp.<...>` (top-level — sidecar layout)
///   * `model.mtp.<...>` / `model.mtp_layers.<i>.<...>`
///   * `language_model.model.mtp.<...>` (VLM-prefixed)
fn is_mtp_key(k: &str) -> bool {
    k.contains(".mtp.") || k.contains(".mtp_layers.") || k.starts_with("mtp.")
}

/// Find the prefix under which `fc.weight` lives. Returns the prefix string
/// up to and including `.mtp` (e.g. `"mtp"`, `"model.mtp"`,
/// `"language_model.model.mtp"`).
fn detect_mtp_prefix(weights: &HashMap<String, Array>) -> Option<String> {
    let candidates = ["mtp", "model.mtp", "language_model.model.mtp"];
    for p in candidates.iter() {
        if weights.contains_key(&format!("{}.fc.weight", p)) {
            return Some(p.to_string());
        }
    }
    None
}

/// Attempt to load the MTP head from a flat weight map. Returns `Ok(None)`
/// (not an error) when no MTP weights are present, when the config does not
/// declare any MTP layers, or when the head can't be assembled.
pub fn load_mtp_head(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
) -> Result<Option<MtpHead>, mlx_rs_core::error::Error> {
    let num_layers = args.mtp_num_hidden_layers();
    if num_layers <= 0 {
        return Ok(None);
    }

    let detected: Vec<String> = weights.keys().filter(|k| is_mtp_key(k)).cloned().collect();
    if detected.is_empty() {
        return Ok(None);
    }

    match load_mtp_head_from_weights(weights, args, &detected) {
        Ok(opt) => Ok(opt),
        Err(err) => {
            eprintln!(
                "[mtp] detected {} MTP weight keys but construction failed: {err}; \
                 falling back to AR",
                detected.len()
            );
            Ok(None)
        }
    }
}

fn load_mtp_head_from_weights(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
    detected: &[String],
) -> Result<Option<MtpHead>, mlx_rs_core::error::Error> {
    let num_layers = args.mtp_num_hidden_layers();
    let tc = &args.text_config;

    // MTP-specific quantization (group_size/bits may differ from the main
    // trunk). Falls back to the main quantization if no override is set.
    let mtp_quant = args.mtp_quantization().ok_or_else(|| {
        mlx_rs_core::error::Error::Model(
            "MTP layers need a quantization config (mtplx_mtp_quantization or top-level)".into(),
        )
    })?;
    let (group_size, bits) = (mtp_quant.group_size, mtp_quant.bits);

    let prefix = match detect_mtp_prefix(weights) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Top-level adapters: fc (plain BF16) + three RMS norms.
    let fc_w = get_weight(weights, &format!("{}.fc.weight", prefix))?;
    let fc = nn::Linear {
        weight: Param::new(fc_w),
        bias: Param::new(None),
    };

    let pre_fc_norm_embedding = load_rms_norm(
        weights,
        &format!("{}.pre_fc_norm_embedding.weight", prefix),
        tc.rms_norm_eps,
    )?;
    let pre_fc_norm_hidden = load_rms_norm(
        weights,
        &format!("{}.pre_fc_norm_hidden.weight", prefix),
        tc.rms_norm_eps,
    )?;
    let norm = load_rms_norm(
        weights,
        &format!("{}.norm.weight", prefix),
        tc.rms_norm_eps,
    )?;

    // MTP transformer blocks. Force full-attention because MTP layers are
    // always full-attention in DeepSeek-V3 / Qwen3-Next.
    let mut layers = Vec::with_capacity(num_layers as usize);
    for i in 0..num_layers {
        let layer_prefix = format!("{}.layers.{}", prefix, i);
        let block = load_transformer_block(weights, &layer_prefix, tc, group_size, bits, true)?;
        layers.push(block);
    }

    Ok(Some(MtpHead {
        num_layers,
        hidden_size: tc.hidden_size,
        use_dedicated_embeddings: args.mtp_use_dedicated_embeddings(),
        detected_weight_keys: detected.to_vec(),
        fc,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        norm,
        layers,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn detects_mtp_key_variants() {
        assert!(is_mtp_key("mtp.fc.weight"));
        assert!(is_mtp_key("mtp.layers.0.input_layernorm.weight"));
        assert!(is_mtp_key("model.mtp.fc.weight"));
        assert!(is_mtp_key("model.mtp_layers.0.self_attn.q_proj.weight"));
        assert!(is_mtp_key("language_model.model.mtp.fc.weight"));
        assert!(!is_mtp_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!is_mtp_key("model.embed_tokens.weight"));
    }

    /// Build a synthetic weight map matching the MTPLX-Optimized-Speed
    /// sidecar layout (top-level `mtp.fc` + per-layer transformer blocks)
    /// and run `forward(zeros, zeros)` end-to-end. The host config is
    /// minimal dense so we don't need MoE expert weights.
    #[test]
    fn synthetic_forward_runs_end_to_end() {
        use crate::config::{ModelArgs, QuantizationConfig, RopeParameters, TextConfig};

        // Tiny dimensions so the test runs fast.
        let hidden = 64_i32;
        let vocab = 128_i32;
        let n_heads = 4_i32;
        let n_kv = 2_i32;
        let head_dim = 16_i32;
        let q_group_size = 32_i32;
        let q_bits = 4_i32;

        let tc = TextConfig {
            hidden_size: hidden,
            num_hidden_layers: 1,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            head_dim,
            rms_norm_eps: 1e-6,
            vocab_size: vocab,
            max_position_embeddings: 4096,
            layer_types: vec!["full_attention".to_string()],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            rope_parameters: RopeParameters {
                rope_theta: 10_000.0,
                partial_rotary_factor: 0.25,
                rope_type: None,
            },
            attn_output_gate: false,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            quantization: Some(QuantizationConfig {
                group_size: q_group_size,
                bits: q_bits,
            }),
            mtp_num_hidden_layers: Some(1),
            mtp_use_dedicated_embeddings: Some(false),
        };
        let args = ModelArgs {
            text_config: tc.clone(),
            tie_word_embeddings: false,
            quantization: Some(QuantizationConfig {
                group_size: q_group_size,
                bits: q_bits,
            }),
            vision_config: None,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            language_model_only: None,
            mtp_num_hidden_layers: Some(1),
            mtp_use_dedicated_embeddings: Some(false),
            mtplx_mtp_quantization: Some(QuantizationConfig {
                group_size: q_group_size,
                bits: q_bits,
            }),
        };

        let intermediate = 2 * hidden;

        let mut w: HashMap<String, Array> = HashMap::new();
        let insert_qlinear = |w: &mut HashMap<String, Array>, prefix: &str, out: i32, inp: i32| {
            let dense = Array::zeros::<f32>(&[out, inp])
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();
            let (packed, scales, biases) = mlx_rs::ops::quantize(
                &dense,
                q_group_size,
                q_bits,
                None::<&'static str>,
            )
            .unwrap();
            w.insert(format!("{}.weight", prefix), packed);
            w.insert(format!("{}.scales", prefix), scales);
            w.insert(format!("{}.biases", prefix), biases);
        };
        let insert_rms = |w: &mut HashMap<String, Array>, key: &str, dim: i32| {
            let arr = Array::ones::<f32>(&[dim])
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();
            w.insert(key.to_string(), arr);
        };
        let insert_plain = |w: &mut HashMap<String, Array>, key: &str, shape: &[i32]| {
            let arr = Array::zeros::<f32>(shape)
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();
            w.insert(key.to_string(), arr);
        };

        // Top-level adapters.
        insert_plain(&mut w, "mtp.fc.weight", &[hidden, 2 * hidden]);
        insert_rms(&mut w, "mtp.pre_fc_norm_embedding.weight", hidden);
        insert_rms(&mut w, "mtp.pre_fc_norm_hidden.weight", hidden);
        insert_rms(&mut w, "mtp.norm.weight", hidden);

        // One MTP transformer layer (full attention + dense MLP).
        let lp = "mtp.layers.0";
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.q_proj", lp),
            n_heads * head_dim * 2,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.k_proj", lp),
            n_kv * head_dim,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.v_proj", lp),
            n_kv * head_dim,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.o_proj", lp),
            hidden,
            n_heads * head_dim,
        );
        insert_rms(&mut w, &format!("{}.self_attn.q_norm.weight", lp), head_dim);
        insert_rms(&mut w, &format!("{}.self_attn.k_norm.weight", lp), head_dim);
        insert_rms(&mut w, &format!("{}.input_layernorm.weight", lp), hidden);
        insert_rms(
            &mut w,
            &format!("{}.post_attention_layernorm.weight", lp),
            hidden,
        );
        insert_qlinear(&mut w, &format!("{}.mlp.gate_proj", lp), intermediate, hidden);
        insert_qlinear(&mut w, &format!("{}.mlp.up_proj", lp), intermediate, hidden);
        insert_qlinear(&mut w, &format!("{}.mlp.down_proj", lp), hidden, intermediate);

        let _ = vocab; // unused — caller now applies host LM head.

        let mut head = load_mtp_head(&w, &args)
            .expect("load_mtp_head failed")
            .expect("load_mtp_head returned None on a fully populated synthetic weight map");
        assert!(!head.is_stub());
        assert_eq!(head.num_layers, 1);

        let hidden_arr = Array::zeros::<f32>(&[1, 1, hidden])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let emb_arr = Array::zeros::<f32>(&[1, 1, hidden])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();

        let out = head.forward(&hidden_arr, &emb_arr).expect("forward failed");
        let shape = out.shape();
        assert_eq!(shape, &[1, 1, hidden], "unexpected hidden shape: {:?}", shape);
    }

    /// Load the real MTPLX sidecar from disk and verify a forward pass.
    /// Ignored by default since the checkpoint isn't guaranteed to be on CI.
    /// Run with: `cargo test --release -p qwen3-6-mlx mtp::tests::loads_real_mtplx_sidecar -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn loads_real_mtplx_sidecar() {
        use std::path::PathBuf;

        let model_dir = PathBuf::from("../models/Qwen3.6-27B-MTPLX-Optimized-Speed");
        if !model_dir.join("mtp.safetensors").exists() {
            // Try a sibling layout.
            let alt = PathBuf::from("models/Qwen3.6-27B-MTPLX-Optimized-Speed");
            if !alt.join("mtp.safetensors").exists() {
                eprintln!("skipping: mtp.safetensors not found at {:?}", model_dir);
                return;
            }
        }
        let real_dir = if model_dir.join("mtp.safetensors").exists() {
            model_dir
        } else {
            PathBuf::from("models/Qwen3.6-27B-MTPLX-Optimized-Speed")
        };

        let config_path = real_dir.join("config.json");
        let cfg_json = std::fs::read_to_string(&config_path).expect("read config.json");
        let args: ModelArgs = serde_json::from_str(&cfg_json).expect("parse config");

        let weights = Array::load_safetensors(&real_dir.join("mtp.safetensors"))
            .expect("load mtp.safetensors");
        let weights_map: HashMap<String, Array> = weights.into_iter().collect();

        let mut head = load_mtp_head(&weights_map, &args)
            .expect("load_mtp_head errored")
            .expect("load_mtp_head returned None on the real sidecar");
        assert!(!head.is_stub());
        assert_eq!(head.num_layers, args.mtp_num_hidden_layers());

        let h = head.hidden_size;
        let hidden_arr = Array::zeros::<f32>(&[1, 1, h])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let emb_arr = Array::zeros::<f32>(&[1, 1, h])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let out = head.forward(&hidden_arr, &emb_arr).expect("forward failed");
        assert_eq!(out.shape(), &[1, 1, h]);
    }
}
