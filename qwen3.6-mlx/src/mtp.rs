//! Multi-Token Prediction (MTP) head for Qwen3.6.
//!
//! Qwen3.6 declares `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings`
//! in its config (`text_config.mtp_num_hidden_layers: 1` for the 35B-A3B
//! checkpoint). The published `mlx-community` checkpoints strip the MTP
//! weights — see `mlx_lm.models.qwen3_5.Qwen3_5MoEModel.sanitize`, which
//! drops every key containing `"mtp."` — so most loads will find no
//! weights and return `None`.
//!
//! Architecture (per DeepSeek-V3 MTP and Qwen3-Next references):
//!
//! ```text
//!   prev_token_emb  ──► enorm ─┐
//!   host_hidden     ──► hnorm ─┴─► concat[-1] ─► eh_proj (2H → H)
//!                                                  │
//!                                                  ▼
//!                                       TransformerBlock × num_layers
//!                                                  │
//!                                                  ▼
//!                                       shared_head.norm (RMSNorm)
//!                                                  │
//!                                                  ▼  matmul shared_head.head.T
//!                                                  ▼
//!                                                logits [B,1,V]
//! ```
//!
//! Each MTP layer reuses the host model's standard `TransformerBlock`
//! (`input_layernorm` + full attention + `post_attention_layernorm` + MoE
//! or dense MLP). MTP layers are full-attention by convention even when
//! the host has a hybrid layer schedule.
//!
//! Weight-key naming is best-effort: stock Qwen3.6 / Qwen3-Next strips
//! these so we don't have a canonical layout. We try, in order:
//!
//!   1. `mtp.layers.<i>.<...>`              (DeepSeek-V3 style, top-level)
//!   2. `model.mtp.layers.<i>.<...>`        (host-prefixed DeepSeek-V3)
//!   3. `model.mtp_layers.<i>.<...>`        (Qwen3-Next sanitize convention)
//!   4. `language_model.model.mtp_layers.<i>.<...>` (VLM-prefixed)
//!
//! The `shared_head` projection is optional: when absent we fall back to
//! the host's `lm_head` / tied embedding via the caller.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    module::Module,
    nn,
    ops::concatenate_axis,
    Array,
};
use mlx_rs_core::cache::KVCache;

use crate::cache::HybridCache;
use crate::config::ModelArgs;
use crate::model::{
    get_weight, load_rms_norm, load_transformer_block, make_quantized_linear, TransformerBlock,
};

/// One MTP decoder layer: pre-block adapters + transformer block.
///
/// The transformer block is shape-compatible with the host model's
/// `TransformerBlock` and is loaded via `load_transformer_block` with
/// `force_full_attention = true`.
pub struct MtpDecoderLayer {
    pub enorm: nn::RmsNorm,
    pub hnorm: nn::RmsNorm,
    pub eh_proj: mlx_rs::quantization::MaybeQuantized<nn::Linear>,
    pub block: TransformerBlock,
}

/// MTP head for drafting `num_layers` extra speculative tokens per cycle.
///
/// For Qwen3.6's `mtp_num_hidden_layers == 1` config this is one
/// `MtpDecoderLayer` followed by `shared_head_norm` + projection.
pub struct MtpHead {
    /// Number of MTP layers (matches `mtp_num_hidden_layers`).
    pub num_layers: i32,
    /// Hidden size (matches the host model).
    pub hidden_size: i32,
    /// Whether this head carries its own embedding table; if false the
    /// caller should pass embeddings from the host model.
    pub use_dedicated_embeddings: bool,
    /// Names of weight keys that were detected in the checkpoint. Empty
    /// when no MTP weights were present (stub mode).
    pub detected_weight_keys: Vec<String>,
    /// Real MTP layers. When this is empty the head is a stub.
    pub layers: Vec<MtpDecoderLayer>,
    /// Final RMSNorm before the LM projection.
    pub shared_head_norm: Option<nn::RmsNorm>,
    /// LM head weight `[vocab, hidden]`. When `None` the caller is
    /// expected to project via the host model's tied embedding.
    pub shared_head_weight: Option<Array>,
}

impl MtpHead {
    /// True when no real MTP weights were found in the checkpoint.
    pub fn is_stub(&self) -> bool {
        self.layers.is_empty()
    }

    /// Draft the next-token logits given the last hidden state and the
    /// previously-emitted token's embedding.
    ///
    /// * `host_hidden`: `[B, 1, H]` last hidden state of the target model.
    /// * `prev_token_emb`: `[B, 1, H]` embedding of the previously committed
    ///   token (from the host embed table when `use_dedicated_embeddings` is
    ///   false).
    ///
    /// Returns logits `[B, 1, vocab]` for the K-th-ahead token (K = layers).
    ///
    /// For `num_layers > 1` the layers are composed iteratively: each MTP
    /// layer's output replaces `host_hidden` for the next layer (we don't
    /// have a sample step in between, so this is a "deep" rather than
    /// "wide" unroll; matches DeepSeek-V3's behavior when the head is used
    /// without re-embedding).
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

        let mut h = host_hidden.clone();
        let e = prev_token_emb.clone();

        for layer in self.layers.iter_mut() {
            let h_norm = layer.hnorm.forward(&h)?;
            let e_norm = layer.enorm.forward(&e)?;
            let cat = concatenate_axis(&[&h_norm, &e_norm], -1)?;
            let m = layer.eh_proj.forward(&cat)?;

            // The MTP block runs over a single position with no prior
            // context — a fresh KV cache, no mask. The block itself is
            // shape-compatible with the host model's TransformerBlock.
            let mut cache = vec![HybridCache::KV(KVCache::new())];
            // `cache[0]` is the only entry; pass &mut directly.
            let cache_slot = &mut cache[0];
            h = layer.block.forward(&m, None, cache_slot)?;
        }

        // Optional final RMSNorm + LM projection.
        let h = if let Some(norm) = self.shared_head_norm.as_mut() {
            norm.forward(&h)?
        } else {
            h
        };

        match self.shared_head_weight.as_ref() {
            Some(w) => {
                // logits = h @ w.T
                mlx_rs::ops::matmul(&h, &w.t())
            }
            None => Err(Exception::custom(
                "MtpHead::forward: shared_head_weight is None — \
                 caller should run the host LM head instead, but that path \
                 is not wired through this method yet.",
            )),
        }
    }
}

/// Attempt to load the MTP head from a flat weight map.
///
/// Returns `Ok(None)` (not an error) when:
///   * the config does not declare any MTP layers, OR
///   * no `mtp.*` / `model.mtp_layers.*` keys are present, OR
///   * the required keys for the documented layout are not all present.
///
/// Most released Qwen3.6 checkpoints strip the MTP block, so the common
/// path is `Ok(None)`.
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

    load_mtp_head_from_weights(weights, args, &detected).or_else(|err| {
        eprintln!(
            "[mtp] detected {} MTP weight keys but construction failed: {err}; \
             falling back to AR",
            detected.len()
        );
        Ok(None)
    })
}

/// Try each documented prefix convention and return the first that has the
/// canonical `enorm` weight key under `layer 0`. Returns `None` if none
/// matched.
fn detect_mtp_prefix(weights: &HashMap<String, Array>) -> Option<String> {
    let candidates = [
        "mtp.layers".to_string(),
        "model.mtp.layers".to_string(),
        "model.mtp_layers".to_string(),
        "language_model.model.mtp_layers".to_string(),
        "language_model.model.mtp.layers".to_string(),
    ];
    for prefix in candidates.iter() {
        let probe = format!("{}.0.enorm.weight", prefix);
        if weights.contains_key(&probe) {
            return Some(prefix.clone());
        }
    }
    None
}

fn load_mtp_head_from_weights(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
    detected: &[String],
) -> Result<Option<MtpHead>, mlx_rs_core::error::Error> {
    let num_layers = args.mtp_num_hidden_layers();
    let tc = &args.text_config;
    let quant = args
        .quantization()
        .ok_or_else(|| mlx_rs_core::error::Error::Model("MTP needs quantization config".into()))?;
    let (group_size, bits) = (quant.group_size, quant.bits);

    let prefix = match detect_mtp_prefix(weights) {
        Some(p) => p,
        None => return Ok(None),
    };

    let mut layers = Vec::with_capacity(num_layers as usize);
    for i in 0..num_layers {
        let layer_prefix = format!("{}.{}", prefix, i);

        let enorm =
            load_rms_norm(weights, &format!("{}.enorm.weight", layer_prefix), tc.rms_norm_eps)?;
        let hnorm =
            load_rms_norm(weights, &format!("{}.hnorm.weight", layer_prefix), tc.rms_norm_eps)?;

        let eh_proj = mlx_rs::quantization::MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.eh_proj", layer_prefix),
            group_size,
            bits,
        )?);

        // The MTP transformer block. Force full-attention because MTP
        // layers in DeepSeek-V3 / Qwen3-Next are always full-attention,
        // and we have no separate layer_types list for them.
        let block = load_transformer_block(weights, &layer_prefix, tc, group_size, bits, true)?;

        layers.push(MtpDecoderLayer {
            enorm,
            hnorm,
            eh_proj,
            block,
        });
    }

    // `shared_head` is logically attached to the last MTP layer but the
    // weight layout sometimes places it at the layer prefix (DeepSeek-V3)
    // and sometimes at the head prefix. Try both.
    let last_layer_prefix = format!("{}.{}", prefix, num_layers - 1);
    let head_prefix_candidates = [
        format!("{}.shared_head", last_layer_prefix),
        format!(
            "{}.shared_head",
            prefix.trim_end_matches(".layers").trim_end_matches("_layers")
        ),
    ];

    let mut shared_head_norm = None;
    let mut shared_head_weight = None;
    for hp in head_prefix_candidates.iter() {
        let norm_key = format!("{}.norm.weight", hp);
        if weights.contains_key(&norm_key) {
            shared_head_norm = Some(load_rms_norm(weights, &norm_key, tc.rms_norm_eps)?);

            // Head projection may be quantized (.weight + .scales + .biases)
            // or plain (.weight only). Prefer quantized when present.
            let head_w_key = format!("{}.head.weight", hp);
            let head_s_key = format!("{}.head.scales", hp);
            if weights.contains_key(&head_w_key) {
                if weights.contains_key(&head_s_key) {
                    let ql =
                        make_quantized_linear(weights, &format!("{}.head", hp), group_size, bits)?;
                    // Dequantize once and stash as a contiguous matrix.
                    let dq = mlx_rs::ops::dequantize(
                        &ql.inner.weight,
                        &ql.scales,
                        &ql.biases,
                        ql.group_size,
                        ql.bits,
                        None::<&str>,
                    )?;
                    shared_head_weight = Some(dq);
                } else {
                    shared_head_weight = Some(get_weight(weights, &head_w_key)?);
                }
            }
            break;
        }
    }

    Ok(Some(MtpHead {
        num_layers,
        hidden_size: tc.hidden_size,
        use_dedicated_embeddings: args.mtp_use_dedicated_embeddings(),
        detected_weight_keys: detected.to_vec(),
        layers,
        shared_head_norm,
        shared_head_weight,
    }))
}

/// True when a weight key belongs to the MTP block. Matches the suffix
/// conventions used across the Qwen3.5 / Qwen3-Next / DeepSeek MTP forks:
///   * `mtp.<...>` (top-level)
///   * `model.mtp.<...>` / `model.mtp_layers.<i>.<...>`
///   * `language_model.model.mtp.<...>` (VLM-prefixed)
///   * `<prefix>.mtp_layers.<i>.<...>`
fn is_mtp_key(k: &str) -> bool {
    k.contains(".mtp.") || k.contains(".mtp_layers.") || k.starts_with("mtp.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn detects_mtp_key_variants() {
        assert!(is_mtp_key("mtp.layers.0.input_layernorm.weight"));
        assert!(is_mtp_key("model.mtp.0.eh_proj.weight"));
        assert!(is_mtp_key("model.mtp_layers.0.shared_head.norm.weight"));
        assert!(is_mtp_key(
            "language_model.model.mtp_layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_mtp_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!is_mtp_key("model.embed_tokens.weight"));
    }

    /// Build a synthetic weight map for a single-layer MTP head and run
    /// `forward(zeros, zeros)` end-to-end. The host model config is a
    /// minimal dense-MLP config so we don't need to materialize MoE expert
    /// weights — `force_full_attention=true` is satisfied trivially.
    ///
    /// This proves the wiring is right even though no real checkpoint
    /// currently ships MTP weights.
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

        // Synthetic config: one layer total, dense MLP, one MTP layer.
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
        };

        let intermediate = 2 * hidden;

        // Helpers to insert a quantized linear's three weights with the
        // expected shapes. nn::quantize stores the packed matrix as
        // `[out, in * bits / 32]` int32 and scales/biases as `[out, in /
        // group_size]` in the activation dtype. We use bfloat16 throughout
        // to match the rest of the crate.
        let mut w: HashMap<String, Array> = HashMap::new();
        let insert_qlinear = |w: &mut HashMap<String, Array>, prefix: &str, out: i32, inp: i32| {
            // We use mlx-rs's quantize() to produce shape-correct payloads
            // from a zeros matrix. Going via quantize avoids encoding the
            // packed-int32 layout manually.
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
            // RMSNorm weight is just a [dim] vector.
            let arr = Array::ones::<f32>(&[dim])
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap();
            w.insert(key.to_string(), arr);
        };

        let mtp_prefix = "mtp.layers.0";

        // enorm / hnorm
        insert_rms(&mut w, &format!("{}.enorm.weight", mtp_prefix), hidden);
        insert_rms(&mut w, &format!("{}.hnorm.weight", mtp_prefix), hidden);

        // eh_proj: 2H -> H
        insert_qlinear(&mut w, &format!("{}.eh_proj", mtp_prefix), hidden, 2 * hidden);

        // self_attn: q -> n_heads*head_dim*2 (gated), k/v -> n_kv*head_dim
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.q_proj", mtp_prefix),
            n_heads * head_dim * 2,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.k_proj", mtp_prefix),
            n_kv * head_dim,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.v_proj", mtp_prefix),
            n_kv * head_dim,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.self_attn.o_proj", mtp_prefix),
            hidden,
            n_heads * head_dim,
        );
        insert_rms(
            &mut w,
            &format!("{}.self_attn.q_norm.weight", mtp_prefix),
            head_dim,
        );
        insert_rms(
            &mut w,
            &format!("{}.self_attn.k_norm.weight", mtp_prefix),
            head_dim,
        );

        // input/post layernorm
        insert_rms(
            &mut w,
            &format!("{}.input_layernorm.weight", mtp_prefix),
            hidden,
        );
        insert_rms(
            &mut w,
            &format!("{}.post_attention_layernorm.weight", mtp_prefix),
            hidden,
        );

        // dense MLP (since num_experts is None)
        insert_qlinear(
            &mut w,
            &format!("{}.mlp.gate_proj", mtp_prefix),
            intermediate,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.mlp.up_proj", mtp_prefix),
            intermediate,
            hidden,
        );
        insert_qlinear(
            &mut w,
            &format!("{}.mlp.down_proj", mtp_prefix),
            hidden,
            intermediate,
        );

        // shared_head: norm + plain (non-quantized) head projection.
        insert_rms(
            &mut w,
            &format!("{}.shared_head.norm.weight", mtp_prefix),
            hidden,
        );
        // Plain head weight [vocab, hidden].
        let head_w = Array::ones::<f32>(&[vocab, hidden])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        w.insert(format!("{}.shared_head.head.weight", mtp_prefix), head_w);

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

        let logits = head.forward(&hidden_arr, &emb_arr).expect("forward failed");
        let shape = logits.shape();
        assert_eq!(shape, &[1, 1, vocab], "unexpected logits shape: {:?}", shape);
    }
}
