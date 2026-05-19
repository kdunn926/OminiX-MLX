//! `glm_ocr_text` decoder scaffold.
//!
//! Glm4-family transformer (RMS norm, SwiGLU MLP, GQA) with two
//! glm-ocr-specific bits:
//!   - **mrope** — multimodal RoPE. The text decoder must accept
//!     position ids that come from the processor's "image plays a 2D
//!     spatial grid + 1 temporal slot" mapping; `mrope_section` lists
//!     how many rotation dims belong to each axis (text / spatial-h /
//!     spatial-w / temporal). For pure-text input mrope degenerates to
//!     standard 1D RoPE.
//!   - **MTP head** — `num_nextn_predict_layers = 1` means a single
//!     extra transformer block + LM-head shortcut, used for
//!     speculative drafting (1 lookahead token per cycle).

use mlx_rs::Array;

use crate::config::GlmOcrTextConfig;
use crate::error::Error;

pub struct TextDecoder {
    pub config: GlmOcrTextConfig,
    // TODO(real impl):
    //   - embed_tokens: nn::Embedding
    //   - layers: Vec<DecoderBlock>  (RMS norm, mrope GQA, RMS norm, SwiGLU MLP)
    //   - final_norm: nn::RmsNorm
    //   - lm_head: Linear or tied embedding
    //   - mtp_head: optional single-block draft head
    _placeholder: (),
}

impl TextDecoder {
    pub fn load_from_weights(
        config: GlmOcrTextConfig,
        _weights: &std::collections::HashMap<String, Array>,
    ) -> Result<Self, Error> {
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(Error::Model(format!(
                "text.hidden_size {} not divisible by num_attention_heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }
        if let Some(sections) = config.rope_parameters.mrope_section.as_ref() {
            let sum: i32 = sections.iter().sum();
            let half = config.head_dim / 2;
            if sum != half {
                return Err(Error::Model(format!(
                    "mrope_section sums to {sum}, expected {half} (head_dim/2)"
                )));
            }
        }
        Ok(Self {
            config,
            _placeholder: (),
        })
    }

    /// Forward stub. Real impl: prefill / decode with mrope position ids.
    pub fn forward_stub(&mut self, _input_embeds: &Array) -> Result<Array, mlx_rs::error::Exception> {
        Err(mlx_rs::error::Exception::custom(
            "glm-ocr-mlx text decoder forward not yet implemented",
        ))
    }
}
