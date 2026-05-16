//! Gemma 4 text-only inference on Apple Silicon with MLX.

pub mod chat;
pub mod vision;
pub mod model;

pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    sampler::{DefaultSampler, Sampler},
    utils::{
        create_attention_mask, create_causal_mask, scaled_dot_product_attention, AttentionMask,
        SdpaMask,
    },
};

pub use chat::{
    Gemma4ChatConfig, Gemma4ChatPipeline, Gemma4ChatResponse, Gemma4ChatTemplate,
    Gemma4Conversation, Gemma4FunctionTool, Gemma4Message, Gemma4ParsedAssistantResponse,
    Gemma4Role, Gemma4SpecialTokens, Gemma4Tool, Gemma4ToolCall, Gemma4ToolRegistry,
    Gemma4ToolResult, Gemma4ToolSpec, EOS_TOKEN_IDS,
};

pub use model::{
    get_model_args, init_cache, load_model, load_model_with_overrides, load_tokenizer,
    load_vl_model, restore_cache, snapshot_cache, Attention, AttentionInput, DecoderLayer,
    DecoderLayerInput, DenseMlp, Experts, Gemma4Config, Gemma4TextConfig, Gemma4VlModel,
    Generate, GenerateState, LanguageModel, Model, ModelInput, Router, UnscaledRmsNorm,
};
pub use vision::{preprocess_image_gemma4, EmbedVision, Gemma4VisionConfig, VisionModel};

pub struct GemmaVlChatMessage {
    pub role: String,
    pub content: String,
    pub n_vision_tokens: Option<usize>,
    pub has_image: bool,
}

pub fn build_gemma4_vl_chat_tokens(
    tokenizer: &tokenizers::Tokenizer,
    messages: &[GemmaVlChatMessage],
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Result<Vec<i32>> {
    let bos = tokenizer
        .token_to_id("<bos>")
        .ok_or_else(|| Error::Tokenizer("Missing <bos> token".into()))? as i32;
    let turn_start = tokenizer
        .token_to_id("<|turn>")
        .ok_or_else(|| Error::Tokenizer("Missing <|turn> token".into()))? as i32;
    let turn_end = tokenizer
        .token_to_id("<turn|>")
        .ok_or_else(|| Error::Tokenizer("Missing <turn|> token".into()))? as i32;
    let newline = tokenizer
        .token_to_id("\n")
        .ok_or_else(|| Error::Tokenizer("Missing newline token".into()))? as i32;

    let encode = |text: &str| -> Result<Vec<i32>> {
        let enc = tokenizer.encode(text, false)?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    };

    let mut normalized = Vec::new();
    if messages.first().map(|m| m.role.as_str()) != Some("system") {
        normalized.push(GemmaVlChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            n_vision_tokens: None,
            has_image: false,
        });
    }
    for msg in messages {
        normalized.push(GemmaVlChatMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
            n_vision_tokens: msg.n_vision_tokens,
            has_image: msg.has_image,
        });
    }

    let mut tokens = vec![bos];
    for msg in &normalized {
        tokens.push(turn_start);
        tokens.extend_from_slice(&encode(&msg.role)?);
        tokens.push(newline);
        if msg.has_image {
            tokens.push(boi_token_id as i32);
            for _ in 0..msg.n_vision_tokens.unwrap_or_default() {
                tokens.push(image_token_id as i32);
            }
            tokens.push(eoi_token_id as i32);
            tokens.push(newline);
        }
        if !msg.content.is_empty() {
            tokens.extend_from_slice(&encode(&msg.content)?);
        }
        tokens.push(turn_end);
        tokens.push(newline);
    }

    tokens.push(turn_start);
    tokens.extend_from_slice(&encode("assistant")?);
    tokens.push(newline);
    Ok(tokens)
}
