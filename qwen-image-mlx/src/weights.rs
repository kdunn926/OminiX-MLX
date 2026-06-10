//! Weight loading utilities for Qwen-Image
//!
//! Handles loading SafeTensors weights and mapping them to model structure

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;
use safetensors::SafeTensors;

use crate::error::QwenImageError;
use crate::transformer::{QwenTransformer, QwenTransformerConfig};
use crate::vae::QwenVAE;

/// Load SafeTensors file into a HashMap
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, Array>, QwenImageError> {
    let data = std::fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let array = Array::try_from(tensor).map_err(|e| {
            QwenImageError::WeightLoadError(format!("Failed to convert tensor {}: {:?}", name, e))
        })?;
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

/// Load multiple SafeTensors shards
pub fn load_safetensors_shards<P: AsRef<Path>>(
    paths: &[P],
) -> Result<HashMap<String, Array>, QwenImageError> {
    let mut all_weights = HashMap::new();

    for path in paths {
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }

    Ok(all_weights)
}

/// Weight name mapping for transformer
pub struct TransformerWeightMapper;

impl TransformerWeightMapper {
    /// Map HuggingFace weight names to our structure
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Common mappings
        name = name.replace(".attn1.", ".attn.");
        name = name.replace(".attn2.", ".attn.");
        name = name.replace("to_out.0.", "attn_to_out.");

        // Norm mappings

        // FFN mappings
        name = name.replace("ff.net.0.proj.", "mlp_in.");
        name = name.replace("ff.net.2.", "mlp_out.");
        name = name.replace("ff_context.net.0.proj.", "ff_context.mlp_in.");
        name = name.replace("ff_context.net.2.", "ff_context.mlp_out.");

        // Embedding mappings
        name = name.replace("x_embedder.", "patch_embed.");

        // Output mappings

        name
    }

    /// Sanitize weight names (remove prefixes, etc.)
    pub fn sanitize_weights(
        weights: HashMap<String, Array>,
    ) -> HashMap<String, Array> {
        weights
            .into_iter()
            .map(|(k, v)| (Self::map_name(&k), v))
            .collect()
    }
}

/// Weight name mapping for VAE
pub struct VAEWeightMapper;

impl VAEWeightMapper {
    /// Map HuggingFace weight names to our structure
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();

        // Encoder mappings

        // Decoder mappings

        // ResBlock mappings

        // Attention mappings
        name = name.replace(".group_norm.", ".norm.");
        name = name.replace(".to_out.0.", ".to_out.");

        name
    }

    /// Sanitize weight names
    pub fn sanitize_weights(
        weights: HashMap<String, Array>,
    ) -> HashMap<String, Array> {
        weights
            .into_iter()
            .map(|(k, v)| (Self::map_name(&k), v))
            .collect()
    }
}

/// Convert HashMap<String, Array> to HashMap<Rc<str>, Array> for update_flattened
fn to_rc_keys(weights: HashMap<String, Array>) -> HashMap<std::rc::Rc<str>, Array> {
    weights
        .into_iter()
        .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
        .collect()
}

/// Load transformer from SafeTensors
pub fn load_transformer<P: AsRef<Path>>(
    path: P,
    config: QwenTransformerConfig,
) -> Result<QwenTransformer, QwenImageError> {
    let weights = load_safetensors(path)?;
    let weights = TransformerWeightMapper::sanitize_weights(weights);
    let weights_rc = to_rc_keys(weights);

    let mut transformer = QwenTransformer::new(config)?;
    transformer.update_flattened(weights_rc);

    Ok(transformer)
}

/// Load VAE from SafeTensors
pub fn load_vae<P: AsRef<Path>>(path: P) -> Result<QwenVAE, QwenImageError> {
    let weights = load_safetensors(path)?;
    let weights = VAEWeightMapper::sanitize_weights(weights);
    let weights_rc = to_rc_keys(weights);

    let mut vae = QwenVAE::new()?;
    vae.update_flattened(weights_rc);

    Ok(vae)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_weight_mapping() {
        let name = "transformer_blocks.0.attn1.to_q.weight";
        let mapped = TransformerWeightMapper::map_name(name);
        assert!(mapped.contains("to_q"));
    }

    #[test]
    fn test_vae_weight_mapping() {
        let name = "encoder.down_blocks.0.resnets.0.norm1.weight";
        let mapped = VAEWeightMapper::map_name(name);
        assert!(mapped.contains("norm1"));
    }
}
