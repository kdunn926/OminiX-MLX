//! Scaffold runner for GLM-OCR. Currently just loads the model and
//! reports what's parsed; real generation lands when the vision +
//! text forward paths are filled in.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/GLM-OCR"));
    eprintln!("Loading GLM-OCR scaffold from {}...", model_dir.display());
    let model = glm_ocr_mlx::load_from_path(&model_dir)?;
    println!(
        "model_type        : {}",
        model.config.model_type
    );
    println!(
        "text_config       : {} layers, hidden={} heads={}/{} mtp_layers={}",
        model.config.text_config.num_hidden_layers,
        model.config.text_config.hidden_size,
        model.config.text_config.num_attention_heads,
        model.config.text_config.num_key_value_heads,
        model.config.text_config.num_nextn_predict_layers,
    );
    println!(
        "vision_config     : {} layers, hidden={} heads={} image={} patch={} merge={} temporal={} → out_hidden={}",
        model.config.vision_config.depth,
        model.config.vision_config.hidden_size,
        model.config.vision_config.num_heads,
        model.config.vision_config.image_size,
        model.config.vision_config.patch_size,
        model.config.vision_config.spatial_merge_size,
        model.config.vision_config.temporal_patch_size,
        model.config.vision_config.out_hidden_size,
    );
    println!(
        "mrope_section     : {:?}",
        model.config.text_config.rope_parameters.mrope_section
    );
    println!(
        "image tokens (id) : start={:?} end={:?} pad={:?}",
        model.config.image_start_token_id,
        model.config.image_end_token_id,
        model.config.image_token_id,
    );
    println!(
        "preprocessor      : patch={} temp_patch={} merge={} shortest={} longest={}",
        model.preprocessor.patch_size,
        model.preprocessor.temporal_patch_size,
        model.preprocessor.merge_size,
        model.preprocessor.shortest_edge,
        model.preprocessor.longest_edge,
    );
    eprintln!(
        "scaffold OK: vision + text forwards are stubs. Generation will \
         land in a follow-up.",
    );
    Ok(())
}
