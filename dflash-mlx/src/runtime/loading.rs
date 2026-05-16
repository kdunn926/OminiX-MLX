use super::{bundle::RuntimeBundle, registry::MODEL_SUPPORT_SPECS};
use crate::engine::qwen36_adapter::DraftCheckpointInfo;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeLoadError {
    #[error("runtime bundle path does not exist: {0}")]
    MissingPath(PathBuf),
}

pub fn load_runtime_bundle(path: impl AsRef<Path>) -> Result<RuntimeBundle, RuntimeLoadError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(RuntimeLoadError::MissingPath(path.to_path_buf()));
    }

    let model_id = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let support_spec = MODEL_SUPPORT_SPECS
        .iter()
        .find(|spec| {
            spec.draft_model_id == model_id
                || spec.target_model_ids.iter().any(|id| id == &model_id)
        })
        .cloned();

    Ok(RuntimeBundle::new(path.to_path_buf(), support_spec))
}

/// Locate a DFlash draft checkpoint that pairs with the given target model dir.
///
/// Resolution order:
/// 1. `dflash_draft_path` key inside `<target_dir>/config.json` (absolute, or
///    relative to the target dir).
/// 2. Sibling dir `<target_parent>/<target_name>-DFlash`.
/// 3. Sibling dir with the trailing quant suffix stripped, e.g. target
///    `Qwen3.6-35B-A3B-4bit` → `Qwen3.6-35B-A3B-DFlash`.
///
/// A candidate is only accepted if it contains a `config.json` whose
/// `architectures` advertises `DFlashDraftModel`.
pub fn discover_draft_for_target(target_dir: impl AsRef<Path>) -> Option<PathBuf> {
    let target_dir = target_dir.as_ref();

    if let Ok(text) = std::fs::read_to_string(target_dir.join("config.json")) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(raw) = cfg.get("dflash_draft_path").and_then(|v| v.as_str()) {
                let p = Path::new(raw);
                let resolved = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    target_dir.join(p)
                };
                if is_valid_dflash_draft(&resolved) {
                    return Some(resolved);
                }
            }
        }
    }

    let parent = target_dir.parent()?;
    let name = target_dir.file_name()?.to_str()?;

    let suffix_candidate = parent.join(format!("{name}-DFlash"));
    if is_valid_dflash_draft(&suffix_candidate) {
        return Some(suffix_candidate);
    }

    if let Some(stem) = strip_quant_suffix(name) {
        let stem_candidate = parent.join(format!("{stem}-DFlash"));
        if is_valid_dflash_draft(&stem_candidate) {
            return Some(stem_candidate);
        }
    }

    None
}

fn is_valid_dflash_draft(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    matches!(
        DraftCheckpointInfo::load(path),
        Ok(info) if info.is_native_rust_supported()
    )
}

fn strip_quant_suffix(name: &str) -> Option<&str> {
    for suffix in ["-4bit", "-8bit", "-bf16", "-fp16"] {
        if let Some(stem) = name.strip_suffix(suffix) {
            return Some(stem);
        }
    }
    None
}
