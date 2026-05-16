use super::{bundle::RuntimeBundle, registry::MODEL_SUPPORT_SPECS};
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
