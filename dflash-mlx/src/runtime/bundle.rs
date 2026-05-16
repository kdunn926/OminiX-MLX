use super::registry::ModelSupportSpec;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct RuntimeBundle {
    pub model_dir: PathBuf,
    pub support_spec: Option<ModelSupportSpec>,
}

impl RuntimeBundle {
    pub fn new(model_dir: PathBuf, support_spec: Option<ModelSupportSpec>) -> Self {
        Self {
            model_dir,
            support_spec,
        }
    }
}
