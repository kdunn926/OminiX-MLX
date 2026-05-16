#[derive(Debug, Clone)]
pub struct ModelSupportSpec {
    pub target_model_ids: Vec<String>,
    pub draft_model_id: String,
    pub backend: String,
    pub w4_defaults: bool,
}

/// Initially empty; populated as target models are confirmed compatible.
pub static MODEL_SUPPORT_SPECS: &[ModelSupportSpec] = &[];
