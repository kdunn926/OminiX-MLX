pub mod cache;
pub mod engine;
pub mod kernels;
pub mod model;
pub mod rollback;
pub mod runtime;
pub mod verify_qmm;

pub use engine::acceptance::match_acceptance_length;
pub use engine::config::{AdaptiveBlockPolicy, SpeculativeCycleConfig};
pub use engine::copyspec::CopySpecIndex;
pub use engine::draft_adapter::DFlashDraftAdapter;
pub use engine::qwen36_adapter::{DraftCheckpointInfo, MockDraftAdapter, Qwen36TargetAdapter};
pub use engine::spec_epoch::{
    DFlashSession, DraftBlock, DraftModel, GenerateEvent, SessionMetrics, TargetModel,
};
pub use kernels::{gated_delta_with_tape, tape_replay};
pub use model::{DFlashDraftLayer, DFlashDraftModel, DFlashDraftModelArgs};
pub use rollback::RecurrentRollbackCache;

#[cfg(test)]
pub(crate) fn mlx_test_guard() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}
