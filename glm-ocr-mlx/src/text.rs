//! Compatibility shim. The real text decoder now lives in
//! [`crate::text_decoder`] — see `Glm4OcrForCausalLM` for the forward
//! API. This module remains so the scaffold's `TextDecoder` export
//! keeps compiling for any external imports; remove in phase 5.

pub use crate::text_decoder::Glm4OcrForCausalLM as TextDecoder;
