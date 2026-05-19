use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("vision error: {0}")]
    Vision(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("image error: {0}")]
    Image(String),
    #[error("mlx error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    #[error("safetensors error: {0}")]
    Safetensors(String),
}
