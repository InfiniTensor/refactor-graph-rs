use crate::Edge;

pub type InferResult = Result<Vec<Edge>, InferError>;

#[derive(Debug, PartialEq)]
pub enum InferError {
    SizeMismatch,
    DataTypeMismatch,
    BroadcastError,
}
