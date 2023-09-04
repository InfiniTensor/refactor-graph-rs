use super::{
    infer::{InferResult, OutputInference},
    m::impl_op,
};
use crate::{Edge, InferError};

/// See <https://onnx.ai/onnx/operators/onnx__MatMul.html>.
#[derive(PartialEq, Eq, Debug)]
pub struct MatMul;

impl_op!(MatMul);

impl OutputInference for MatMul {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            Err(InferError::InputsLenMismatch)
        } else {
            todo!()
        }
    }
}
