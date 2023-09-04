use super::{
    infer::{InferResult, OutputInference},
    m::impl_op,
};
use crate::Edge;

/// See <https://onnx.ai/onnx/operators/onnx__MatMul.html>.
#[derive(PartialEq, Eq, Debug)]
pub struct MatMul;

impl_op!(MatMul);

impl OutputInference for MatMul {
    fn infer(&self, _inputs: &[Edge]) -> InferResult {
        todo!()
    }
}
