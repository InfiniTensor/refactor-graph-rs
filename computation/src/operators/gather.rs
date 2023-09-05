use super::infer::{InferError, InferResult, OutputInference};
use super::m::impl_op;

#[derive(Debug, Eq, PartialEq)]
pub struct Gather {
    axis: usize,
}

impl_op!(Gather);

impl OutputInference for Gather {
    fn infer(&self, inputs: &[crate::Edge]) -> InferResult {
        todo!()
    }
}
