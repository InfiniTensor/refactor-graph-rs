use super::{
    infer::{InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::Edge;

/// 双目算子。
#[derive(PartialEq, Eq, Debug)]
pub struct Binary {
    ty: BinaryOpType,
}

impl_op!(Binary);

impl OutputInference for Binary {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            Err(InferError::ShapeMismatch)
        } else {
            todo!()
        }
    }
}

/// 双目运算类型。
#[derive(PartialEq, Eq, Debug)]
pub enum BinaryOpType {}
