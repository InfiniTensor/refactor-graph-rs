use super::{
    eq::{Downcast, OperatorEq},
    infer::{InferResult, OutputInference},
};
use crate::Edge;
use std::any::Any;

#[derive(PartialEq, Debug)]
pub struct MatMul;

impl Downcast for MatMul {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl OperatorEq for MatMul {
    #[inline]
    fn op_eq(&self, rhs: &dyn OperatorEq) -> bool {
        rhs.as_any().is::<Self>()
    }
}

impl OutputInference for MatMul {
    fn infer(&self, _inputs: &[Edge]) -> InferResult {
        todo!()
    }
}
