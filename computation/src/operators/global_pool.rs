use super::{
    infer::{InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::{Edge, Shape, Tensor};
use smallvec::smallvec;

/// 全图池化算子。
#[derive(PartialEq, Eq, Debug)]
pub(super) struct GlobalPool {
    pub ty: GlobalPoolOpType,
}

impl_op!(GlobalPool);

impl OutputInference for GlobalPool {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 1 {
            Err(InferError::InputsLenMismatch)
        } else {
            let input = inputs[0].as_ref();
            let dt = input.data_type();
            if !dt.is_ieee754() {
                return Err(InferError::DataTypeMismatch);
            }
            let shape = &input.shape().0;
            if shape.len() < 2 {
                return Err(InferError::ShapeMismatch);
            }
            let mut ans = Shape(smallvec![1.into(); shape.len()]);
            ans.0[..2].clone_from_slice(&shape[..2]);
            Ok(vec![Edge::new(Tensor::without_data(dt, ans))])
        }
    }
}

/// 全图池化运算类型。
#[derive(PartialEq, Eq, Debug)]
pub enum GlobalPoolOpType {}
