use super::{
    infer::{multidir_broadcast, InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::{Edge, Tensor};

/// 双目算子。
#[derive(PartialEq, Eq, Debug)]
pub struct Binary {
    ty: BinaryOpType,
}

impl_op!(Binary);

impl OutputInference for Binary {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            Err(InferError::InputsLenMismatch)
        } else {
            let a = inputs[0].as_ref();
            let b = inputs[1].as_ref();
            let dt = a.data_type();
            if !dt.is_numeric() || b.data_type() != dt {
                return Err(InferError::DataTypeMismatch);
            }
            let Some(shape) = multidir_broadcast(&[a.shape(), b.shape()]) else {
                return Err(InferError::BroadcastError);
            };
            Ok(vec![Edge::new(Tensor::without_data(dt, shape))])
        }
    }
}

/// 双目运算类型。
#[derive(PartialEq, Eq, Debug)]
pub enum BinaryOpType {
    /// See <https://onnx.ai/onnx/operators/onnx__Add.html>.
    Add,
    /// See <https://onnx.ai/onnx/operators/onnx__Sub.html>.
    Sub,
    /// See <https://onnx.ai/onnx/operators/onnx__Mul.html>.
    Mul,
    /// See <https://onnx.ai/onnx/operators/onnx__Div.html>.
    Div,
}
