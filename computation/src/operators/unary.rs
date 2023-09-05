use super::{
    infer::{InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::Edge;

/// 单目算子。
#[derive(PartialEq, Eq, Debug)]
pub(super) struct Unary {
    pub ty: UnaryOpType,
}

impl_op!(Unary);

impl OutputInference for Unary {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 1 {
            Err(InferError::InputsLenMismatch)
        } else {
            use UnaryOpType::*;
            let data_type = inputs[0].data_type();
            match self.ty {
                Abs | Relu | PRelu if !data_type.is_numeric() => Err(InferError::DataTypeMismatch),
                Acos | Acosh | Asin | Asinh | Atan | Atanh | Cos | Cosh | Sin | Shinh | Tan
                    if !data_type.is_ieee754() =>
                {
                    Err(InferError::DataTypeMismatch)
                }
                Tanh if !data_type.is_float() => Err(InferError::DataTypeMismatch),
                _ => Ok(inputs.to_vec()),
            }
        }
    }
}

/// 单目运算类型。
#[derive(PartialEq, Eq, Debug)]
pub(super) enum UnaryOpType {
    Abs,
    Relu,
    PRelu,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Cos,
    Cosh,
    Sin,
    Shinh,
    Tan,
    Tanh,
}
