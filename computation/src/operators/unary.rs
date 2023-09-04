use super::{
    infer::{InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::Edge;

/// 单目算子。
#[derive(PartialEq, Eq, Debug)]
pub struct Unary {
    ty: UnaryOpType,
}

impl_op!(Unary);

impl OutputInference for Unary {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 1 {
            Err(InferError::ShapeMismatch)
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
pub enum UnaryOpType {
    /// See <https://onnx.ai/onnx/operators/onnx__Abs.html>.
    Abs,
    /// See <https://onnx.ai/onnx/operators/onnx__Relu.html>.
    Relu,
    /// See <https://onnx.ai/onnx/operators/onnx__PRelu.html>.
    PRelu,
    /// See <https://onnx.ai/onnx/operators/onnx__Acos.html>.
    Acos,
    /// See <https://onnx.ai/onnx/operators/onnx__Acosh.html>.
    Acosh,
    /// See <https://onnx.ai/onnx/operators/onnx__Asin.html>.
    Asin,
    /// See <https://onnx.ai/onnx/operators/onnx__Asinh.html>.
    Asinh,
    /// See <https://onnx.ai/onnx/operators/onnx__Atan.html>.
    Atan,
    /// See <https://onnx.ai/onnx/operators/onnx__Atanh.html>.
    Atanh,
    /// See <https://onnx.ai/onnx/operators/onnx__Cos.html>.
    Cos,
    /// See <https://onnx.ai/onnx/operators/onnx__Cosh.html>.
    Cosh,
    /// See <https://onnx.ai/onnx/operators/onnx__Sin.html>.
    Sin,
    /// See <https://onnx.ai/onnx/operators/onnx__Sinh.html>.
    Shinh,
    /// See <https://onnx.ai/onnx/operators/onnx__Tan.html>.
    Tan,
    /// See <https://onnx.ai/onnx/operators/onnx__Tanh.html>.
    Tanh,
}
