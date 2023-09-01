use super::{InferError, InferResult, OperatorInference};
use crate::edge::Edge;

pub struct UnaryOperator {
    unary_type: UnaryType,
}

impl OperatorInference for UnaryOperator {
    fn infer(&self, inputs: &Vec<Edge>) -> InferResult {
        // Check if the input size is 1.
        if inputs.len() != 1 {
            return Err(InferError::SizeMismatch);
        }

        // Check tensor data type.
        let data_type = inputs[0].tensor().unwrap().data_type();
        match self.unary_type {
            UnaryType::Abs | UnaryType::Relu | UnaryType::PRelu => {
                if !data_type.is_numeric() {
                    return Err(InferError::DataTypeMismatch);
                }
            }
            UnaryType::Acos
            | UnaryType::Acosh
            | UnaryType::Asin
            | UnaryType::Asinh
            | UnaryType::Atan
            | UnaryType::Atanh
            | UnaryType::Cos
            | UnaryType::Cosh
            | UnaryType::Sin
            | UnaryType::Shinh
            | UnaryType::Tan => {
                if !data_type.is_ieee754() {
                    return Err(InferError::DataTypeMismatch);
                }
            }
            UnaryType::Tanh => {
                if !data_type.is_float() {
                    return Err(InferError::DataTypeMismatch);
                }
            }
        }

        Ok(inputs.clone())
    }
}

pub enum UnaryType {
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
