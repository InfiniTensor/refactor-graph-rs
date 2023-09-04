use super::{
    infer::{InferError, InferResult, OutputInference},
    m::impl_op,
};
use crate::{DimExpr, Edge, Shape, Tensor};
use common::DataType;
use std::slice::from_raw_parts;

/// 修改形状算子。
#[derive(PartialEq, Eq, Debug)]
pub struct Reshape;

impl_op!(Reshape);

impl OutputInference for Reshape {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            Err(InferError::InputsLenMismatch)
        } else {
            let input = inputs[0].as_ref();
            let shape = inputs[1].as_ref();
            if !matches!(shape.data_type(), DataType::I64) {
                return Err(InferError::DataTypeMismatch);
            }
            if shape.shape().0.len() != 1 {
                return Err(InferError::ShapeMismatch);
            }
            if !shape.has_data() {
                todo!();
            }
            let shape = {
                let DimExpr::Value(shape_shape) = shape.shape().0[0] else {
                    todo!()
                };
                unsafe { from_raw_parts(shape.raw_data_unsafe() as *const i64, shape_shape as _) }
            };
            let mut pos_neg1 = -1isize;
            let mut ans = Shape(smallvec::smallvec![0.into(); shape.len()]);
            for (i, &dim) in shape.iter().enumerate() {
                match dim {
                    -1 => {
                        if pos_neg1 >= 0 {
                            return Err(InferError::ShapeMismatch);
                        }
                        pos_neg1 = i as _;
                        ans.0[i] = 1.into();
                    }
                    0 => {
                        if i >= input.shape().0.len() {
                            return Err(InferError::ShapeMismatch);
                        }
                        ans.0[i] = input.shape().0[i].clone();
                    }
                    _ => {
                        ans.0[i] = dim.into();
                    }
                }
            }
            let old = input
                .shape()
                .0
                .iter()
                .map(|d| match d {
                    &DimExpr::Value(val) => val,
                    DimExpr::Variable(_) => todo!(),
                })
                .product::<i64>();
            let new = ans
                .0
                .iter()
                .map(|d| match d {
                    &DimExpr::Value(val) => val,
                    DimExpr::Variable(_) => todo!(),
                })
                .product::<i64>();
            if pos_neg1 >= 0 {
                if old % new != 0 {
                    return Err(InferError::ShapeMismatch);
                } else {
                    ans.0[pos_neg1 as usize] = (old / new).into();
                }
            } else if old != new {
                return Err(InferError::ShapeMismatch);
            }
            Ok(vec![Edge::new(Tensor::without_data(
                input.data_type(),
                ans,
            ))])
        }
    }
}
