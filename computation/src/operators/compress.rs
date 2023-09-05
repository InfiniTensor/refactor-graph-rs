use std::slice::from_raw_parts;

use super::infer::{InferError, InferResult, OutputInference};
use super::m::impl_op;
use crate::{DimExpr, Edge, Shape, Tensor};

use smallvec::smallvec;

/// Selectes slices from an input tensor along a given axis where condition evaluates
/// to True for each axis index.
///
/// https://onnx.ai/onnx/operators/onnx__Compress.html
#[derive(Debug, Eq, PartialEq)]
pub struct Compress {
    /// (Optional) Axis along which to take slices. If not specified,
    /// input is flattened before elements being selected.
    /// Negative value means counting dimensions from the back.
    /// Accepted range is [-r, r-1] where r = rank(input).
    axis: Option<i32>,
    /// Array that selects which slices to be selected.
    condition: Vec<bool>,
}

impl_op!(Compress);

impl OutputInference for Compress {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            return Err(InferError::InputsLenMismatch);
        }
        let dim = &inputs[0].as_ref().shape().0;
        let rank = dim.len();
        if rank == 0 {
            return Err(InferError::RankMismatch);
        }

        let condition = inputs[1].as_ref();
        let condition_shape = condition.shape().0[0].value()?;
        let condition_data = unsafe {
            from_raw_parts(
                condition.data_ptr()?.as_ptr() as *const bool,
                condition_shape as usize,
            )
        };

        // Outputs: Tensor of rank r if axis is specified. Otherwise output is a Tensor
        // of rank r 1.

        let mut ans_dim = smallvec![];

        if let Some(axis) = self.axis {
            ans_dim = dim.clone();
            ans_dim[axis as usize] =
                DimExpr::Value(condition_data.iter().filter(|x| **x).count() as i64);
        } else {
            ans_dim.push(DimExpr::Value(1));
        }

        let shape = Shape(ans_dim);
        Ok(vec![Edge::new(Tensor::without_data(
            inputs[0].data_type(),
            shape,
        ))])
    }
}
