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
    pub axis: Option<i32>,
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
                condition.raw_data_unsafe() as *const bool,
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

#[cfg(test)]
mod test {
    use std::mem::ManuallyDrop;
    use std::ptr;

    use crate::operators::compress::Compress;
    use crate::operators::infer::OutputInference;
    use crate::{DimExpr, Shape, Tensor};
    use common::DataType;

    use smallvec::smallvec;

    #[test]
    fn test_compress() {
        let compress_op = Compress { axis: Some(1) };
        let input = Tensor::without_data(
            DataType::F64,
            Shape(smallvec![
                DimExpr::Value(2),
                DimExpr::Value(3),
                DimExpr::Value(2)
            ]),
        );
        let mut condition_data = vec![true, false, true];
        let condition = Tensor::with_data(
            DataType::BOOL,
            Shape(smallvec![DimExpr::Value(3)]),
            condition_data.as_mut_ptr() as *mut u8,
        );
        let res = compress_op
            .infer([input.into(), condition.into()].as_ref())
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(
            res[0].as_ref().shape().clone(),
            Shape(smallvec![
                DimExpr::Value(2),
                DimExpr::Value(2),
                DimExpr::Value(2)
            ])
        );
    }
}
