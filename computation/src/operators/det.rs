use super::infer::{InferError, InferResult, OutputInference};
use super::m::impl_op;
use crate::{DimExpr, Edge, Shape, Tensor};

use smallvec::smallvec;

/// Det calculates determinant of a square matrix or batches of square matrices.
#[derive(Debug, Eq, PartialEq)]
pub struct Det {}

impl_op!(Det);

impl OutputInference for Det {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        assert_eq!(inputs.len(), 1);
        // 求行列式的输出，只有两维输出一个标量，三维及以上输出除最后
        // 两维外的所有维度

        let dim = &inputs[0].as_ref().shape().0;
        let rank = dim.len();
        let mut ans_dim = smallvec![];
        if rank == 1 {
            return Err(InferError::RankMismatch);
        } else if rank == 2 {
            ans_dim.push(DimExpr::Value(1));
        } else {
            for i in 0..(rank - 2) {
                ans_dim.push(dim[i].clone());
            }
        }
        let shape = Shape(ans_dim);
        Ok(vec![Edge::new(Tensor::without_data(
            inputs[0].data_type(),
            shape,
        ))])
    }
}
