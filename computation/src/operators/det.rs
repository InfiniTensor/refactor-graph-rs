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

#[cfg(test)]
mod test {
    #[test]
    fn test_det() {
        use crate::operators::OutputInference;
        use crate::{DimExpr, Edge};

        use common::DataType;

        use smallvec::smallvec;

        let det_op = super::Det {};

        let martix_2x2 = crate::Tensor::without_data(
            DataType::F64,
            crate::Shape(smallvec![DimExpr::Value(2), DimExpr::Value(2)]),
        );
        let det_2x2 = det_op.infer(&[Edge::new(martix_2x2)]).unwrap();
        assert_eq!(
            det_2x2[0].as_ref().shape(),
            &crate::Shape(smallvec![DimExpr::Value(1)])
        );

        let matrix_3x4x4x4 = crate::Tensor::without_data(
            DataType::F64,
            crate::Shape(smallvec![
                DimExpr::Value(3),
                DimExpr::Value(4),
                DimExpr::Value(4),
                DimExpr::Value(4)
            ]),
        );
        let det_3x4x4x4 = det_op.infer(&[Edge::new(matrix_3x4x4x4)]).unwrap();

        assert_eq!(
            det_3x4x4x4[0].as_ref().shape(),
            &crate::Shape(smallvec![DimExpr::Value(3), DimExpr::Value(4)])
        );
    }
}
