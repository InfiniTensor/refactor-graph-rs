use super::infer::{InferError, InferResult, OutputInference};
use super::m::impl_op;

#[derive(Debug, Eq, PartialEq)]
pub struct Gather {
    pub axis: usize,
}

impl_op!(Gather);

impl OutputInference for Gather {
    fn infer(&self, inputs: &[crate::Edge]) -> InferResult {
        if inputs.len() != 2 {
            return Err(InferError::InputsLenMismatch);
        }
        let shape_a = inputs[0].as_ref().shape();
        let shape_b = inputs[1].as_ref().shape();

        let mut ans_dim = shape_a.0.clone();
        ans_dim.remove(self.axis);
        ans_dim.insert_many(self.axis, shape_b.0.clone());

        Ok(vec![crate::Edge::new(crate::Tensor::without_data(
            inputs[0].data_type(),
            crate::Shape(ans_dim),
        ))])
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_gather() {
        use crate::operators::infer::OutputInference;

        use smallvec::smallvec;

        let gather_op = super::Gather { axis: 0 };
        let a = crate::Tensor::without_data(
            common::DataType::F64,
            crate::Shape(smallvec![
                crate::DimExpr::Value(3),
                crate::DimExpr::Value(3)
            ]),
        );
        let indicates = crate::Tensor::without_data(
            common::DataType::I64,
            crate::Shape(smallvec![crate::DimExpr::Value(3)]),
        );

        let gather_single_axis = gather_op
            .infer(&[crate::Edge::new(a), crate::Edge::new(indicates)])
            .unwrap();

        assert_eq!(
            gather_single_axis[0].as_ref().shape(),
            &crate::Shape(smallvec![
                crate::DimExpr::Value(3),
                crate::DimExpr::Value(3)
            ])
        );
    }
}
