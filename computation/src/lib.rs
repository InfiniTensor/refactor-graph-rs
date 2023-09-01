//! # Computation

// #![deny(warnings)]
#![deny(missing_docs)]
#![allow(implied_bounds_entailment)]

mod edge;
mod operators;

pub use operators::OpType;

extern crate common;
extern crate itertools;
extern crate smallvec;

#[cfg(test)]
mod test {
    use std::vec;

    use crate::edge::{Shape, Tensor};
    use crate::operators::{infer_multi_broadcast, ElementWise, OperatorInference};

    #[test]
    fn test_broadcast() {
        let a = Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]);
        let b = Shape::new(vec![None, None, None, None]);
        let c = infer_multi_broadcast(a, b);
        assert_eq!(
            c,
            Some(Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]))
        );

        let a = Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]);
        let b = Shape::new(vec![None, None, None, Some(5)]);
        let c = infer_multi_broadcast(a, b);
        assert_eq!(
            c,
            Some(Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]))
        );

        let a = Shape::new(vec![None, Some(1), Some(4), Some(5)]);
        let b = Shape::new(vec![Some(2), Some(3), Some(1), Some(1)]);
        let c = infer_multi_broadcast(a, b);
        assert_eq!(
            c,
            Some(Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]))
        );

        let a = Shape::new(vec![None, Some(3), Some(4), Some(5)]);
        let b = Shape::new(vec![Some(2), Some(1), Some(1), Some(1)]);
        let c = infer_multi_broadcast(a, b);
        assert_eq!(
            c,
            Some(Shape::new(vec![Some(2), Some(3), Some(4), Some(5)]))
        );

        let a = Shape::new(vec![None, None, Some(2), Some(1)]);
        let b = Shape::new(vec![None, Some(8), Some(4), Some(3)]);
        let c = infer_multi_broadcast(a, b);
        assert_eq!(c, None);
    }

    #[test]
    fn test_element_wise_op() {
        use crate::edge::{Edge, Tensor};
        use crate::operators::ElementWiseType;

        use common::DataType;

        let add_op = ElementWise::new(ElementWiseType::Add);
        let a = Shape::new(vec![None, Some(3), Some(4), Some(5)]);
        let b = Shape::new(vec![Some(2), Some(1), Some(1), Some(1)]);
        let tensor0 = Tensor::new(DataType::F64, a);
        let tensor1 = Tensor::new(DataType::F64, b);

        let edges = vec![Edge::Tensor(tensor0), Edge::Tensor(tensor1)];
        let outputs = add_op.infer(&edges);

        assert_eq!(
            outputs,
            Ok(vec![Edge::Tensor(Tensor::new(
                DataType::F64,
                Shape::new(vec![Some(2), Some(3), Some(4), Some(5)])
            ))])
        );
    }
}
