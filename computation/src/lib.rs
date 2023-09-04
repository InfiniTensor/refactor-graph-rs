//! # Computation

// #![deny(warnings)]
#![deny(missing_docs)]
#![allow(implied_bounds_entailment)]

mod edge;
mod operators;
mod tensor;

use std::rc::Rc;

pub use operators::OpType;
pub use tensor::{DimExpr, Shape, Tensor};

extern crate common;
extern crate itertools;
extern crate smallvec;

/// 节点是一个算子，它可以有多个输入和多个输出。
///
/// 作为图表示的一种优化，具有相同信息的节点可以共享节点信息对象。
pub type Node = Rc<dyn Operator>;

/// 在优化过程中，边可能在不同子图间共享。共享的只是信息，拓扑结构是不会共享的。
pub type Edge = Rc<Tensor>;

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
