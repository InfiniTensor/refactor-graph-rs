//! # Computation

// #![deny(warnings)]
#![deny(missing_docs)]

mod operators;
mod tensor;

use operators::Operator;
use std::rc::Rc;

pub use tensor::{DimExpr, Shape, Tensor};

/// 节点是一个算子，它可以有多个输入和多个输出。
///
/// 作为图表示的一种优化，具有相同信息的节点可以共享节点信息对象。
type Node = Rc<dyn Operator>;

/// 在优化过程中，边可能在不同子图间共享。共享的只是信息，拓扑结构是不会共享的。
type Edge = Rc<Tensor>;

/// 计算图。
pub struct Graph(pub graph_topo::Graph<Node, Edge>);
