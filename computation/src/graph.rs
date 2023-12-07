use crate::{Operator, Tensor};

/// 计算图。
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Graph(graph_topo::Graph<Operator, Tensor>);
