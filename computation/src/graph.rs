use crate::{Edge, Node};
use graph_topo::GraphTopoSearcher;

pub struct Graph {
    pub topology: GraphTopoSearcher,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {}
