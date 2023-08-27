use crate::{Edge, Node};
use graph_topo::GraphTopo;

pub struct Graph {
    pub topology: GraphTopo,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
