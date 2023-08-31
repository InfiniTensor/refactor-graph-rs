use crate::{Edge, Node};
use graph_topo::Searcher;

pub struct Graph {
    pub topology: Searcher,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {}
