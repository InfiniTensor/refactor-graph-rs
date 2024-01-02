use crate::driver;
use graph_topo::GraphTopo;

pub struct Graph {
    graph: driver::Graph,
    topology: GraphTopo,
    edges: Vec<MemOffset>,
    static_mem: driver::Blob,
    stack: driver::Blob,
}

enum MemOffset {
    Static(usize),
    Stack(usize),
}
