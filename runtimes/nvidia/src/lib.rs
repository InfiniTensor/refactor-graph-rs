//! Nvidia GPU 运行时。

#![cfg(detected_cuda)]

use graph_topo::GraphTopo;

mod driver;

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
