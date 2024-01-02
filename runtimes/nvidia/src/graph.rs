use crate::driver::{self, ContextGuard};
use graph_topo::GraphTopo;
use stack_calculator::{flat, unidir, RealtimeCalculator};
use std::sync::Arc;

pub struct Graph {
    ctx: Arc<driver::Context>,
    graph: driver::ExecutableGraph,
    topology: GraphTopo,
    edges: Vec<MemOffset>,
    static_mem: driver::Blob,
    stack: driver::Blob,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct MemOffset(usize);

impl MemOffset {
    const INVALID: MemOffset = MemOffset(usize::MAX);
}

impl Graph {
    #[inline]
    pub fn new(src: &computation::Graph, dev: usize) -> Self {
        driver::devices()[dev]
            .context()
            .apply(|ctx| ctx.runtime_graph(src))
    }

    #[inline]
    pub fn run(&self) {
        self.ctx.apply(|ctx| {
            let stream = ctx.stream();
            unsafe { self.graph.launch_on(&stream) }
        })
    }
}

impl ContextGuard<'_> {
    pub fn runtime_graph(&self, src: &computation::Graph) -> Graph {
        let src = &src.0;

        let mut flat = flat::RealtimeCalculator::default();
        let mut unidir = unidir::RealtimeCalculator::default();

        let mut edges = vec![MemOffset::INVALID; src.edges.len()];

        driver::init();
        let graph = driver::Graph::new();

        let mut static_mem = self.malloc(flat.peak());

        Graph {
            ctx: self.clone_ctx(),
            graph: graph.instantiate(self),
            topology: src.topology.clone(),
            edges,
            static_mem,
            stack: self.malloc(unidir.peak()),
        }
    }
}
