use crate::driver::{self, ContextGuard};
use computation::Tensor;
use graph_topo::GraphTopo;
use stack_calculator::{flat, unidir, RealtimeCalculator};
use std::{alloc::Layout, collections::BTreeSet, sync::Arc};

pub struct Graph {
    ctx: Arc<driver::Context>,
    graph: driver::ExecutableGraph,
    topology: GraphTopo,
    edges: Vec<MemOffset>,
    static_mem: driver::DevicePtr,
    stack: driver::DevicePtr,
}

impl Drop for Graph {
    fn drop(&mut self) {
        self.ctx.apply(|ctx| {
            ctx.free(self.static_mem.take());
            ctx.free(self.stack.take());
        });
    }
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

    #[inline]
    pub fn copy_in_one<T>(&mut self, i: usize, data: &[T]) {
        let i = self.topology.global_inputs().nth(i).unwrap();
        let offset = self.edges[i].offset();
        self.ctx.apply(|ctx| unsafe {
            self.static_mem.copy_in(offset, data, ctx);
        });
    }

    #[inline]
    pub fn copy_out_one<T>(&mut self, i: usize, data: &mut [T]) {
        let i = self.topology.global_outputs()[i];
        let offset = self.edges[i as usize].offset();
        self.ctx.apply(|ctx| unsafe {
            self.static_mem.copy_out(offset, data, ctx);
        });
    }

    #[inline]
    pub fn copy_in<'a, I, T: 'a>(&mut self, data: I)
    where
        I: IntoIterator<Item = (&'a usize, &'a [T])>,
    {
        let start = self.topology.global_inputs().start;
        self.ctx.apply(|ctx| {
            let stream = ctx.stream();
            for (i, data) in data {
                let offset = self.edges[start + i].offset();
                unsafe { self.static_mem.copy_in_async(offset, data, &stream) };
            }
        });
    }

    #[inline]
    pub fn copy_out<'a, I, T: 'a>(&mut self, data: I)
    where
        I: IntoIterator<Item = (&'a usize, &'a mut [T])>,
    {
        let global_output = self.topology.global_outputs();
        self.ctx.apply(|ctx| {
            let stream = ctx.stream();
            for (i, data) in data {
                let offset = self.edges[global_output[*i] as usize].offset();
                unsafe { self.static_mem.copy_out_async(offset, data, &stream) };
            }
        });
    }
}

impl ContextGuard<'_> {
    pub fn runtime_graph(&self, src: &computation::Graph) -> Graph {
        let src = &src.0;

        let mut static_mem = flat::RealtimeCalculator::default();
        let mut stack = unidir::RealtimeCalculator::default();

        let mut edges = vec![MemOffset::INVALID; src.edges.len()];
        let mut local_edges = BTreeSet::<usize>::new();

        #[allow(non_camel_case_types)]
        type urc = u16;
        const STATIC: urc = urc::MAX;
        let mut edge_rc = vec![0 as urc; src.edges.len()];
        for edge_idx in src.topology.connections() {
            edge_rc[edge_idx] += 1;
        }

        src.topology
            .global_inputs()
            .chain(src.topology.global_outputs())
            .for_each(|edge_idx| {
                edge_rc[edge_idx] = STATIC;
                edges[edge_idx] = MemOffset::from_static(
                    // 全图输入输出分配在静态存储区
                    static_mem.alloc(cuda_layout(&src.edges[edge_idx])).start,
                );
            });

        let mut graph = driver::Graph::new();

        for (node_idx, inputs, outputs) in &src.topology {
            let (op, _) = &src.nodes[node_idx];
            // TODO 分配栈空间，构造计算节点
        }

        let static_mem = {
            let stream = self.stream();
            let mut static_mem = self.malloc(static_mem.peak());
            for edge_idx in local_edges {
                let offset = edges[edge_idx].offset();
                let tensor = &src.edges[edge_idx].0;
                let ptr = tensor.blob.as_ref().unwrap().get().cast::<u8>();
                let len = tensor.blob_mem_layout().size();
                unsafe {
                    let data = std::slice::from_raw_parts(ptr, len);
                    static_mem.copy_in_async(offset, data, &stream);
                }
            }
            static_mem
        };

        Graph {
            ctx: self.clone_ctx(),
            graph: graph.instantiate(self),
            topology: src.topology.clone(),
            edges,
            static_mem,
            stack: self.malloc(stack.peak()),
        }
    }
}

#[inline(always)]
fn cuda_layout(edge: &(Tensor, String)) -> Layout {
    edge.0.blob_mem_layout().align_to(256).unwrap()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct MemOffset(usize);

impl MemOffset {
    const INVALID: MemOffset = MemOffset(usize::MAX);
    const BIT: usize = 1 << (usize::BITS - 1);

    fn from_static(offset: usize) -> Self {
        Self(offset | Self::BIT)
    }

    fn is_static(self) -> bool {
        self.0 & Self::BIT != 0
    }

    fn offset(self) -> usize {
        self.0 & !Self::BIT
    }
}
