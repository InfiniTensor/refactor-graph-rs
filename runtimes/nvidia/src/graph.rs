use crate::{
    driver::{self, ContextGuard},
    kernel::{GraphBuilder, GraphUser, Resources},
};
use stack_calculator::{flat, unidir, RealtimeCalculator};
use std::{alloc::Layout, collections::BTreeSet, sync::Arc};

pub struct Graph {
    ctx: Arc<driver::Context>,
    executable: driver::ExecutableGraph,
    #[allow(unused)] // stay here to keep resource lifetime
    resources: Resources,
    static_mem: driver::DevicePtr,
    stack: driver::DevicePtr,
    offsets: graph_topo::Graph<usize, MemOffset>,
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
            unsafe { self.executable.launch_on(&stream) }
        })
    }

    #[inline]
    pub fn copy_in_one<T>(&mut self, i: usize, data: &[T]) {
        let i = self.offsets.topology.global_inputs().nth(i).unwrap();
        let offset = self.offsets.edges[i].offset();
        self.ctx.apply(|ctx| unsafe {
            self.static_mem.copy_in(offset, data, ctx);
        });
    }

    #[inline]
    pub fn copy_out_one<T>(&mut self, i: usize, data: &mut [T]) {
        let i = self.offsets.topology.global_outputs()[i];
        let offset = self.offsets.edges[i as usize].offset();
        self.ctx.apply(|ctx| unsafe {
            self.static_mem.copy_out(offset, data, ctx);
        });
    }

    #[inline]
    pub fn copy_in<'a, I, T: 'a>(&mut self, data: I)
    where
        I: IntoIterator<Item = (&'a usize, &'a [T])>,
    {
        let start = self.offsets.topology.global_inputs().start;
        self.ctx.apply(|ctx| {
            let stream = ctx.stream();
            for (i, data) in data {
                let offset = self.offsets.edges[start + i].offset();
                unsafe { self.static_mem.copy_in_async(offset, data, &stream) };
            }
        });
    }

    #[inline]
    pub fn copy_out<'a, I, T: 'a>(&mut self, data: I)
    where
        I: IntoIterator<Item = (&'a usize, &'a mut [T])>,
    {
        let global_output = self.offsets.topology.global_outputs();
        self.ctx.apply(|ctx| {
            let stream = ctx.stream();
            for (i, data) in data {
                let offset = self.offsets.edges[global_output[*i] as usize].offset();
                unsafe { self.static_mem.copy_out_async(offset, data, &stream) };
            }
        });
    }
}

#[allow(non_camel_case_types)]
type urc = u16;
const STATIC: urc = urc::MAX;
const CUDA_ALIGN: usize = 256;

impl ContextGuard<'_> {
    pub fn runtime_graph(&self, src: &computation::Graph) -> Graph {
        let mut static_mem: flat::RealtimeCalculator = flat::RealtimeCalculator::default();
        let mut stack = unidir::RealtimeCalculator::default();

        let mut nodes = vec![usize::MAX; src.0.nodes.len()];
        let mut edges = vec![MemOffset::INVALID; src.0.edges.len()];
        let mut local_edges = BTreeSet::<usize>::new();

        // 计算边引用计数
        let mut edge_rc = vec![0 as urc; src.0.edges.len()];
        for edge_idx in src.0.topology.connections() {
            edge_rc[edge_idx] += 1;
        }

        // 为输入输出分配静态存储区
        src.0
            .topology
            .global_inputs()
            .chain(src.0.topology.global_outputs())
            .for_each(|edge_idx| {
                alloc_static(src, edge_idx, &mut edges, &mut edge_rc, &mut static_mem)
            });

        // 计算工作空间需求，分配栈空间
        let mut builders = Vec::<Box<dyn GraphBuilder>>::with_capacity(src.0.nodes.len());
        let mut resources = Resources::default();
        for (node_idx, inputs, outputs) in &src.0.topology {
            let (op, _) = &src.0.nodes[node_idx];
            let builder = op.builder(&mut resources, self);
            let workspace = builder.worksapce().align_to(CUDA_ALIGN).unwrap();
            builders.push(builder);

            // alloc for outputs
            for edge_idx in outputs.clone() {
                if edge_rc[edge_idx] != STATIC {
                    alloc_stack(src, edge_idx, &mut edges, &mut stack);
                }
            }
            // alloc for workspaces
            alloc_workspace(workspace, node_idx, &mut nodes, &mut stack);
            // free for temp outputs
            for edge_idx in outputs {
                if edge_rc[edge_idx] == 0 {
                    free_stack(src, edge_idx, &edges[edge_idx], &mut stack);
                }
            }
            // free for inputs or alloc for local static inputs
            for edge_idx in inputs {
                let offset = edges[edge_idx];
                if offset == MemOffset::INVALID {
                    local_edges.insert(edge_idx);
                    alloc_static(src, edge_idx, &mut edges, &mut edge_rc, &mut static_mem);
                } else {
                    let rc = &mut edge_rc[edge_idx];
                    debug_assert_ne!(*rc, 0);
                    *rc -= 1;
                    if *rc == 0 {
                        free_stack(src, edge_idx, &offset, &mut stack);
                    }
                }
            }
        }

        // 实际分配显存空间
        let resources = resources;
        let edges = edges;
        let (static_mem, stack) = {
            let stream = self.stream();

            let mut static_mem = stream.malloc(static_mem.peak());
            let stack = stream.malloc(stack.peak());

            for edge_idx in local_edges {
                let offset = edges[edge_idx].offset();
                let tensor = &src.0.edges[edge_idx].0;
                let ptr = tensor.blob.as_ref().unwrap().get().cast::<u8>();
                let len = tensor.blob_mem_layout().size();
                unsafe {
                    let data = std::slice::from_raw_parts(ptr, len);
                    static_mem.copy_in_async(offset, data, &stream);
                }
            }

            (static_mem, stack)
        };

        let mut graph = driver::Graph::new();
        for (node_idx, inputs, outputs) in &src.0.topology {
            // TODO 计算实际地址
            let mut temp = Vec::with_capacity(1 + inputs.len() + outputs.len());
            temp.extend(inputs.iter().map(|i| edges[*i as usize]).map(|offset| {
                if offset.is_static() {
                    todo!()
                } else {
                    todo!()
                }
            }));
            builders[node_idx].push_to(
                &mut graph,
                &resources,
                &temp[0],
                &temp[1..][..inputs.len()],
                &temp[1 + inputs.len()..],
            )
        }

        Graph {
            ctx: self.clone_ctx(),
            executable: graph.instantiate(self),
            resources,
            static_mem,
            stack,
            offsets: graph_topo::Graph {
                topology: src.0.topology.clone(),
                nodes,
                edges,
            },
        }
    }
}

fn alloc_workspace(
    workspace: Layout,
    node_idx: usize,
    nodes: &mut [usize],
    stack: &mut unidir::RealtimeCalculator,
) {
    let workspace = stack.alloc(workspace);
    nodes[node_idx] = workspace.start;
    stack.free(workspace);
}

fn alloc_stack(
    src: &computation::Graph,
    edge_idx: usize,
    edges: &mut [MemOffset],
    calculator: &mut unidir::RealtimeCalculator,
) {
    let layout = src.0.edges[edge_idx]
        .0
        .blob_mem_layout()
        .align_to(CUDA_ALIGN)
        .unwrap();
    let offset = calculator.alloc(layout).start;
    edges[edge_idx] = MemOffset::from_stack(offset);
}

fn free_stack(
    src: &computation::Graph,
    edge_idx: usize,
    offset: &MemOffset,
    calculator: &mut unidir::RealtimeCalculator,
) {
    let start = offset.offset();
    let len = src.0.edges[edge_idx].0.blob_mem_layout().size();
    calculator.free(start..start + len);
}

fn alloc_static(
    src: &computation::Graph,
    edge_idx: usize,
    edges: &mut [MemOffset],
    edge_rc: &mut [urc],
    calculator: &mut flat::RealtimeCalculator,
) {
    let layout = src.0.edges[edge_idx]
        .0
        .blob_mem_layout()
        .align_to(CUDA_ALIGN)
        .unwrap();
    let offset = calculator.alloc(layout).start;
    edges[edge_idx] = MemOffset::from_static(offset);
    edge_rc[edge_idx] = STATIC;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct MemOffset(usize);

impl MemOffset {
    const INVALID: Self = Self(usize::MAX);
    const BIT: usize = 1 << (usize::BITS - 1);

    #[inline]
    const fn from_static(offset: usize) -> Self {
        Self(offset)
    }

    #[inline]
    const fn from_stack(offset: usize) -> Self {
        Self(offset | Self::BIT)
    }

    #[inline]
    const fn is_static(self) -> bool {
        self.0 & Self::BIT == 0
    }

    #[inline]
    fn offset(self) -> usize {
        debug_assert_ne!(self, Self::INVALID);
        self.0 & !Self::BIT
    }
}
