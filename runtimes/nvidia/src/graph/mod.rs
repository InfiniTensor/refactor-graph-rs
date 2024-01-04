mod builder_collector;
mod mem_offset;

use crate::{
    driver::{self, ContextGuard},
    kernel::Resources,
};
use builder_collector::BuilderCollector;
use mem_offset::MemOffset;
use stack_calculator::{flat, unidir, RealtimeCalculator};
use std::{collections::BTreeSet, sync::Arc};

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
            (*self.static_mem + offset).copy_in(data, ctx);
        });
    }

    #[inline]
    pub fn copy_out_one<T>(&mut self, i: usize, data: &mut [T]) {
        let i = self.offsets.topology.global_outputs()[i];
        let offset = self.offsets.edges[i as usize].offset();
        self.ctx.apply(|ctx| unsafe {
            (*self.static_mem + offset).copy_out(data, ctx);
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
                unsafe { (*self.static_mem + offset).copy_in_async(data, &stream) };
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
                unsafe { (*self.static_mem + offset).copy_out_async(data, &stream) };
            }
        });
    }
}

impl ContextGuard<'_> {
    pub fn runtime_graph(&self, src: &computation::Graph) -> Graph {
        let mut nodes = vec![usize::MAX; src.0.nodes.len()];
        let mut edges = vec![MemOffset::INVALID; src.0.edges.len()];
        let mut local_edges = BTreeSet::<usize>::new();

        // 计算边引用计数
        let mut edge_rc = src.0.edge_rc();

        let mut static_mem: flat::RealtimeCalculator = flat::RealtimeCalculator::default();
        let mut stack = unidir::RealtimeCalculator::default();

        // 计算工作空间需求，分配栈空间
        let mut builders = BuilderCollector::new(src);
        for (node_idx, inputs, outputs) in &src.0.topology {
            let layouts = builders.push(node_idx, &inputs, &outputs, self);
            let (workspace, layouts) = layouts.split_first().unwrap();
            let (inputs_, outputs_) = layouts.split_at(inputs.len());
            let inputs = inputs.into_iter().zip(inputs_.iter().cloned());
            let outputs = outputs.into_iter().zip(outputs_.iter().cloned());

            // alloc for outputs
            for (edge_idx, layout) in outputs.clone() {
                edges[edge_idx] = if edge_rc[edge_idx].is_global() {
                    MemOffset::from_static(static_mem.alloc(layout).start)
                } else {
                    MemOffset::from_stack(stack.alloc(layout).start)
                };
            }
            // alloc for workspaces
            {
                let workspace = stack.alloc(*workspace);
                nodes[node_idx] = workspace.start;
                stack.free(workspace);
            }
            // free for temp outputs
            for (edge_idx, layout) in outputs {
                if edge_rc[edge_idx].is_free() {
                    let start = edges[edge_idx].offset();
                    stack.free(start..start + layout.size());
                }
            }
            // free for inputs or alloc for local static inputs
            for (edge_idx, layout) in inputs {
                let offset = edges[edge_idx];
                if offset.is_invalid() {
                    local_edges.insert(edge_idx);
                    edges[edge_idx] = MemOffset::from_static(static_mem.alloc(layout).start);
                } else {
                    if edge_rc[edge_idx].free() {
                        let start = offset.offset();
                        stack.free(start..start + layout.size());
                    }
                }
            }
        }

        // 实际分配显存空间
        let (builders, resources) = builders.take();
        let edges = edges;
        let (static_mem, stack) = {
            let stream = self.stream();

            let static_mem = stream.malloc(static_mem.peak());
            let stack = stream.malloc(stack.peak());

            let global_inputs = src.0.topology.global_inputs();
            for edge_idx in local_edges {
                if !global_inputs.contains(&edge_idx) {
                    let offset = edges[edge_idx].offset();
                    let tensor = &src.0.edges[edge_idx].0;
                    let ptr = tensor.blob.as_ref().unwrap().get().cast::<u8>();
                    let len = tensor.blob_mem_layout().size();
                    unsafe {
                        let data = std::slice::from_raw_parts(ptr, len);
                        (*static_mem + offset).copy_in_async(data, &stream);
                    }
                }
            }

            (static_mem, stack)
        };

        let mut graph = driver::Graph::new();
        for (node_idx, inputs, outputs) in &src.0.topology {
            let ptrs = inputs
                .iter()
                .map(|i| *i as usize)
                .chain(outputs)
                .map(|i| edges[i])
                .map(|offset| {
                    (if offset.is_static() {
                        *static_mem
                    } else {
                        *stack
                    }) + offset.offset()
                })
                .collect::<Vec<_>>();
            let workspace = *stack + nodes[node_idx];
            let (inputs, outputs) = ptrs.split_at(inputs.len());
            builders[node_idx].push_to(&mut graph, &resources, &workspace, inputs, outputs);
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
