//! Cpu 运行时。

// #![deny(warnings, missing_docs)]

mod operator;
mod stack;

use graph_topo::OutputEdge;
use operator::lower;
use rayon::prelude::*;
use stack_calculator::Calculator;
use std::{cell::RefCell, sync::Arc, usize};

pub struct Graph {
    graph: graph_topo::Graph<RoutineWorkspace, Blob>,
    stack: Vec<u8>,
}

struct RoutineWorkspace {
    routine: Box<dyn Routine>,
    workspace: usize,
}

impl Graph {
    pub fn new(computation::Graph(src): &computation::Graph, calculator: impl Calculator) -> Self {
        let nodes = src.topology.iter().collect::<Vec<_>>();
        let mut nodes = nodes
            .into_par_iter()
            .map(|(node_idx, inputs, outputs)| {
                let mut tensors = Vec::with_capacity(inputs.len() + outputs.len());
                for &OutputEdge(i) in inputs {
                    tensors.push(&src.edges[i as usize].0);
                }
                for i in outputs {
                    tensors.push(&src.edges[i].0);
                }
                let (inputs, outputs) = tensors.split_at(inputs.len());
                lower(&src.nodes[node_idx].0, inputs, outputs)
            })
            .collect::<Vec<_>>();
        let (edges, stack_len) =
            stack::calculate(&src.topology, &mut nodes, &src.edges, calculator);
        Self {
            graph: graph_topo::Graph {
                topology: src.topology.clone(),
                nodes,
                edges,
            },
            stack: unsafe { uninit_vec(stack_len) },
        }
    }

    pub fn run(&mut self) {
        let stack = self.stack.as_mut_ptr();
        let mut i = Vec::<*const ()>::with_capacity(8);
        let mut o = Vec::<*mut ()>::with_capacity(8);
        for (node_idx, inputs, outputs) in &self.graph.topology {
            let RoutineWorkspace { routine, workspace } = &self.graph.nodes[node_idx];
            let workspace = unsafe { stack.add(*workspace) as _ };
            i.extend(
                inputs
                    .into_iter()
                    .map(|t| self.graph.edges[t.0 as usize].as_ptr(stack)),
            );
            o.extend(
                outputs
                    .into_iter()
                    .map(|t| self.graph.edges[t as usize].as_mut_ptr(stack)),
            );
            routine.run(workspace, &i, &o);
            i.clear();
            o.clear();
        }
    }
}

trait Routine: Send {
    fn run(&self, workspace: *mut (), inputs: &[*const ()], outputs: &[*mut ()]);
}

#[derive(Clone, Debug)]
enum Blob {
    Constant(computation::Blob),
    Variable(Option<Arc<RefCell<Vec<u8>>>>),
    OnStack(usize),
}

impl Blob {
    const UNINIT: Self = Self::OnStack(usize::MAX);

    #[inline]
    fn variable(size: usize) -> Self {
        Self::Variable(Some(Arc::new(RefCell::new(unsafe { uninit_vec(size) }))))
    }

    #[inline]
    fn extern_variable() -> Self {
        Self::Variable(None)
    }

    #[inline]
    fn as_ptr(&self, stack: *const u8) -> *const () {
        self.as_mut_ptr(stack as _) as _
    }

    #[inline]
    fn as_mut_ptr(&self, stack: *mut u8) -> *mut () {
        match self {
            Blob::Constant(blob) => blob.data.as_ptr() as _,
            Blob::Variable(Some(blob)) => blob.borrow().as_ptr() as _,
            Blob::OnStack(offset) => unsafe { stack.add(*offset) as _ },
            Blob::Variable(None) => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn uninit_vec<T>(size: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(size);
    unsafe { vec.set_len(size) };
    vec
}
