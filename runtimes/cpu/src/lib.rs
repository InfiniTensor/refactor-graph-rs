//! Cpu 运行时。

#![deny(warnings, missing_docs)]

mod operator;
mod stack;

use operator::lower;
use rayon::prelude::*;
use stack_calculator::Calculator;
use std::{
    alloc::{self, Layout},
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
    usize,
};

/// 运行时图。
pub struct Graph {
    graph: graph_topo::Graph<RoutineWorkspace, Blob>,
    stack: Vec<u8>,
}

struct RoutineWorkspace {
    routine: Box<dyn Routine>,
    workspace: usize,
}

impl Graph {
    /// 创建运行时图。
    pub fn new(computation::Graph(src): &computation::Graph, calculator: impl Calculator) -> Self {
        let nodes = src.topology.iter().collect::<Vec<_>>();
        let mut nodes = nodes
            .into_par_iter()
            .map(|(node_idx, inputs, outputs)| {
                let mut tensors = Vec::with_capacity(inputs.len() + outputs.len());
                tensors.extend(inputs.clone().into_iter().map(|i| &src.edges[i].0));
                tensors.extend(outputs.into_iter().map(|i| &src.edges[i].0));
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

    /// 运行。
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
                    .map(|t| self.graph.edges[t].as_ptr(stack)),
            );
            o.extend(
                outputs
                    .into_iter()
                    .map(|t| self.graph.edges[t].as_mut_ptr(stack)),
            );
            routine.run(workspace, &i, &o);
            i.clear();
            o.clear();
        }
    }

    /// 拷贝数据到图中位置。
    pub fn copy_in(&mut self, i: usize, data: &[u8]) {
        let blob = &mut self.graph.edges[i];
        match blob {
            Blob::Variable(var) => var.copy_from(data),
            Blob::OnStack(offset) => unsafe {
                let dst = self.stack.as_mut_ptr().add(*offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len())
            },
            Blob::Constant(_) | Blob::Extern(_) => {
                let mut var = Box::new(Variable::new(data.len()));
                var.copy_from(data);
                *blob = Blob::Variable(var);
            }
        }
    }

    /// 拷贝图中位置的数据。
    pub fn copy_out(&self, i: usize, data: &mut [u8]) {
        let blob = &self.graph.edges[i];
        match blob {
            Blob::Variable(var) => var.copy_to(data),
            Blob::Extern(Some(var)) => var.copy_to(data),
            Blob::Constant(blob) => unsafe {
                std::ptr::copy_nonoverlapping(blob.get(), data.as_mut_ptr() as _, data.len())
            },
            Blob::OnStack(offset) => unsafe {
                let src = self.stack.as_ptr().add(*offset);
                std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len())
            },
            Blob::Extern(None) => panic!(),
        }
    }
}

trait Routine: Send {
    fn run(&self, workspace: *mut (), inputs: &[*const ()], outputs: &[*mut ()]);
}

#[derive(Debug)]
enum Blob {
    Constant(computation::Blob),
    Variable(Box<Variable>),
    Extern(Option<Arc<Variable>>),
    OnStack(usize),
}

impl Blob {
    #[inline]
    fn variable(size: usize) -> Self {
        Self::Variable(Box::new(Variable::new(size)))
    }

    #[inline]
    fn empty_extern() -> Self {
        Self::Extern(None)
    }

    #[inline]
    fn as_ptr(&self, stack: *const u8) -> *const () {
        match self {
            Blob::Constant(blob) => blob.get(),
            Blob::Variable(var) => var.as_ptr(),
            Blob::Extern(Some(var)) => var.as_ptr(),
            Blob::OnStack(offset) => unsafe { stack.add(*offset) as _ },
            Blob::Extern(None) => panic!(),
        }
    }

    #[inline]
    fn as_mut_ptr(&mut self, stack: *mut u8) -> *mut () {
        match self {
            Blob::Variable(var) => var.as_mut_ptr(),
            Blob::OnStack(offset) => unsafe { stack.add(*offset) as _ },
            Blob::Constant(_) | Blob::Extern(_) => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn uninit_vec<T>(size: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(size);
    #[allow(clippy::uninit_vec)]
    vec.set_len(size);
    vec
}

#[derive(Debug)]
struct Variable {
    ptr: AtomicPtr<u8>,
    size: usize,
}

impl Variable {
    const ALIGN: usize = std::mem::align_of::<usize>();

    #[inline]
    fn new(size: usize) -> Self {
        Self {
            ptr: AtomicPtr::new({
                let layout = Layout::from_size_align(size, Self::ALIGN).unwrap();
                unsafe { alloc::alloc(layout) }
            }),
            size,
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const () {
        self.ptr.load(Ordering::Acquire) as _
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut () {
        self.ptr.load(Ordering::Acquire) as _
    }

    #[inline]
    fn copy_from(&mut self, src: &[u8]) {
        debug_assert_eq!(src.len(), self.size);
        let ptr = self.ptr.load(Ordering::Acquire);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, self.size) };
    }

    #[inline]
    fn copy_to(&self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), self.size);
        let ptr = self.ptr.load(Ordering::Acquire);
        unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), self.size) };
    }
}

impl Drop for Variable {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(
                self.ptr.swap(std::ptr::null_mut(), Ordering::Relaxed) as _,
                Layout::from_size_align_unchecked(self.size, Self::ALIGN),
            )
        }
    }
}
