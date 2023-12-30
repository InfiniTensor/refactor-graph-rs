use super::{
    bindings as cuda,
    context::{Context, ContextGuard},
};
use std::{ptr::null_mut, sync::Arc};

pub(crate) struct Graph {
    ctx: Arc<Context>,
    graph: cuda::CUgraph,
}

impl ContextGuard<'_> {
    pub fn graph(&self) -> Graph {
        let mut graph: cuda::CUgraph = null_mut();
        cuda::invoke!(cuGraphCreate(&mut graph, 0));
        Graph {
            ctx: self.clone_ctx(),
            graph,
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        self.ctx
            .apply(|_| cuda::invoke!(cuGraphDestroy(self.graph)));
    }
}
