use super::{
    bindings as cuda,
    context::{Context, ContextGuard},
};
use std::{ptr::null_mut, sync::Arc};

pub(crate) struct Stream {
    ctx: Arc<Context>,
    stream: cuda::CUstream,
}

impl ContextGuard<'_> {
    pub fn stream(&self) -> Stream {
        let mut stream: cuda::CUstream = null_mut();
        cuda::invoke!(cuStreamCreate(&mut stream, 0));
        Stream {
            ctx: self.clone_ctx(),
            stream,
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.ctx
            .apply(|_| cuda::invoke!(cuStreamDestroy_v2(self.stream)));
    }
}
