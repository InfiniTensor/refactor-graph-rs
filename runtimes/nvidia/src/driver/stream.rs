use super::{
    bindings as cuda,
    context::{Context, ContextGuard},
    AsRaw, WithCtx,
};
use std::{ptr::null_mut, sync::Arc};

pub(crate) struct Stream {
    pub(super) ctx: Arc<Context>,
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
    #[inline]
    fn drop(&mut self) {
        self.synchronize();
        self.ctx
            .apply(|_| cuda::invoke!(cuStreamDestroy_v2(self.stream)));
    }
}

impl AsRaw<cuda::CUstream> for Stream {
    #[inline]
    unsafe fn as_raw(&self) -> cuda::CUstream {
        self.stream
    }
}

impl WithCtx for Stream {
    #[inline]
    unsafe fn ctx(&self) -> cuda::CUcontext {
        self.ctx.as_raw()
    }
}

impl Stream {
    #[inline]
    pub fn synchronize(&self) {
        cuda::invoke!(cuStreamSynchronize(self.stream));
    }
}
