use crate::{bindings as cuda, device::Device};
use std::{ptr::null_mut, sync::Arc};

#[repr(transparent)]
pub struct Context(cuda::CUcontext);

impl Device {
    #[inline]
    pub fn context(&self) -> Arc<Context> {
        let mut context: CUcontext = null_mut();
        cuda::invoke!(cuCtxCreate_v2(&mut context, 0, self.index));
        Arc::new(Context(context))
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        cuda::invoke!(cuCtxDestroy_v2(self.0));
    }
}

impl Context {
    #[inline]
    pub fn synchronize(&self) {
        cuda::invoke!(cuCtxPushCurrent_v2(self.0));
        cuda::invoke!(cuCtxSynchronize());
        let mut drop: CUcontext = null_mut();
        cuda::invoke!(cuCtxPopCurrent_v2(&mut drop));
    }
}

pub struct ContextGuard<'a> {
    pub(crate) ctx: &'a Arc<Context>,
    pub(crate) stream: cuda::CUstream,
}

impl Context {
    #[inline]
    pub fn push<'a>(self: &'a Arc<Self>) -> ContextGuard<'a> {
        cuda::invoke!(cuCtxPushCurrent_v2(self.0));

        let mut stream: CUstream = null_mut();
        cuda::invoke!(cuStreamCreate(&mut stream, 0));

        ContextGuard { ctx: self, stream }
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        cuda::invoke!(cuStreamDestroy_v2(self.stream));

        let mut drop: CUcontext = null_mut();
        cuda::invoke!(cuCtxPopCurrent_v2(&mut drop));
    }
}
