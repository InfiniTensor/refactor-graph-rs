﻿use crate::{bindings as cuda, device::Device};
use std::{ptr::null_mut, sync::Arc};

#[repr(transparent)]
pub(crate) struct Context(cuda::CUcontext);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Device {
    #[inline]
    pub fn context(&self) -> Arc<Context> {
        let mut context: CUcontext = null_mut();
        cuda::invoke!(cuCtxCreate_v2(&mut context, 0, self.index));
        cuda::invoke!(cuCtxPopCurrent_v2(null_mut()));
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
    pub fn apply<T>(self: &Arc<Self>, f: impl FnOnce(&ContextGuard<'_>) -> T) -> T {
        let guard = self.push();
        f(&guard)
    }
}

pub(crate) struct ContextGuard<'a>(&'a Arc<Context>);

impl Context {
    #[inline]
    fn push<'a>(self: &'a Arc<Context>) -> ContextGuard<'a> {
        cuda::invoke!(cuCtxPushCurrent_v2(self.0));
        ContextGuard(self)
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut top: CUcontext = null_mut();
        cuda::invoke!(cuCtxPopCurrent_v2(&mut top));
        debug_assert_eq!(top, self.0 .0)
    }
}

impl ContextGuard<'_> {
    #[inline]
    pub fn clone_ctx(&self) -> Arc<Context> {
        self.0.clone()
    }

    #[inline]
    pub fn synchronize(&self) {
        cuda::invoke!(cuCtxSynchronize());
    }
}
