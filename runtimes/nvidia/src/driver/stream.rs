use super::{bindings as cuda, context::ContextGuard, AsRaw};
use std::{marker::PhantomData, ptr::null_mut};

pub(crate) struct Stream<'a>(cuda::CUstream, PhantomData<&'a ()>);

impl ContextGuard<'_> {
    pub fn stream(&self) -> Stream {
        let mut stream: cuda::CUstream = null_mut();
        cuda::invoke!(cuStreamCreate(&mut stream, 0));
        Stream(stream, PhantomData)
    }
}

impl Drop for Stream<'_> {
    #[inline]
    fn drop(&mut self) {
        self.synchronize();
        cuda::invoke!(cuStreamDestroy_v2(self.0));
    }
}

impl AsRaw<cuda::CUstream> for Stream<'_> {
    #[inline]
    unsafe fn as_raw(&self) -> cuda::CUstream {
        self.0
    }
}

impl Stream<'_> {
    #[inline]
    pub fn synchronize(&self) {
        cuda::invoke!(cuStreamSynchronize(self.0));
    }
}
