use crate::{bindings as cuda, context::ContextGuard};
use std::ptr::null_mut;

pub(crate) struct Stream<'a> {
    stream: cuda::CUstream,
    _ctx: &'a ContextGuard<'a>,
}

impl ContextGuard<'_> {
    pub fn stream(&self) -> Stream<'_> {
        let mut stream: CUstream = null_mut();
        cuda::invoke!(cuStreamCreate(&mut stream, 0));
        Stream { stream, _ctx: self }
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        cuda::invoke!(cuStreamDestroy_v2(self.stream));
    }
}
