use super::{bindings as cuda, context::ContextGuard};

pub struct Blob {
    ptr: cuda::CUdeviceptr,
    size: usize,
}

impl ContextGuard<'_> {
    pub fn malloc(&self, size: usize) -> Blob {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAlloc_v2(&mut ptr, size));
        Blob { ptr, size }
    }
}
