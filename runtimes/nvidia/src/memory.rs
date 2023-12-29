use crate::{
    bindings as cuda,
    context::{Context, ContextGuard},
};
use std::{alloc::Layout, ffi::c_void, ptr::NonNull, sync::Arc};

pub struct Blob {
    ctx: Arc<Context>,
    ptr: cuda::CUdeviceptr,
    host: Option<NonNull<c_void>>,
    size: usize,
}

impl ContextGuard<'_> {
    pub fn malloc(&self, size: usize) -> Blob {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAllocAsync(&mut ptr, size, self.stream));
        Blob {
            ctx: self.ctx.clone(),
            ptr,
            host: None,
            size,
        }
    }

    pub fn malloc_zeroed(&self, size: usize) -> Blob {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAllocAsync(&mut ptr, size, self.stream));
        cuda::invoke!(cuMemsetD8Async(ptr, 0, size, self.stream));
        Blob {
            ctx: self.ctx.clone(),
            ptr,
            host: None,
            size,
        }
    }

    pub fn h2d_cpy<T>(&self, src: &[T]) -> Blob {
        let size = Layout::array::<T>(src.len()).unwrap().size();

        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAllocAsync(&mut ptr, size, self.stream));

        let mut host = std::ptr::null_mut();
        cuda::invoke!(cuMemAllocHost_v2(&mut host, size));
        cuda::invoke!(cuMemcpy(host as _, src.as_ptr() as _, size));
        cuda::invoke!(cuMemcpyHtoDAsync_v2(ptr, host, size, self.stream));

        Blob {
            ctx: self.ctx.clone(),
            ptr,
            host: Some(NonNull::new(host).unwrap()),
            size,
        }
    }
}

impl Blob {
    pub fn drop_host(&self) {
        if let Some(host) = self.host {
            self.ctx.synchronize();
            cuda::invoke!(cuMemFreeHost(host.as_ptr()));
        }
    }

    pub fn d2h_cpy<T>(&self) -> Vec<T> {
        let mut host = {
            let size = self.size / std::mem::size_of::<T>();
            let mut vec = Vec::with_capacity(size);
            unsafe { vec.set_len(size) };
            vec
        };
        self.ctx.synchronize();
        cuda::invoke!(cuMemcpyDtoH_v2(
            host.as_mut_ptr() as *mut _,
            self.ptr,
            self.size
        ));
        host
    }
}
