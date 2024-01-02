use super::{
    bindings as cuda,
    context::{Context, ContextGuard},
    AsRaw, WithCtx,
};
use std::{
    alloc::Layout,
    ffi::c_void,
    ptr::{null_mut, NonNull},
    sync::Arc,
};

#[derive(Debug)]
pub struct Blob {
    ctx: Arc<Context>,
    ptr: cuda::CUdeviceptr,
    host: Option<NonNull<c_void>>,
    len: usize,
}

impl ContextGuard<'_> {
    pub fn malloc(&self, size: usize) -> Blob {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAlloc_v2(&mut ptr, size));
        Blob {
            ctx: self.clone_ctx(),
            ptr,
            host: None,
            len: size,
        }
    }

    pub fn h2d_cpy<T>(&self, src: &[T]) -> Blob {
        let len = Layout::for_value(src).size();
        let src = src.as_ptr() as *mut c_void;

        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::invoke!(cuMemAlloc_v2(&mut ptr, len));
        cuda::invoke!(cuMemcpyHtoD_v2(ptr, src, len));

        Blob {
            ctx: self.clone_ctx(),
            ptr,
            host: None,
            len,
        }
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        self.ctx.apply(|_| {
            cuda::invoke!(cuMemFree_v2(self.ptr));
            if let Some(host) = self.host.take() {
                cuda::invoke!(cuMemFreeHost(host.as_ptr()));
            }
        });
    }
}

impl AsRaw<cuda::CUdeviceptr> for Blob {
    #[inline]
    unsafe fn as_raw(&self) -> cuda::CUdeviceptr {
        self.ptr
    }
}

impl WithCtx for Blob {
    #[inline]
    unsafe fn ctx(&self) -> cuda::CUcontext {
        self.ctx.as_raw()
    }
}

impl Blob {
    pub fn zero(&mut self) {
        self.ctx
            .apply(|_| cuda::invoke!(cuMemsetD8_v2(self.ptr, 0, self.len)));
    }

    pub fn d2h_cpy<T>(&self) -> Vec<T> {
        let mut host = {
            let size = self.len / std::mem::size_of::<T>();
            let mut vec = Vec::with_capacity(size);
            #[allow(clippy::uninit_vec)]
            unsafe {
                vec.set_len(size)
            };
            vec
        };
        self.ctx
            .apply(|_| cuda::invoke!(cuMemcpyDtoH_v2(host.as_mut_ptr() as _, self.ptr, self.len)));
        host
    }
}

#[test]
fn test_memcpy() {
    for dev in super::device::devices() {
        let ctx = dev.context();

        let mut blob = ctx.apply(|ctx| ctx.malloc(1024));
        blob.zero();
        let mut vec = blob.d2h_cpy::<u32>();
        assert_eq!(vec.len(), 1024 / 4);
        assert!(vec.iter().all(|&x| x == 0));

        vec[0] = 1;
        vec[1024 / 4 - 1] = 2;
        let blob = ctx.apply(|ctx| ctx.h2d_cpy(&vec));
        let vec = blob.d2h_cpy::<u32>();
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1024 / 4 - 1], 2);

        let mut vec_ = vec![0u32; vec.len()];
        ctx.apply(|_| {
            let params = cuda::CUDA_MEMCPY3D {
                srcXInBytes: 0,
                srcY: 0,
                srcZ: 0,
                srcLOD: 0,
                srcMemoryType: cuda::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
                srcHost: null_mut(),
                srcDevice: blob.ptr,
                srcArray: null_mut(),
                reserved0: null_mut(),
                srcPitch: 0,
                srcHeight: 0,
                dstXInBytes: 0,
                dstY: 0,
                dstZ: 0,
                dstLOD: 0,
                dstMemoryType: cuda::CUmemorytype_enum::CU_MEMORYTYPE_HOST,
                dstHost: vec_.as_mut_ptr() as _,
                dstDevice: 0,
                dstArray: null_mut(),
                reserved1: null_mut(),
                dstPitch: 0,
                dstHeight: 0,
                WidthInBytes: blob.len,
                Height: 1,
                Depth: 1,
            };
            cuda::invoke!(cuMemcpy3D_v2(&params))
        });
        assert_eq!(vec, vec_);
    }
}
