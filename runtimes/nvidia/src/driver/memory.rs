use super::{bindings as cuda, context::ContextGuard, stream::Stream, AsRaw};
use std::{
    alloc::Layout,
    ops::{Add, Deref},
};

#[derive(Default, Debug)]
#[repr(transparent)]
pub struct DevicePtr(cuda::CUdeviceptr);

impl ContextGuard<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevicePtr {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::driver!(cuMemAlloc_v2(&mut ptr, size));
        DevicePtr(ptr)
    }

    #[inline]
    pub fn free(&self, ptr: DevicePtr) {
        cuda::driver!(cuMemFree_v2(ptr.0));
    }
}

impl Stream<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevicePtr {
        let mut ptr: cuda::CUdeviceptr = 0;
        cuda::driver!(cuMemAllocAsync(&mut ptr, size, self.as_raw()));
        DevicePtr(ptr)
    }

    #[inline]
    pub fn free(&self, ptr: DevicePtr) {
        cuda::driver!(cuMemFreeAsync(ptr.0, self.as_raw()));
    }
}

impl Drop for DevicePtr {
    #[inline]
    fn drop(&mut self) {
        assert_eq!(self.0, 0);
    }
}

impl Deref for DevicePtr {
    type Target = RefDevicePtr;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(self) }
    }
}

impl DevicePtr {
    #[inline]
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct RefDevicePtr(cuda::CUdeviceptr);

impl Add<usize> for RefDevicePtr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as cuda::CUdeviceptr)
    }
}

impl RefDevicePtr {
    #[inline]
    pub unsafe fn copy_in<T>(&mut self, data: &[T], _ctx: &ContextGuard) {
        cuda::driver!(cuMemcpyHtoD_v2(
            self.0,
            data.as_ptr().cast(),
            Layout::array::<T>(data.len()).unwrap().size()
        ));
    }

    #[inline]
    pub unsafe fn copy_out<T>(&mut self, data: &mut [T], _ctx: &ContextGuard) {
        cuda::driver!(cuMemcpyDtoH_v2(
            data.as_mut_ptr().cast(),
            self.0,
            Layout::array::<T>(data.len()).unwrap().size()
        ));
    }

    #[inline]
    pub unsafe fn copy_in_async<T>(&mut self, data: &[T], stream: &Stream) {
        cuda::driver!(cuMemcpyHtoDAsync_v2(
            self.0,
            data.as_ptr().cast(),
            Layout::array::<T>(data.len()).unwrap().size(),
            stream.as_raw(),
        ));
    }

    #[inline]
    pub unsafe fn copy_out_async<T>(&mut self, data: &mut [T], stream: &Stream) {
        cuda::driver!(cuMemcpyDtoHAsync_v2(
            data.as_mut_ptr().cast(),
            self.0,
            Layout::array::<T>(data.len()).unwrap().size(),
            stream.as_raw(),
        ));
    }
}
