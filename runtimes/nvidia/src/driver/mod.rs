mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    macro_rules! driver {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::driver::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, CUresult::CUDA_SUCCESS);
        }};
    }

    macro_rules! nvrtc {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::driver::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, nvrtcResult::NVRTC_SUCCESS);
        }};
    }

    #[inline(always)]
    pub(crate) fn init() {
        driver!(cuInit(0));
    }

    pub(super) use {driver, nvrtc};
}

mod context;
mod device;
mod graph;
mod memory;
pub mod nvrtc;
mod stream;

trait AsRaw<T> {
    unsafe fn as_raw(&self) -> T;
}

trait WithCtx {
    unsafe fn ctx(&self) -> bindings::CUcontext;
}

pub(crate) use context::{Context, ContextGuard};
pub(crate) use device::devices;
pub(crate) use graph::{ExecutableGraph, Graph};
pub(crate) use memory::{DevicePtr, RefDevicePtr};
