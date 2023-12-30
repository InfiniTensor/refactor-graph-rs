mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    macro_rules! invoke {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::driver::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cudaError_enum::CUDA_SUCCESS);
        }};
    }

    pub(crate) use invoke;
}

mod context;
mod device;
mod graph;
mod memory;
mod stream;
