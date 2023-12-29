#![cfg(detected_cuda)]

mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    macro_rules! invoke {
        ($f:expr) => {
            #[allow(unused_imports)]
            use $crate::bindings::*;
            let err = unsafe { $f };
            assert_eq!(err, cudaError_enum::CUDA_SUCCESS);
        };
    }

    pub(crate) use invoke;
}

mod context;
mod device;
mod memory;

pub use device::{devices, Device};

#[test]
fn test() {
    let devices = devices();
    assert!(!devices.is_empty());
    for device in devices {
        let ctx = device.context();
        let guard = ctx.push();

        let mut value = vec![0u32; 4 << 20];
        for (i, x) in value.iter_mut().enumerate() {
            *x = i as u32;
        }

        let blob = guard.h2d_cpy(&value);
        let ans = blob.d2h_cpy();
        assert_eq!(value, ans);
    }
}
