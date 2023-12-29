#![cfg(detected_cuda)]

use std::{
    sync::{Mutex, OnceLock},
    vec::Vec,
};

mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

fn manager() -> &'static Mutex<DeviceManager> {
    static MANAGER: OnceLock<Mutex<DeviceManager>> = OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(DeviceManager::new()))
}

struct DeviceInfo {
    capability: (i32, i32),
    total_memory: usize,
}

struct DeviceManager {
    devices: Vec<DeviceInfo>,
}

impl DeviceManager {
    fn new() -> Self {
        let err = unsafe { bindings::cuInit(0) };
        assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

        let mut device_count = 0i32;
        let err = unsafe { bindings::cuDeviceGetCount(&mut device_count) };
        assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

        let devices = (0..device_count)
            .map(|i| {
                let mut device: bindings::CUdevice = 0;
                let err = unsafe { bindings::cuDeviceGet(&mut device, i) };
                assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

                let mut major = 0i32;
                let err = unsafe {
                    bindings::cuDeviceGetAttribute(
                    &mut major,
                    bindings::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                )
                };
                assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);
                let mut minor = 0i32;
                let err = unsafe {
                    bindings::cuDeviceGetAttribute(
                    &mut minor,
                    bindings::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                )
                };
                assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

                let mut bytes = 0usize;
                let err = unsafe { bindings::cuDeviceTotalMem_v2(&mut bytes, device) };
                assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

                DeviceInfo {
                    capability: (major, minor),
                    total_memory: bytes,
                }
            })
            .collect();

        Self { devices }
    }
}

#[test]
fn test() {
    let manager = manager().lock().unwrap();
    for (i, dev) in manager.devices.iter().enumerate() {
        println!(
            "gpu{i}: ver{}.{} mem={}",
            dev.capability.0, dev.capability.1, dev.total_memory
        );
    }
}
