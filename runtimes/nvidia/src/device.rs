use crate::bindings as cuda;
use std::{sync::OnceLock, vec::Vec};

pub(crate) fn devices() -> &'static Vec<Device> {
    static MANAGER: OnceLock<Vec<Device>> = OnceLock::new();
    MANAGER.get_or_init(|| {
        cuda::invoke!(cuInit(0));

        let mut device_count = 0i32;
        cuda::invoke!(cuDeviceGetCount(&mut device_count));

        (0..device_count).map(Device::new).collect()
    })
}

pub(crate) struct Device {
    pub index: i32,
    pub capability: (i32, i32),
    pub total_memory: usize,
}

impl Device {
    pub fn new(index: i32) -> Self {
        let mut device: cuda::CUdevice = 0;
        cuda::invoke!(cuDeviceGet(&mut device, index));

        let mut major = 0i32;
        let mut minor = 0i32;
        cuda::invoke!(cuDeviceGetAttribute(
            &mut major,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        ));
        cuda::invoke!(cuDeviceGetAttribute(
            &mut minor,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        ));

        let mut bytes = 0usize;
        cuda::invoke!(cuDeviceTotalMem_v2(&mut bytes, device));
        Self {
            index,
            capability: (major, minor),
            total_memory: bytes,
        }
    }
}

#[test]
fn test() {
    for (i, dev) in devices().iter().enumerate() {
        println!(
            "gpu{i}: ver{}.{} mem={}",
            dev.capability.0, dev.capability.1, dev.total_memory
        );
    }
}
