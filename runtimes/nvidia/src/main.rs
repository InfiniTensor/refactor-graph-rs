mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

fn main() {
    let err = unsafe { bindings::cuInit(0) };
    assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

    let mut device: bindings::CUdevice = 0;
    let err = unsafe { bindings::cuDeviceGet(&mut device, 0) };
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

    println!("CUDA device: {device}, compute capability: {major}.{minor}");

    let mut bytes = 0usize;
    let err = unsafe { bindings::cuDeviceTotalMem_v2(&mut bytes, device) };
    assert_eq!(err, bindings::cudaError_enum::CUDA_SUCCESS);

    println!("CUDA device memory: {bytes} bytes");
}
