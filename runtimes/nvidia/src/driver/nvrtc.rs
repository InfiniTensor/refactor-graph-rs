﻿use super::{bindings as cuda, Context, ContextGuard};
use std::{
    collections::{hash_map::Keys, HashMap},
    ffi::{c_char, CStr, CString},
    ptr::{null, null_mut},
    sync::Arc,
    sync::{Mutex, OnceLock},
};

static MODULES: OnceLock<Mutex<HashMap<String, Arc<Module>>>> = OnceLock::new();

pub(crate) fn compile<'a, I, U, V>(code: &str, symbols: I, ctx: &ContextGuard)
where
    I: IntoIterator<Item = (U, V)>,
    U: AsRef<str>,
    V: AsRef<str>,
{
    let symbols = symbols
        .into_iter()
        .map(|(k, v)| (k.as_ref().to_owned(), v.as_ref().to_owned()))
        .collect::<HashMap<_, _>>();
    // 先检查一遍并确保静态对象创建
    let modules = if let Some(modules) = MODULES.get() {
        if check_hold(&*modules.lock().unwrap(), symbols.keys()) {
            return;
        }
        modules
    } else {
        MODULES.get_or_init(|| Default::default())
    };
    // 编译
    let (module, log) = Module::from_src(code, ctx);
    println!("{log}");
    // 再上锁检查一遍
    let module = Arc::new(module.unwrap());
    let mut map = modules.lock().unwrap();
    if !check_hold(&*map, symbols.keys()) {
        for k in symbols.keys() {
            // 确认指定的符号都存在
            module.get_function(k);
            map.insert(k.clone(), module.clone());
        }
    }
}

pub(crate) fn get_function(name: &str) -> Option<cuda::CUfunction> {
    MODULES.get().and_then(|modules| {
        modules
            .lock()
            .unwrap()
            .get(name)
            .map(|module| module.get_function(name))
    })
}

fn check_hold(map: &HashMap<String, Arc<Module>>, symbols: Keys<'_, String, String>) -> bool {
    let len = symbols.len();
    let had = symbols.filter(|&k| map.contains_key(k)).count();
    if had == len {
        true
    } else if had == 0 {
        false
    } else {
        panic!()
    }
}

struct Module {
    ctx: Arc<Context>,
    module: cuda::CUmodule,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    #[inline]
    fn drop(&mut self) {
        cuda::driver!(cuModuleUnload(self.module));
    }
}

impl Module {
    fn from_src(code: &str, ctx: &ContextGuard) -> (Result<Self, cuda::nvrtcResult>, String) {
        let code = {
            let mut headers = String::new();

            if code.contains("half") {
                headers.push_str("#include <cuda_fp16.h>\n");
            }
            if code.contains("nv_bfloat16") {
                headers.push_str("#include <cuda_bf16.h>\n");
            }

            if !headers.is_empty() {
                headers.push_str(code);
                CString::new(headers.as_str())
            } else {
                CString::new(code)
            }
            .unwrap()
        };
        let mut program: cuda::nvrtcProgram = null_mut();
        cuda::nvrtc!(nvrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));

        let options = vec![
            CString::new("--std=c++17").unwrap(),
            CString::new("--gpu-architecture=compute_80").unwrap(),
            CString::new(format!("-I{}/include", std::env!("CUDA_ROOT"))).unwrap(),
        ];
        let options = options
            .iter()
            .map(|s| s.as_ptr().cast::<c_char>())
            .collect::<Vec<_>>();

        let result =
            unsafe { cuda::nvrtcCompileProgram(program, options.len() as _, options.as_ptr()) };
        let log = {
            let mut log_len = 0;
            cuda::nvrtc!(nvrtcGetProgramLogSize(program, &mut log_len));
            if log_len > 1 {
                let mut log = vec![0u8; log_len];
                cuda::nvrtc!(nvrtcGetProgramLog(program, log.as_mut_ptr().cast()));
                log.pop();
                String::from_utf8(log).unwrap()
            } else {
                String::new()
            }
        };
        if result != cuda::nvrtcResult::NVRTC_SUCCESS {
            return (Err(result), log);
        }

        let ptx = {
            let mut ptx_len = 0;
            cuda::nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            cuda::nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            cuda::nvrtc!(nvrtcDestroyProgram(&mut program));
            ptx
        };
        let ptx = CStr::from_bytes_with_nul(ptx.as_slice()).unwrap();

        let mut module: cuda::CUmodule = null_mut();
        cuda::driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        (
            Ok(Self {
                ctx: ctx.clone_ctx(),
                module,
            }),
            log,
        )
    }

    #[inline]
    fn get_function(&self, name: &str) -> cuda::CUfunction {
        let name = CString::new(name).unwrap();
        let mut func: cuda::CUfunction = null_mut();
        self.ctx
            .apply(|_| cuda::driver!(cuModuleGetFunction(&mut func, self.module, name.as_ptr())));
        func
    }
}

#[test]
fn test_env() {
    let cuda_root = std::env!("CUDA_ROOT");
    assert!(!cuda_root.is_empty());
    // println!("cuda root = \"{}\"", cuda_root);
}

#[test]
fn test_behavior() {
    const SRC: &str = r#"
extern "C" __global__ void kernel() {
    printf("Hello World from GPU!\n");
}
"#;

    let Some(dev) = super::device::devices().iter().next() else {
        return;
    };
    dev.context().apply(|ctx| {
        let code = CString::new(SRC).unwrap();
        let mut program: cuda::nvrtcProgram = null_mut();
        cuda::nvrtc!(nvrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));

        cuda::nvrtc!(nvrtcCompileProgram(program, 0, null()));

        let ptx = {
            let mut ptx_len = 0;
            cuda::nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            cuda::nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            cuda::nvrtc!(nvrtcDestroyProgram(&mut program));
            ptx
        };

        let ptx = CStr::from_bytes_with_nul(ptx.as_slice()).unwrap();
        let mut module: cuda::CUmodule = null_mut();
        cuda::driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));

        let name = CString::new("kernel").unwrap();
        let mut function: cuda::CUfunction = null_mut();
        cuda::driver!(cuModuleGetFunction(&mut function, module, name.as_ptr()));

        cuda::driver!(cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            null_mut(),
            null_mut(),
            null_mut()
        ));
        ctx.synchronize();
    });
}

#[test]
fn test_module() {
    const SRC: &str = r#"
extern "C" __global__ void kernel() {
    printf("Hello World from GPU!\n");
}
"#;

    let Some(dev) = super::device::devices().iter().next() else {
        return;
    };
    dev.context().apply(|ctx| {
        let (module, _log) = Module::from_src(SRC, ctx);
        let module = module.unwrap();
        let function = module.get_function("kernel");

        cuda::driver!(cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            null_mut(),
            null_mut(),
            null_mut()
        ));
        ctx.synchronize();
    });
}
