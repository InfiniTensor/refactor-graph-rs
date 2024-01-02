use super::{
    bindings as cuda,
    context::{Context, ContextGuard},
    stream::Stream,
    AsRaw, WithCtx,
};
use std::{ffi::c_void, ptr::null_mut, sync::Arc};

pub(crate) struct Graph {
    ctx: Arc<Context>,
    graph: cuda::CUgraph,
    first_node: cuda::CUgraphNode,
    last_node: cuda::CUgraphNode,
}

impl ContextGuard<'_> {
    pub fn graph(&self) -> Graph {
        let mut graph: cuda::CUgraph = null_mut();
        cuda::invoke!(cuGraphCreate(&mut graph, 0));
        Graph {
            ctx: self.clone_ctx(),
            graph,
            first_node: null_mut(),
            last_node: null_mut(),
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        self.ctx
            .apply(|_| cuda::invoke!(cuGraphDestroy(self.graph)));
    }
}

impl Graph {
    pub fn push_memcpy(
        &mut self,
        dst: *mut c_void,
        src: *const c_void,
        len: usize,
        ty: MemcpyType,
    ) {
        let last_node = self.replace_last();
        let deps = Self::as_deps(&last_node);
        cuda::invoke!(cuGraphAddMemcpyNode(
            &mut self.last_node,
            self.graph,
            deps.as_ptr(),
            deps.len(),
            &params_memcpy3d(dst, src as _, len, ty),
            self.ctx.as_raw()
        ))
    }

    pub fn push_kernel(
        &mut self,
        grid_dims: [u32; 3],
        block_dims: [u32; 3],
        kernel: cuda::CUfunction,
        params: *mut *mut c_void,
        shared_mem_bytes: u32,
    ) {
        let last_node = self.replace_last();
        let deps = Self::as_deps(&last_node);
        cuda::invoke!(cuGraphAddKernelNode_v2(
            &mut self.last_node,
            self.graph,
            deps.as_ptr(),
            deps.len(),
            &cuda::CUDA_KERNEL_NODE_PARAMS {
                func: kernel,
                gridDimX: grid_dims[0],
                gridDimY: grid_dims[1],
                gridDimZ: grid_dims[2],
                blockDimX: block_dims[0],
                blockDimY: block_dims[1],
                blockDimZ: block_dims[2],
                sharedMemBytes: shared_mem_bytes,
                kernelParams: params,
                extra: null_mut(),
                kern: null_mut(),
                ctx: self.ctx.as_raw()
            }
        ))
    }

    #[inline]
    fn replace_last(&mut self) -> cuda::CUgraphNode {
        let ans = std::mem::replace(&mut self.last_node, null_mut());
        if self.first_node.is_null() {
            self.first_node = self.last_node;
        }
        ans
    }

    #[inline]
    fn as_deps(last_node: &cuda::CUgraphNode) -> &[cuda::CUgraphNode] {
        if !last_node.is_null() {
            std::slice::from_ref(&last_node)
        } else {
            &[]
        }
    }
}

pub(crate) struct ExecutableGraph {
    ctx: Arc<Context>,
    graph: cuda::CUgraphExec,
}

impl Graph {
    pub fn instantiate(&self) -> ExecutableGraph {
        let mut graph: cuda::CUgraphExec = null_mut();
        self.ctx
            .apply(|_| cuda::invoke!(cuGraphInstantiateWithFlags(&mut graph, self.graph, 0)));
        ExecutableGraph {
            ctx: self.ctx.clone(),
            graph,
        }
    }
}

impl Drop for ExecutableGraph {
    fn drop(&mut self) {
        self.ctx
            .apply(|_| cuda::invoke!(cuGraphExecDestroy(self.graph)));
    }
}

impl WithCtx for ExecutableGraph {
    #[inline]
    unsafe fn ctx(&self) -> cuda::CUcontext {
        self.ctx.as_raw()
    }
}

impl ExecutableGraph {
    pub fn launch_on(&self, stream: Stream) {
        unsafe { debug_assert_eq!(self.ctx(), stream.ctx()) };
        self.ctx
            .apply(|_| cuda::invoke!(cuGraphLaunch(self.graph, stream.as_raw())));
    }
}

pub enum MemcpyType {
    H2H,
    H2D,
    D2H,
    D2D,
}

fn params_memcpy3d(
    dst: *mut c_void,
    src: cuda::CUdeviceptr,
    len: usize,
    ty: MemcpyType,
) -> cuda::CUDA_MEMCPY3D {
    let mut ans = cuda::CUDA_MEMCPY3D {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: cuda::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        srcHost: null_mut(),
        srcDevice: 0,
        srcArray: null_mut(),
        reserved0: null_mut(),
        srcPitch: 0,
        srcHeight: 0,
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstMemoryType: cuda::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        dstHost: null_mut(),
        dstDevice: 0,
        dstArray: null_mut(),
        reserved1: null_mut(),
        dstPitch: 0,
        dstHeight: 0,
        WidthInBytes: len,
        Height: 1,
        Depth: 1,
    };
    match ty {
        MemcpyType::D2D => {
            ans.srcDevice = src as _;
            ans.dstDevice = dst as _;
        }
        MemcpyType::H2H => {
            ans.srcMemoryType = cuda::CUmemorytype_enum::CU_MEMORYTYPE_HOST;
            ans.srcHost = src as _;
            ans.dstMemoryType = cuda::CUmemorytype_enum::CU_MEMORYTYPE_HOST;
            ans.dstHost = dst as _;
        }
        MemcpyType::H2D => {
            ans.srcMemoryType = cuda::CUmemorytype_enum::CU_MEMORYTYPE_HOST;
            ans.srcHost = src as _;
            ans.dstDevice = dst as _;
        }
        MemcpyType::D2H => {
            ans.srcDevice = src as _;
            ans.dstMemoryType = cuda::CUmemorytype_enum::CU_MEMORYTYPE_HOST;
            ans.dstHost = dst as _;
        }
    };
    ans
}

#[test]
fn test_memcpy3d() {
    for dev in super::device::devices() {
        let size = std::alloc::Layout::array::<usize>(1024).unwrap().size();
        dev.context().apply(|_| {
            let mut dev: cuda::CUdeviceptr = 0;
            let mut host = null_mut();
            let mut ans = null_mut();

            cuda::invoke!(cuMemAlloc_v2(&mut dev, size));
            cuda::invoke!(cuMemHostAlloc(&mut host, size, 0));
            cuda::invoke!(cuMemHostAlloc(&mut ans, size, 0));

            unsafe { std::slice::from_raw_parts_mut(host as *mut usize, 1024) }
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = i);

            cuda::invoke!(cuMemcpyHtoD_v2(dev, host, size));
            cuda::invoke!(cuMemcpy3D_v2(&params_memcpy3d(
                ans,
                dev,
                size,
                MemcpyType::D2H
            )));

            assert_eq!(
                unsafe { std::slice::from_raw_parts(host as *mut usize, 1024) },
                unsafe { std::slice::from_raw_parts(ans as *mut usize, 1024) }
            );

            cuda::invoke!(cuMemFree_v2(dev));
            cuda::invoke!(cuMemFreeHost(host));
            cuda::invoke!(cuMemFreeHost(ans));
        });
    }
}
