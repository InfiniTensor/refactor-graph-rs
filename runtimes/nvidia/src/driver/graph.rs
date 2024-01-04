use super::{bindings as cuda, context::ContextGuard, stream::Stream, AsRaw};
use std::{ffi::c_void, ptr::null_mut};

pub(crate) struct Graph {
    graph: cuda::CUgraph,
    first_node: cuda::CUgraphNode,
    last_node: cuda::CUgraphNode,
}

impl Drop for Graph {
    fn drop(&mut self) {
        cuda::driver!(cuGraphDestroy(self.graph));
    }
}

impl Graph {
    pub fn new() -> Self {
        let mut graph: cuda::CUgraph = null_mut();
        cuda::driver!(cuGraphCreate(&mut graph, 0));
        Self {
            graph,
            first_node: null_mut(),
            last_node: null_mut(),
        }
    }

    pub fn push_memcpy(
        &mut self,
        dst: *mut c_void,
        src: *const c_void,
        len: usize,
        ty: MemcpyType,
        ctx: &ContextGuard,
    ) {
        let last_node = self.replace_last();
        let deps = Self::as_deps(&last_node);
        cuda::driver!(cuGraphAddMemcpyNode(
            &mut self.last_node,
            self.graph,
            deps.as_ptr(),
            deps.len(),
            &params_memcpy3d(dst, src as _, len, ty),
            ctx.as_raw(),
        ))
    }

    pub fn push_kernel(
        &mut self,
        grid_dims: [u32; 3],
        block_dims: [u32; 3],
        kernel: cuda::CUfunction,
        params: *mut *mut c_void,
        shared_mem_bytes: u32,
        ctx: &ContextGuard,
    ) {
        let last_node = self.replace_last();
        let deps = Self::as_deps(&last_node);
        cuda::driver!(cuGraphAddKernelNode_v2(
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
                ctx: ctx.as_raw(),
            }
        ))
    }

    pub fn record_child_graph(&mut self, f: impl FnOnce(&Stream), ctx: &ContextGuard) {
        let mut child_graph: cuda::CUgraph = null_mut();
        let capture_mode = cuda::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL;
        {
            let stream = ctx.stream();
            cuda::driver!(cuStreamBeginCapture_v2(stream.as_raw(), capture_mode));
            f(&stream);
            cuda::driver!(cuStreamEndCapture(stream.as_raw(), &mut child_graph));
        }

        let last_node = self.replace_last();
        let deps = Self::as_deps(&last_node);
        cuda::driver!(cuGraphAddChildGraphNode(
            &mut self.last_node,
            self.graph,
            deps.as_ptr(),
            deps.len(),
            child_graph
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

pub(crate) struct ExecutableGraph(cuda::CUgraphExec);

impl Graph {
    #[inline]
    pub fn instantiate(&self, _: &ContextGuard) -> ExecutableGraph {
        let mut graph: cuda::CUgraphExec = null_mut();
        cuda::driver!(cuGraphInstantiateWithFlags(&mut graph, self.graph, 0));
        ExecutableGraph(graph)
    }
}

impl Drop for ExecutableGraph {
    #[inline]
    fn drop(&mut self) {
        cuda::driver!(cuGraphExecDestroy(self.0));
    }
}

impl ExecutableGraph {
    /// 在 `stream` 上启动图。
    ///
    /// # Safety
    ///
    /// `stream` 所在的上下文与图中节点的上下文必须有正确的关系。
    #[inline]
    pub unsafe fn launch_on(&self, stream: &Stream) {
        cuda::driver!(cuGraphLaunch(self.0, stream.as_raw()));
    }
}

pub enum MemcpyType {
    H2H,
    H2D,
    D2H,
    D2D,
}

/// 构造一个表示 `memcpy(dst, src, len, ty)` 的 [`CUDA_MEMCPY3D`](cuda::CUDA_MEMCPY3D)。
fn params_memcpy3d(
    dst: *mut c_void,
    src: cuda::CUdeviceptr,
    len: usize,
    ty: MemcpyType,
) -> cuda::CUDA_MEMCPY3D {
    use cuda::CUmemorytype::*;
    let mut ans = cuda::CUDA_MEMCPY3D {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: CU_MEMORYTYPE_DEVICE,
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
        dstMemoryType: CU_MEMORYTYPE_DEVICE,
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
            ans.srcMemoryType = CU_MEMORYTYPE_HOST;
            ans.srcHost = src as _;
            ans.dstMemoryType = CU_MEMORYTYPE_HOST;
            ans.dstHost = dst as _;
        }
        MemcpyType::H2D => {
            ans.srcMemoryType = CU_MEMORYTYPE_HOST;
            ans.srcHost = src as _;
            ans.dstDevice = dst as _;
        }
        MemcpyType::D2H => {
            ans.srcDevice = src as _;
            ans.dstMemoryType = CU_MEMORYTYPE_HOST;
            ans.dstHost = dst as _;
        }
    };
    ans
}

/// 测试 [`params_memcpy3d`] 构造的 [`cuMemcpy3D`](cuda::cuMemcpy3D_v2) 参数。
#[test]
fn test_memcpy3d() {
    type TY = usize;
    const LEN: usize = 1024;

    let Some(dev) = super::device::devices().iter().next() else {
        return;
    };

    let size = std::alloc::Layout::array::<TY>(LEN).unwrap().size();
    dev.context().apply(|_| {
        let mut dev: cuda::CUdeviceptr = 0;
        let mut host = null_mut();
        let mut ans = null_mut();

        cuda::driver!(cuMemAlloc_v2(&mut dev, size));
        cuda::driver!(cuMemHostAlloc(&mut host, size, 0));
        cuda::driver!(cuMemHostAlloc(&mut ans, size, 0));

        unsafe { std::slice::from_raw_parts_mut(host as *mut TY, LEN) }
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i);

        cuda::driver!(cuMemcpyHtoD_v2(dev, host, size));
        cuda::driver!(cuMemcpy3D_v2(&params_memcpy3d(
            ans,
            dev,
            size,
            MemcpyType::D2H
        )));

        assert_eq!(
            unsafe { std::slice::from_raw_parts(host as *mut TY, LEN) },
            unsafe { std::slice::from_raw_parts(ans as *mut TY, LEN) }
        );

        cuda::driver!(cuMemFree_v2(dev));
        cuda::driver!(cuMemFreeHost(host));
        cuda::driver!(cuMemFreeHost(ans));
    });
}

/// 测试 cuda graph 与 context 的交互行为。
#[test]
fn test_behavior() {
    let Some(dev) = super::device::devices().iter().next() else {
        return;
    };

    // 创建 cuda graph 不需要 context。
    let mut graph: cuda::CUgraph = null_mut();
    cuda::driver!(cuGraphCreate(&mut graph, 0));

    let mut ctx0: cuda::CUcontext = null_mut();
    let mut ctx1: cuda::CUcontext = null_mut();

    // 创建 cuda graph exec 需要 context。
    let mut execuable: cuda::CUgraphExec = null_mut();
    dev.context().apply(|ctx| {
        cuda::driver!(cuGraphInstantiateWithFlags(&mut execuable, graph, 0));
        ctx0 = unsafe { ctx.as_raw() };
    });
    // 创建 cuda graph exec 的上下文不需要维持生命周期。

    // 执行 cuda graph exec 需要 context。
    dev.context().apply(|ctx| {
        let mut stream: cuda::CUstream = null_mut();
        cuda::driver!(cuStreamCreate(&mut stream, 0));

        cuda::driver!(cuGraphLaunch(execuable, stream));
        cuda::driver!(cuStreamSynchronize(stream));

        cuda::driver!(cuStreamDestroy_v2(stream));
        ctx1 = unsafe { ctx.as_raw() };
    });
    // 销毁 cuda graph exec 不需要 context。
    cuda::driver!(cuGraphExecDestroy(execuable));

    // 创建 cuda graph exec 与执行 cuda graph exec 的 context 不需要相同。
    assert_ne!(ctx0, ctx1);

    // 销毁 cuda graph 不需要 context。
    cuda::driver!(cuGraphDestroy(graph));
}
