#![allow(unused)]

mod foreigner;

use smallvec::SmallVec;
use std::{ptr::NonNull, rc::Rc};

pub use foreigner::ForeignSubgrpah;

pub struct Graph(graph_topo::Graph<Node, Edge>);

type Node = Rc<Kernel>;
type Edge = Rc<MemoryBlock>;

/// 计算核函数。
struct Kernel {
    /// 名字（作为唯一键）。
    name: String,
    /// 所有输入的读法。
    readers: Vec<Layout>,
    /// 所有输出的写法。
    writers: Vec<Layout>,
    /// 核属性。
    attributes: (),
}

/// 内存块。
struct MemoryBlock {
    /// 内存块字节数。
    bytes_len: usize,
    /// 当前数据（可能存在）。
    data: Option<NonNull<u8>>,
    /// 要分配的段。可以用空字符串表示默认值。
    segment: String,
    /// 访存存类型/级别。
    memory_type_level: usize,
}

/// 读写方式。
struct Layout {
    strides: SmallVec<[usize; 4]>,
}
