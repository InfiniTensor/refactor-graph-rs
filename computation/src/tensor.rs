use crate::udim;
use common::DataType;
use std::sync::Arc;

/// 张量。
#[derive(Clone, Debug)]
pub struct Tensor {
    /// 数据类型。
    pub data_type: DataType,
    /// 形状。
    pub shape: Shape,
    /// 数据布局。
    pub layout: LayoutType,
    /// 数据块。
    pub blob: Blob,
}

/// 数据块引用。
#[derive(Clone, Debug)]
pub struct Blob {
    /// 数据源。
    pub data: Arc<Vec<u8>>,
    /// 数据块的偏移。
    pub offset: usize,
}

/// 张量的形状。
#[derive(Clone, Debug)]
pub struct Shape(pub smallvec::SmallVec<[udim; 4]>);

/// 张量的数据布局。
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LayoutType {
    NCHW,
    NHWC,
    ELSE,
}
