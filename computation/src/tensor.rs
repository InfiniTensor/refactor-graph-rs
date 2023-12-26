use crate::udim;
use common::DataType;
use std::{alloc::Layout, fmt, str::FromStr, sync::Arc};

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
    pub blob: Option<Blob>,
}

impl Tensor {
    /// 获取数据块的内存布局。
    #[inline]
    pub fn blob_mem_layout(&self) -> Layout {
        self.data_type.array_layout(self.shape.elements_len() as _)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.blob.is_some() {
            f.write_str("(*) ")?;
        }
        f.write_str(&format!("{:?}", self.data_type).to_lowercase())?;
        let mut shape = self.shape.0.iter();
        let Some(first) = shape.next() else {
            return Ok(());
        };
        write!(f, "<{first}")?;
        while let Some(d) = shape.next() {
            write!(f, "x{d}")?;
        }
        match self.layout {
            LayoutType::NCHW => f.write_str("(NCHW)>"),
            LayoutType::NHWC => f.write_str("(NHWC)>"),
            LayoutType::ELSE => f.write_str(">"),
        }
    }
}

/// 数据块引用。
#[derive(Clone, Debug)]
pub struct Blob {
    /// 数据源。
    pub data: Arc<Vec<u8>>,
    /// 数据块的偏移。
    pub offset: usize,
}

impl Blob {
    /// 获取数据块的指针。
    #[inline]
    pub fn get(&self) -> *const () {
        self.data[self.offset..].as_ptr() as _
    }
}

/// 张量的形状。
#[derive(Clone, Debug)]
pub struct Shape(pub smallvec::SmallVec<[udim; 4]>);

impl Shape {
    /// 获取形状对应的元素数量。
    #[inline]
    pub fn elements_len(&self) -> udim {
        self.0.iter().product::<udim>()
    }
}

/// 张量的数据布局。
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LayoutType {
    NCHW,
    NHWC,
    ELSE,
}

impl FromStr for LayoutType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NCHW" => Ok(LayoutType::NCHW),
            "NHWC" => Ok(LayoutType::NHWC),
            "ELSE" => Ok(LayoutType::ELSE),
            _ => Err(()),
        }
    }
}
