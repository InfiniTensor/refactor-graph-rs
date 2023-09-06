use common::DataType;
use smallvec::SmallVec;
use std::ptr::NonNull;

use crate::InferError;

/// Tensor.
#[derive(PartialEq, Eq, Debug)]
pub struct Tensor {
    dt: DataType,
    shape: Shape,
    data: Option<NonNull<u8>>,
}

impl Tensor {
    /// Creates a new tensor to represent unknown tensor.
    #[inline]
    pub fn unknown() -> Self {
        Self {
            dt: DataType::UNDEFINED,
            shape: Shape(SmallVec::new()),
            data: None,
        }
    }

    /// Creates a new tensor with data allocated.
    #[inline]
    pub fn with_data(dt: DataType, shape: Shape, data: *mut u8) -> Self {
        Self {
            dt,
            shape,
            data: NonNull::new(data),
        }
    }

    /// Creates a new tensor without data allocated.
    #[inline]
    pub fn without_data(dt: DataType, shape: Shape) -> Self {
        Self {
            dt,
            shape,
            data: None,
        }
    }

    /// Checks if the tensor is unknown.
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self.dt, DataType::UNDEFINED) && self.shape.0.is_empty() && self.data.is_none()
    }

    /// Checks if two tensors have the same data type and shape.
    #[inline]
    pub fn info_equal(&self, rhs: &Self) -> bool {
        self.dt == rhs.dt && self.shape == rhs.shape
    }

    /// Checks if the tensor has data allocated.
    #[inline]
    pub const fn has_data(&self) -> bool {
        self.data.is_some()
    }

    /// Gets the data type of the tensor.
    #[inline]
    pub const fn data_type(&self) -> DataType {
        self.dt
    }

    /// Gets the shape of the tensor.
    #[inline]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Gets the data of the tensor.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the tensor has data allocated.
    #[inline]
    pub unsafe fn raw_data_unsafe(&self) -> *const u8 {
        self.data.unwrap().as_ptr()
    }
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self::unknown()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Some(ptr) = self.data.take() {
            let size = self
                .shape
                .0
                .iter()
                .map(|d| match d {
                    DimExpr::Value(val) => val,
                    DimExpr::Variable(_) => unreachable!(),
                })
                .product::<i64>();
            unsafe { std::alloc::dealloc(ptr.as_ptr(), self.dt.array_layout(size as _)) };
        }
    }
}

/// Shape.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Shape(pub smallvec::SmallVec<[DimExpr; 4]>);

impl Shape {
    /// Calculates the size of the shape.
    pub fn size(&self) -> usize {
        let mut ans = 1;
        for dim in self.0.iter() {
            match dim {
                DimExpr::Value(val) if *val > 0 => ans *= *val as usize,
                _ => todo!(),
            }
        }
        ans
    }
}

/// Dimension of shapes maybe a value or a variable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DimExpr {
    /// Shape value.
    Value(i64),
    /// Shape variable.
    Variable(String),
}

impl From<i64> for DimExpr {
    #[inline]
    fn from(value: i64) -> Self {
        Self::Value(value)
    }
}

impl From<String> for DimExpr {
    #[inline]
    fn from(value: String) -> Self {
        Self::Variable(value)
    }
}

impl DimExpr {
    /// Gets the value of the dimension.
    #[inline]
    pub fn value(&self) -> Result<i64, InferError> {
        match self {
            Self::Value(val) => Ok(*val),
            _ => todo!("Cannot get value of a variable dimension."),
        }
    }
}
