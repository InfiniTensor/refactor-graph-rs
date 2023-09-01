use common::DataType;
use smallvec::SmallVec;
use std::ptr::NonNull;

/// Tensor.
#[derive(PartialEq, Eq, Debug)]
pub struct Tensor {
    dt: DataType,
    shape: Shape,
    data: Option<NonNull<u8>>,
}

impl Tensor {
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

    /// Checks if two tensors have the same data type and shape.
    #[inline]
    pub fn info_equal(&self, rhs: &Self) -> bool {
        self.dt == rhs.dt && self.shape == rhs.shape
    }

    /// Checks if the tensor has data allocated.
    #[inline]
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self {
            dt: DataType::UNDEFINED,
            shape: Shape(SmallVec::new()),
            data: None,
        }
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

/// Dimension of shapes maybe a value or a variable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DimExpr {
    /// Shape value.
    Value(i64),
    /// Shape variable.
    Variable(String),
}
