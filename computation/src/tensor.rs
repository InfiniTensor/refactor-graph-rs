use crate::blob::Blob;
use common::DataType;
use smallvec::SmallVec;
use std::rc::Rc;

/// Tensor.
#[derive(PartialEq, Eq, Debug)]
pub struct Tensor {
    dt: DataType,
    shape: Shape,
    data: Rc<Blob>,
}

impl Tensor {
    /// Creates a new tensor to represent unknown tensor.
    #[inline]
    pub fn unknown() -> Self {
        Self {
            dt: DataType::UNDEFINED,
            shape: Shape(SmallVec::new()),
            data: Rc::new(Blob::EMPTY),
        }
    }

    /// Creates a new tensor with data allocated.
    #[inline]
    pub fn with_data(dt: DataType, shape: Shape, data: Rc<Blob>) -> Self {
        Self { dt, shape, data }
    }

    /// Creates a new tensor without data allocated.
    #[inline]
    pub fn without_data(dt: DataType, shape: Shape) -> Self {
        Self {
            dt,
            shape,
            data: Rc::new(Blob::EMPTY),
        }
    }

    /// Checks if the tensor is unknown.
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self.dt, DataType::UNDEFINED) && self.shape.0.is_empty() && !self.data.exist()
    }

    /// Checks if two tensors have the same data type and shape.
    #[inline]
    pub fn info_equal(&self, rhs: &Self) -> bool {
        self.dt == rhs.dt && self.shape == rhs.shape
    }

    /// Checks if the tensor has data allocated.
    #[inline]
    pub fn has_data(&self) -> bool {
        self.data.exist()
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
        self.data.as_ptr_unchecked()
    }
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self::unknown()
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
    Variable(Rc<String>),
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
        Self::Variable(Rc::new(value))
    }
}
