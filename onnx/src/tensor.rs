use common::DataType;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct Tensor {
    dt: DataType,
    shape: Shape,
    data: *mut u8,
}

impl Tensor {
    #[inline]
    pub const fn with_data(dt: DataType, shape: Shape, data: *mut u8) -> Self {
        Self { dt, shape, data }
    }

    #[inline]
    pub const fn without_data(dt: DataType, shape: Shape) -> Self {
        Self {
            dt,
            shape,
            data: null_mut(),
        }
    }

    #[inline]
    pub fn info_equal(&self, rhs: &Self) -> bool {
        self.dt == rhs.dt && self.shape == rhs.shape
    }

    #[inline]
    pub fn has_data(&self) -> bool {
        !self.data.is_null()
    }
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self {
            dt: DataType::UNDEFINED,
            shape: Default::default(),
            data: null_mut(),
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.data.is_null() {
            return;
        }
        let mut size = 1;
        for d in &self.shape.0 {
            match d {
                DimExpr::Value(val) => size *= val,
                DimExpr::Variable(_) => return,
            }
        }
        unsafe {
            std::alloc::dealloc(
                std::mem::replace(&mut self.data, null_mut()),
                self.dt.array_layout(size as _),
            )
        };
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DimExpr {
    Value(i64),
    Variable(String),
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Shape(pub smallvec::SmallVec<[DimExpr; 4]>);
