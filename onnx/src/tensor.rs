use common::DataType;
use smallvec::SmallVec;
use std::ptr::NonNull;

#[derive(PartialEq, Eq, Debug)]
pub struct Tensor {
    dt: DataType,
    shape: Shape,
    data: Option<NonNull<u8>>,
}

impl Tensor {
    #[inline]
    pub fn with_data(dt: DataType, shape: Shape, data: *mut u8) -> Self {
        Self {
            dt,
            shape,
            data: NonNull::new(data),
        }
    }

    #[inline]
    pub fn without_data(dt: DataType, shape: Shape) -> Self {
        Self {
            dt,
            shape,
            data: None,
        }
    }

    #[inline]
    pub fn info_equal(&self, rhs: &Self) -> bool {
        self.dt == rhs.dt && self.shape == rhs.shape
    }

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
                .map(|x| match x {
                    DimExpr::Value(val) => val,
                    DimExpr::Variable(_) => unreachable!(),
                })
                .product::<i64>();
            unsafe { std::alloc::dealloc(ptr.as_ptr(), self.dt.array_layout(size as _)) };
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Shape(pub smallvec::SmallVec<[DimExpr; 4]>);

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DimExpr {
    Value(i64),
    Variable(String),
}
