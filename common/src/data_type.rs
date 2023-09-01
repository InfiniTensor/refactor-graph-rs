use std::alloc::Layout;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DataType {
    UNDEFINED = 0,
    F32 = 1,
    U8 = 2,
    I8 = 3,
    U16 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    STRING = 8,
    BOOL = 9,
    FP16 = 10,
    F64 = 11,
    U32 = 12,
    U64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BF16 = 16,
}

impl DataType {
    pub const fn layout(&self) -> Layout {
        match self {
            DataType::UNDEFINED => unreachable!(),
            DataType::F32 => Layout::new::<f32>(),
            DataType::U8 => Layout::new::<u8>(),
            DataType::I8 => Layout::new::<i8>(),
            DataType::U16 => Layout::new::<u16>(),
            DataType::I16 => Layout::new::<i16>(),
            DataType::I32 => Layout::new::<i32>(),
            DataType::I64 => Layout::new::<i64>(),
            DataType::STRING => unreachable!(),
            DataType::BOOL => Layout::new::<bool>(),
            DataType::FP16 => todo!(),
            DataType::F64 => Layout::new::<f64>(),
            DataType::U32 => Layout::new::<u32>(),
            DataType::U64 => Layout::new::<u64>(),
            DataType::COMPLEX64 => todo!(),
            DataType::COMPLEX128 => todo!(),
            DataType::BF16 => todo!(),
        }
    }

    pub fn array_layout(&self, len: usize) -> Layout {
        match self {
            DataType::UNDEFINED => unreachable!(),
            DataType::F32 => Layout::array::<f32>(len).unwrap(),
            DataType::U8 => Layout::array::<u8>(len).unwrap(),
            DataType::I8 => Layout::array::<i8>(len).unwrap(),
            DataType::U16 => Layout::array::<u16>(len).unwrap(),
            DataType::I16 => Layout::array::<i16>(len).unwrap(),
            DataType::I32 => Layout::array::<i32>(len).unwrap(),
            DataType::I64 => Layout::array::<i64>(len).unwrap(),
            DataType::STRING => unreachable!(),
            DataType::BOOL => Layout::array::<bool>(len).unwrap(),
            DataType::FP16 => todo!(),
            DataType::F64 => Layout::array::<f64>(len).unwrap(),
            DataType::U32 => Layout::array::<u32>(len).unwrap(),
            DataType::U64 => Layout::array::<u64>(len).unwrap(),
            DataType::COMPLEX64 => todo!(),
            DataType::COMPLEX128 => todo!(),
            DataType::BF16 => todo!(),
        }
    }

    #[inline]
    pub const fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::F32
                | DataType::U8
                | DataType::I8
                | DataType::U16
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::FP16
                | DataType::F64
                | DataType::U32
                | DataType::U64
                | DataType::BF16
        )
    }

    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::U8
                | DataType::I8
                | DataType::U16
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U32
                | DataType::U64
        )
    }

    #[inline]
    pub const fn is_ieee754(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F64 | DataType::FP16)
    }

    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F64 | DataType::FP16 | DataType::BF16
        )
    }

    #[inline]
    pub const fn is_bool(&self) -> bool {
        matches!(self, DataType::BOOL)
    }
}

pub trait AsDataType {
    fn as_data_type() -> DataType;
}

impl AsDataType for f32 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::F32
    }
}

impl AsDataType for u8 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::U8
    }
}

impl AsDataType for i8 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::I8
    }
}

impl AsDataType for u16 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::U16
    }
}

impl AsDataType for i16 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::I16
    }
}

impl AsDataType for i32 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::I32
    }
}

impl AsDataType for i64 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::I64
    }
}

impl AsDataType for bool {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::BOOL
    }
}

impl AsDataType for f64 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::F64
    }
}

impl AsDataType for u32 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::U32
    }
}

impl AsDataType for u64 {
    #[inline]
    fn as_data_type() -> DataType {
        DataType::U64
    }
}
