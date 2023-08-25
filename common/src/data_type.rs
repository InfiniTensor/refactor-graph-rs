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
    pub const fn size(&self) -> usize {
        use core::mem::size_of;
        match self {
            DataType::UNDEFINED => unreachable!(),
            DataType::F32 => size_of::<f32>(),
            DataType::U8 => size_of::<u8>(),
            DataType::I8 => size_of::<i8>(),
            DataType::U16 => size_of::<u16>(),
            DataType::I16 => size_of::<i16>(),
            DataType::I32 => size_of::<i32>(),
            DataType::I64 => size_of::<i64>(),
            DataType::STRING => unreachable!(),
            DataType::BOOL => size_of::<bool>(),
            DataType::FP16 => 2,
            DataType::F64 => size_of::<f64>(),
            DataType::U32 => size_of::<u32>(),
            DataType::U64 => size_of::<u64>(),
            DataType::COMPLEX64 => todo!(),
            DataType::COMPLEX128 => todo!(),
            DataType::BF16 => 2,
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
