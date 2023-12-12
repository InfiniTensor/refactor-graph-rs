use std::{alloc::Layout, str::FromStr};

/// 与 Onnx 兼容的数据类型。
///
/// > <https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping>.
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum DataType {
    Undefined = 0,
    F32 = 1,
    U8 = 2,
    I8 = 3,
    U16 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    String = 8,
    Bool = 9,
    FP16 = 10,
    F64 = 11,
    U32 = 12,
    U64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BF16 = 16,
}

impl DataType {
    /// 获取数据类型的 [Layout]。
    pub const fn layout(&self) -> Layout {
        match self {
            DataType::Undefined => unreachable!(),
            DataType::F32 => Layout::new::<f32>(),
            DataType::U8 => Layout::new::<u8>(),
            DataType::I8 => Layout::new::<i8>(),
            DataType::U16 => Layout::new::<u16>(),
            DataType::I16 => Layout::new::<i16>(),
            DataType::I32 => Layout::new::<i32>(),
            DataType::I64 => Layout::new::<i64>(),
            DataType::String => unreachable!(),
            DataType::Bool => Layout::new::<bool>(),
            DataType::FP16 => Layout::new::<half::f16>(),
            DataType::F64 => Layout::new::<f64>(),
            DataType::U32 => Layout::new::<u32>(),
            DataType::U64 => Layout::new::<u64>(),
            DataType::Complex64 => todo!(),
            DataType::Complex128 => todo!(),
            DataType::BF16 => Layout::new::<half::bf16>(),
        }
    }

    /// 获取数组的 [Layout]。
    pub fn array_layout(&self, len: usize) -> Layout {
        match self {
            DataType::Undefined => unreachable!(),
            DataType::F32 => Layout::array::<f32>(len).unwrap(),
            DataType::U8 => Layout::array::<u8>(len).unwrap(),
            DataType::I8 => Layout::array::<i8>(len).unwrap(),
            DataType::U16 => Layout::array::<u16>(len).unwrap(),
            DataType::I16 => Layout::array::<i16>(len).unwrap(),
            DataType::I32 => Layout::array::<i32>(len).unwrap(),
            DataType::I64 => Layout::array::<i64>(len).unwrap(),
            DataType::String => unreachable!(),
            DataType::Bool => Layout::array::<bool>(len).unwrap(),
            DataType::FP16 => todo!(),
            DataType::F64 => Layout::array::<f64>(len).unwrap(),
            DataType::U32 => Layout::array::<u32>(len).unwrap(),
            DataType::U64 => Layout::array::<u64>(len).unwrap(),
            DataType::Complex64 => todo!(),
            DataType::Complex128 => todo!(),
            DataType::BF16 => todo!(),
        }
    }

    /// 判断是否数字数据类型。
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

    /// 判断是否整数数据类型。
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

    /// 判断是否 IEEE754 浮点数数据类型。
    #[inline]
    pub const fn is_ieee754(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F64 | DataType::FP16)
    }

    /// 判断是否浮点数数据类型。
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F64 | DataType::FP16 | DataType::BF16
        )
    }

    /// 判断是否布尔数据类型。
    #[inline]
    pub const fn is_bool(&self) -> bool {
        matches!(self, DataType::Bool)
    }
}

impl FromStr for DataType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fp16" => Ok(DataType::FP16),
            "bf16" => Ok(DataType::BF16),
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),

            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),

            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" => Ok(DataType::I32),
            "i64" => Ok(DataType::I64),

            "bool" => Ok(DataType::Bool),
            "string" => Ok(DataType::String),
            "complex64" => Ok(DataType::Complex64),
            "complex128" => Ok(DataType::Complex128),

            "undefined" => Ok(DataType::Undefined),
            _ => Err(()),
        }
    }
}

/// 将 rust 类型转换为 [DataType]。
pub trait AsDataType {
    /// 获取 rust 类型对应的 [DataType]。
    fn as_data_type() -> DataType;
}

macro_rules! impl_as_data_type_for {
    ($t:ty, $dt:ident) => {
        impl AsDataType for $t {
            #[inline]
            fn as_data_type() -> DataType {
                DataType::$dt
            }
        }
    };
}

impl_as_data_type_for!(f32, F32);
impl_as_data_type_for!(u8, U8);
impl_as_data_type_for!(i8, I8);
impl_as_data_type_for!(u16, U16);
impl_as_data_type_for!(i16, I16);
impl_as_data_type_for!(i32, I32);
impl_as_data_type_for!(i64, I64);
impl_as_data_type_for!(String, String);
impl_as_data_type_for!(bool, Bool);
impl_as_data_type_for!(half::f16, FP16);
impl_as_data_type_for!(f64, F64);
impl_as_data_type_for!(u32, U32);
impl_as_data_type_for!(u64, U64);
impl_as_data_type_for!(half::bf16, BF16);
