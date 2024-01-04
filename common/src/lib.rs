//!

#![deny(warnings, missing_docs)]

mod data_type;

pub use data_type::{AsDataType, DataType};

/// 表示形状里的数值的数据类型。
#[allow(non_camel_case_types)]
pub type udim = u32;

/// 表示有符号的 [udim]，例如用负数表示反向。
#[allow(non_camel_case_types)]
pub type sdim = i32;

/// 表示 [udim] 的差的数据类型。
#[allow(non_camel_case_types)]
pub type ddim = i16;
