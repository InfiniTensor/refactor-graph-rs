use crate::Edge;

pub use self::infer::{InferError, InferResult};
pub use self::op_type::OpType;

mod infer;
mod op_type;

/// 算子。
pub trait Operator {
    /// 根据算子输入推导输出。
    fn infer(&self, inputs: &[Edge]) -> InferResult;
}
