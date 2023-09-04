mod binary;
mod eq;
mod gemm;
mod global_pool;
mod infer;
mod matmul;
mod pool;
mod reshape;
mod unary;

use eq::OperatorEq;
use infer::OutputInference;

pub use binary::{Binary, BinaryOpType};
pub use gemm::Gemm;
pub use global_pool::GlobalPool;
pub use infer::InferError;
pub use matmul::MatMul;
pub use pool::Pool;
pub use unary::{Unary, UnaryOpType};

/// 算子。
pub trait Operator: OutputInference + OperatorEq {}

mod m {
    macro_rules! impl_op {
        ($op:ident) => {
            impl super::Operator for $op {}

            impl super::eq::Downcast for $op {
                #[inline]
                fn as_any(&self) -> &dyn std::any::Any {
                    self
                }
            }

            impl super::eq::OperatorEq for $op {
                #[inline]
                fn op_eq(&self, rhs: &dyn super::eq::OperatorEq) -> bool {
                    rhs.as_any()
                        .downcast_ref::<Self>()
                        .filter(|&rhs| self == rhs)
                        .is_some()
                }
            }
        };
    }

    pub(super) use impl_op;
}
