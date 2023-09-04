mod eq;
mod gemm;
mod infer;
mod matmul;
mod op_type;

use eq::OperatorEq;
use infer::OutputInference;

pub use gemm::Gemm;
pub use infer::InferError;
pub use matmul::MatMul;
pub use op_type::OpType;

/// 算子。
pub trait Operator: OutputInference + OperatorEq {}
