use crate::edge::Edge;

pub use self::element_wise::{ElementWise, ElementWiseType};
pub use self::infer::{infer_multi_broadcast, InferError, InferResult};
pub use self::op_type::OpType;
pub use self::unary::{UnaryOperator, UnaryType};

mod element_wise;
mod infer;
mod matmul;
mod op_type;
mod unary;

pub trait OperatorInference {
    fn infer(&self, inputs: &Vec<Edge>) -> InferResult;
}

pub struct Operator {
    op_type: OpType,
    info: Box<dyn OperatorInference>,
}
