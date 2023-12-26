#![allow(unused_variables)]

use crate::RoutineWorkspace;
use computation::{Operator, Tensor};

pub(super) fn lower(op: &Operator, inputs: &[&Tensor], outputs: &[&Tensor]) -> RoutineWorkspace {
    match op {
        Operator::BatchNormalization { epsilon } => todo!(),
        Operator::Broadcast => todo!(),
        Operator::Cast => todo!(),
        Operator::Clip => todo!(),
        Operator::Concat(_) => todo!(),
        Operator::Conv(_) => todo!(),
        Operator::Gather(_) => todo!(),
        Operator::GlobalPool => todo!(),
        Operator::MatMul {
            alpha,
            beta,
            transpose_a,
            transpose_b,
        } => todo!(),
        Operator::Pool(_) => todo!(),
        Operator::Reduce(_) => todo!(),
        Operator::Select(_) => todo!(),
        Operator::SimpleBinary(_) => todo!(),
        Operator::SimpleUnary(_) => todo!(),
        Operator::Slice(_) => todo!(),
        Operator::Softmax(_) => todo!(),
        Operator::Split(_) => todo!(),
        Operator::Transpose(_) => todo!(),
        Operator::Where => todo!(),
    }
}
