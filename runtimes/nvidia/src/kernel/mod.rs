use crate::driver::{ContextGuard, Graph, RefDevicePtr};
use computation::{Operator, Tensor};
use std::alloc::Layout;

pub(crate) trait GraphBuilder {
    fn worksapce(&self) -> Layout;

    fn push_to(
        &self,
        graph: &mut Graph,
        resources: &Resources,
        workspace: &RefDevicePtr,
        inputs: &[RefDevicePtr],
        outputs: &[RefDevicePtr],
    );
}

pub(crate) trait GraphUser {
    fn builder(
        &self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        resources: &mut Resources,
        ctx: &ContextGuard,
    ) -> Box<dyn GraphBuilder>;
}

#[derive(Default, Debug)]
pub(crate) struct Resources {}

impl GraphUser for Operator {
    fn builder(
        &self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        resources: &mut Resources,
        ctx: &ContextGuard,
    ) -> Box<dyn GraphBuilder> {
        match self {
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
}
