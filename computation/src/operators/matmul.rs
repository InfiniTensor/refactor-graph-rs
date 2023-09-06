use super::{
    infer::{InferResult, OutputInference},
    m::impl_op,
};
use crate::{Edge, InferError, Shape};

use smallvec::smallvec;

/// See <https://onnx.ai/onnx/operators/onnx__MatMul.html>.
#[derive(PartialEq, Eq, Debug)]
pub(super) struct MatMul;

impl_op!(MatMul);

impl OutputInference for MatMul {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            return Err(InferError::InputsLenMismatch);
        }

        // let a = inputs[0].as_ref();
        // let b = inputs[1].as_ref();
        // let rank_a = a.rank();
        // let rank_b = b.rank();

        // if rank_a < 2 || rank_b < 2 {
        //     return Err(InferError::RankMismatch);
        // }

        // let mut ans_dim = smallvec![];

        // // If both arguments are 2-D they are multiplied like conventional matrices.

        // let shape = Shape(ans_dim);
        // Ok(vec![Edge::new(crate::Tensor::without_data(
        //     inputs[0].data_type(),
        //     shape,
        // ))])

        todo!()
    }
}
