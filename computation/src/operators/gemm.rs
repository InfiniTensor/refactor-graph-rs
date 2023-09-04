use super::{
    eq::{Downcast, OperatorEq},
    infer::{InferError, InferResult, OutputInference},
    Operator,
};
use crate::{operators::infer::uinidir_broadcast, Edge, Shape, Tensor};
use smallvec::smallvec;
use std::{any::Any, rc::Rc};

#[derive(PartialEq, Debug)]
pub struct Gemm {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl Operator for Gemm {}

impl Downcast for Gemm {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl OperatorEq for Gemm {
    #[inline]
    fn op_eq(&self, rhs: &dyn OperatorEq) -> bool {
        rhs.as_any()
            .downcast_ref::<Self>()
            .filter(|&rhs| self == rhs)
            .is_some()
    }
}

impl OutputInference for Gemm {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if !matches!(inputs.len(), 2 | 3) {
            Err(InferError::InputsLenMismatch)
        } else {
            let a = inputs[0].as_ref();
            let b = inputs[1].as_ref();
            let dt = a.data_type();
            if !dt.is_numeric() || b.data_type() != dt {
                return Err(InferError::DataTypeMismatch);
            }
            if a.shape().0.len() != 2 || b.shape().0.len() != 2 {
                return Err(InferError::SizeMismatch);
            }

            let a = a.shape().0.as_slice();
            let b = b.shape().0.as_slice();
            let (m, k) = if self.trans_a {
                (&a[1], &a[0])
            } else {
                (&a[0], &a[1])
            };
            let (k_, n) = if self.trans_b {
                (&b[1], &b[0])
            } else {
                (&b[0], &b[1])
            };
            if k != k_ {
                return Err(InferError::SizeMismatch);
            }

            let ans_shape = Shape(smallvec![m.clone(), n.clone()]);
            if inputs.len() == 3 {
                let c = inputs[2].as_ref();
                if c.data_type() != dt {
                    return Err(InferError::DataTypeMismatch);
                }
                if c.shape().0.len() != 2 || !uinidir_broadcast(&ans_shape, c.shape()) {
                    return Err(InferError::SizeMismatch);
                }
            }

            Ok(vec![Rc::new(Tensor::without_data(dt, ans_shape))])
        }
    }
}
