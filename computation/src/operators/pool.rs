use super::{
    infer::{InferResult, OutputInference},
    m::impl_op,
};
use crate::{DimExpr, Edge, InferError, Shape, Tensor};
use smallvec::{smallvec, SmallVec};

/// 单目算子。
#[derive(PartialEq, Eq, Debug)]
pub struct Pool {
    pub(super) ty: PoolOpType,
    pub(super) ceil_mode: bool,
    pub(super) kernel_shape: SmallVec<[i64; 2]>,
    pub(super) attrs: PoolAttributes,
}

impl_op!(Pool);

impl OutputInference for Pool {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 1 {
            Err(InferError::InputsLenMismatch)
        } else {
            let input = inputs[0].as_ref();
            let dt = input.data_type();
            if !dt.is_ieee754() {
                return Err(InferError::DataTypeMismatch);
            }
            let dim = input.shape().0.len();
            if dim != self.kernel_shape.len() + 2 {
                return Err(InferError::ShapeMismatch);
            }
            let Some(ans) = infer_pool(&input.shape().0[2..], &self.kernel_shape, &self.attrs)
            else {
                return Err(InferError::ShapeMismatch);
            };
            Ok(vec![Edge::new(Tensor::without_data(dt, ans))])
        }
    }
}

/// 单目运算类型。
#[derive(PartialEq, Eq, Debug)]
pub(super) enum PoolOpType {
    Average,
}

#[derive(PartialEq, Eq, Debug)]
pub(super) struct PoolAttributes {
    pub dilations: Option<SmallVec<[i64; 2]>>,
    pub pads: Option<SmallVec<[i64; 4]>>,
    pub strides: Option<SmallVec<[i64; 2]>>,
}

pub(super) fn infer_pool(
    input: &[DimExpr],
    kernel: &[i64],
    attrs: &PoolAttributes,
) -> Option<Shape> {
    let dim = input.len();
    if dim != kernel.len() {
        return None;
    }

    let dilations = if let Some(dilations) = attrs.dilations.clone() {
        if dilations.len() != dim {
            return None;
        }
        dilations
    } else {
        smallvec![1; dim]
    };
    let pads = if let Some(pads) = attrs.pads.clone() {
        if pads.len() != dim * 2 {
            return None;
        }
        pads
    } else {
        smallvec![0; dim * 2]
    };
    let strides = if let Some(strides) = attrs.strides.clone() {
        if strides.len() != dim {
            return None;
        }
        strides
    } else {
        smallvec![1; dim]
    };

    let mut ans = Shape(smallvec![DimExpr::Value(0); dim]);
    for i in 0..dim {
        let input = match input[i] {
            DimExpr::Value(val) if val > 0 => val,
            DimExpr::Variable(_) => todo!(),
            _ => unreachable!(),
        };
        let d = input + pads[i] + pads[i + dim];
        let k = (kernel[i] - 1) * dilations[i] + 1;
        ans.0[i] = DimExpr::Value((d - k) / strides[i] + 1);
    }
    Some(ans)
}
