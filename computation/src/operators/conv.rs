use super::{
    infer::{InferResult, OutputInference},
    m::impl_op,
};
use crate::{DimExpr, Edge, InferError, Shape, Tensor};

use smallvec::smallvec;

#[derive(Debug, PartialEq, Eq)]
pub enum PaddingMode {
    Other,
    Valid,
    Same,
}

#[derive(PartialEq, Eq, Debug)]
pub struct Conv {
    /// padding mode
    padding: PaddingMode,
    /// Padding along height dimension.
    ph: i64,
    /// Padding along width dimension.
    pw: i64,
    /// Stride along height dimension.
    sh: i64,
    /// Stride along width dimension.
    sw: i64,
    /// Dilation along height dimension.
    dh: i64,
    /// Dilation along width dimension.
    dw: i64,
}

impl_op!(Conv);

impl OutputInference for Conv {
    fn infer(&self, inputs: &[Edge]) -> InferResult {
        if inputs.len() != 2 {
            return Err(InferError::InputsLenMismatch);
        }
        let input = inputs[0].as_ref();
        let weight = inputs[1].as_ref();

        let data_type = input.data_type();

        let input_shape = input.shape();
        let weight_shape = weight.shape();

        let n = input_shape.0[0].value()?;
        let h = input_shape.0[2].value()?;
        let w = input_shape.0[3].value()?;
        let f = input_shape.0[0].value()?;
        let s = input_shape.0[2].value()?;
        let r = input_shape.0[3].value()?;

        assert_eq!(input_shape.0[1].value()? % weight_shape.0[1].value()?, 0);

        let oh;
        let ow;
        let on = n;
        let oc = f;

        if self.padding == PaddingMode::Other {
            oh = (h - (r - self.sh) * self.dh + self.ph * 2) / self.sh;
            ow = (w - (s - self.sw) * self.dw + self.pw * 2) / self.sw;
        } else if self.padding == PaddingMode::Same {
            oh = h / self.sh;
            ow = w / self.sw;
        } else {
            let ph = 0;
            let pw = 0;
            oh = (h - (r - self.sh) * self.dh + ph * 2) / self.sh;
            ow = (w - (s - self.sw) * self.dw + pw * 2) / self.sw;
        }

        let shape = Shape(smallvec![
            DimExpr::Value(on),
            DimExpr::Value(oc),
            DimExpr::Value(oh),
            DimExpr::Value(ow)
        ]);

        Ok(vec![Edge::new(Tensor::without_data(data_type, shape))])
    }
}
