use crate::edge::{Edge, Shape, ShapeVariable, Tensor};

use super::{InferResult, OperatorInference};

pub struct ElementWise {
    element_wise_type: ElementWiseType,
}

impl ElementWise {
    pub fn new(element_wise_type: ElementWiseType) -> Self {
        Self { element_wise_type }
    }
}

impl OperatorInference for ElementWise {
    fn infer(&self, inputs: &Vec<crate::edge::Edge>) -> InferResult {
        if inputs.len() != 2 {
            return Err(super::InferError::SizeMismatch);
        }
        let input0 = &inputs[0];
        let input1 = &inputs[1];
        match (input0, input1) {
            (Edge::Tensor(tensor0), Edge::Tensor(tensor1)) => {
                // TODO: check data type.
                let shape0 = tensor0.shape();
                let shape1 = tensor1.shape();
                let broadcasted_shape = super::infer_multi_broadcast(shape0, shape1)
                    .ok_or(super::InferError::BroadcastError)?;
                Ok(vec![Edge::Tensor(Tensor::new(
                    tensor0.data_type(),
                    broadcasted_shape,
                ))])
            }
            (Edge::ShapeVariable(shape_var0), Edge::ShapeVariable(shape_var1)) => {
                let shape0 = shape_var0.shape();
                let shape1 = shape_var1.shape();

                if shape0.size() != shape1.size() {
                    return Err(super::InferError::SizeMismatch);
                }
                let size = shape0.dims().len();
                // TODO: check shape.
                let mut dims = vec![None; 4];
                for (i, dim) in dims.iter_mut().enumerate().take(size) {
                    if shape0.dims()[i].is_some() && shape1.dims()[i].is_some() {
                        match self.element_wise_type {
                            ElementWiseType::Add => {
                                *dim = Some(shape0.dims()[i].unwrap() + shape1.dims()[i].unwrap())
                            }
                            ElementWiseType::Sub => {
                                *dim = Some(shape0.dims()[i].unwrap() - shape1.dims()[i].unwrap())
                            }
                            ElementWiseType::Mul => {
                                *dim = Some(shape0.dims()[i].unwrap() * shape1.dims()[i].unwrap())
                            }
                            ElementWiseType::Div => {
                                *dim = Some(shape0.dims()[i].unwrap() / shape1.dims()[i].unwrap())
                            }
                        }
                    } else if (shape0.dims()[i].is_some() && shape1.dims()[i].is_none())
                        || (shape0.dims()[i].is_none() && shape1.dims()[i].is_some())
                    {
                        return Err(super::InferError::SizeMismatch);
                    }
                }
                Ok(vec![Edge::ShapeVariable(ShapeVariable::new(Shape::new(
                    dims,
                )))])
            }
            _ => Err(super::InferError::DataTypeMismatch),
        }
    }
}

pub enum ElementWiseType {
    Add,
    Sub,
    Mul,
    Div,
}
