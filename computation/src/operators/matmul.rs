use super::OperatorInference;

pub struct MatMul {}

impl OperatorInference for MatMul {
    fn infer(&self, inputs: &Vec<crate::edge::Edge>) -> super::InferResult {
        // Ok(inputs)
        todo!()
    }
}
