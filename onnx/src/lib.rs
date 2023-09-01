// #![deny(warnings)]

mod onnx;
mod tensor;

use std::collections::HashMap;

pub use crate::onnx::{load_model, LoadError};
pub use tensor::{DimExpr, Shape, Tensor};

#[derive(Debug)]
pub struct Graph(graph_topo::Graph<Operator, Tensor>);

#[derive(Clone, Debug)]
pub struct Operator {
    op_type: String,
    attributes: HashMap<String, Attribute>,
}

#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
}
