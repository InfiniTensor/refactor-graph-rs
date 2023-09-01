// #![deny(warnings)]

mod onnx;

use computation::Tensor;
use std::collections::HashMap;

pub use crate::onnx::{load_model, save_model, LoadError, SaveError};

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
