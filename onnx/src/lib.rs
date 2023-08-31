// #![deny(warnings)]

use std::collections::HashMap;

pub struct Graph(graph_topo::Graph<Operator, Tensor>);

#[derive(Clone, Debug)]
pub struct Operator {
    op_type: &'static str,
    attributes: HashMap<&'static str, Attribute>,
}

#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    Float(f64),
    Floats(Vec<f64>),
    String(String),
    Strings(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    dt: common::DataType,
    shape: Shape,
    data: *mut u8,
}

#[derive(Clone, Default, Debug)]
pub struct Shape(smallvec::SmallVec<[i64; 4]>);

#[allow(non_snake_case)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
