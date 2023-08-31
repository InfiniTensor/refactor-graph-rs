// #![deny(warnings)]

mod onnx;

use std::{collections::HashMap, ptr::null_mut};

pub use onnx::{load_model, LoadError};

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct Tensor {
    dt: common::DataType,
    shape: Shape,
    data: *mut u8,
}

impl Default for Tensor {
    #[inline]
    fn default() -> Self {
        Self {
            dt: common::DataType::UNDEFINED,
            shape: Default::default(),
            data: null_mut(),
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Shape(smallvec::SmallVec<[i64; 4]>);
