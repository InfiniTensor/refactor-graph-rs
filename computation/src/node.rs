use std::collections::HashMap;

use crate::{Graph, OpType};

pub enum Node {
    Op(Operator),
    Subgraph(Subgraph),
}

pub struct Operator {
    pub op_type: OpType,
    pub attributes: Attributes,
}

pub struct Subgraph(Box<Graph>);

pub struct Attributes(HashMap<String, Attribute>);

pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Vec<u8>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Vec<u8>>),
}
