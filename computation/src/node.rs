use crate::{Graph, OpType};
use std::collections::HashMap;

pub enum Node {
    Op {
        op_type: OpType,
        attributes: HashMap<String, Attribute>,
    },
    Subgraph(Box<Graph>),
}

impl Default for Node {
    #[inline]
    fn default() -> Self {
        Self::Op {
            op_type: Default::default(),
            attributes: Default::default(),
        }
    }
}

#[derive(Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    Float(f64),
    Floats(Vec<f64>),
    String(String),
    Strings(Vec<String>),
}

impl From<Graph> for Node {
    fn from(value: Graph) -> Self {
        Self::Subgraph(Box::new(value))
    }
}
