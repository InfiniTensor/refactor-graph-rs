#![deny(warnings)]

mod graph;
mod onnx;

pub use graph::{Attribute, Graph, Operator};
pub use onnx::{load_model, save_model, LoadError, SaveError};
