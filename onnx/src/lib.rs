#![deny(warnings)]

mod onnx;
mod operators;

pub use onnx::{load_model, model_to_graph, save_model, LoadError, SaveError};
pub use operators::register_operators;
