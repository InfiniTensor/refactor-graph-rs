use computation::{Edge, InferResult, OpType, Operator};
use std::collections::BTreeMap;

pub fn register_operators() {
    let x = BTreeMap::from([
        ("onnx::Abs", infer_unary as _),
        ("onnx::Acos", infer_unary as _),
        ("onnx::Acosh", infer_unary as _),
        ("onnx::Asin", infer_unary as _),
        ("onnx::Asinh", infer_unary as _),
        ("onnx::Atan", infer_unary as _),
        ("onnx::Atanh", infer_unary as _),
        ("onnx::Cos", infer_unary as _),
        ("onnx::Cosh", infer_unary as _),
        ("onnx::Sin", infer_unary as _),
        ("onnx::Sinh", infer_unary as _),
        ("onnx::Tan", infer_unary as _),
        ("onnx::Tanh", infer_unary as _),
        ("onnx::Relu", infer_unary as _),
        ("onnx::PRelu", infer_unary as _),
        ("onnx::Sqrt", infer_unary as _),
        ("onnx::Reshape", infer_reshape as _),
        ("onnx::Add", infer_arithmetic as _),
        ("onnx::Sub", infer_arithmetic as _),
        ("onnx::Mul", infer_arithmetic as _),
        ("onnx::Div", infer_arithmetic as _),
        ("onnx::Gemm", infer_gemm as _),
        ("onnx::MatMul", infer_mat_mul as _),
        ("onnx::CumSum", infer_cum_sum as _),
        ("onnx::Slice", infer_slice as _),
        ("onnx::Shape", infer_shape as _),
        ("onnx::Where", infer_where as _),
        ("onnx::Squeeze", infer_squeeze as _),
        ("onnx::Unsqueeze", infer_squeeze as _),
        ("onnx::Equal", infer_equal as _),
        ("onnx::Softmax", infer_softmax as _),
        ("onnx::Pow", infer_pow as _),
        ("onnx::Reduce", infer_reduce as _),
        ("onnx::Concat", infer_concat as _),
        ("onnx::Gather", infer_gather as _),
        ("onnx::Cast", infer_cast as _),
        ("onnx::Max", infer_max as _),
        ("onnx::Transpose", infer_transpose as _),
        ("onnx::Expand", infer_expand as _),
    ]);
    OpType::register(x);
}

fn infer_unary(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_reshape(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_arithmetic(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_gemm(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_mat_mul(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_cum_sum(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_slice(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_shape(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_where(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_squeeze(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_equal(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_softmax(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_pow(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_reduce(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_concat(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_gather(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_cast(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_max(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_transpose(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
fn infer_expand(_op: &Operator, _inputs: Vec<Edge>) -> InferResult {
    todo!()
}
