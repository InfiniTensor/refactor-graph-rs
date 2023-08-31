#![allow(non_snake_case)]

use crate::{Attribute, Graph, Operator, Shape, Tensor};
use attribute_proto::AttributeType;
use common::DataType;
use graph_topo::Builder as GraphBuilder;
use prost::Message;
use std::collections::{hash_map::Entry, HashMap};
use std::path::Path;
use std::str::from_utf8;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Prost(prost::DecodeError),
}

/// Opens model from a file.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ModelProto, LoadError> {
    ModelProto::decode(
        std::fs::read(path)
            .map_err(|e| LoadError::Io(e))?
            .as_slice(),
    )
    .map_err(|e| LoadError::Prost(e))
}

impl From<ModelProto> for Graph {
    fn from(model: ModelProto) -> Self {
        let mut builder = GraphBuilder::<String, Operator, String, Tensor>::new();
        let graph = model.graph.unwrap();

        let mut name_record = HashMap::new();
        for node in graph.node {
            let name = match name_record.entry(node.name.clone()) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += 1;
                    format!("{}_{}", entry.key(), entry.get())
                }
                Entry::Vacant(entry) => {
                    let name = entry.key().clone();
                    entry.insert(0);
                    name
                }
            };
            builder
                .topology
                .insert(name.clone(), (node.input.clone(), node.output.clone()));
            builder.nodes.insert(
                name,
                Operator {
                    op_type: node.op_type,
                    attributes: node.attribute.into_iter().map(take_attribute).collect(),
                },
            );
        }
        for edge in graph.input {
            let (name, tensor) = build_tensor(edge);
            builder.global_inputs.push(name.clone());
            builder.edges.insert(name, tensor);
        }
        for edge in graph.output {
            let (name, tensor) = build_tensor(edge);
            builder.global_outputs.push(name.clone());
            builder.edges.insert(name, tensor);
        }
        for edge in graph.initializer {
            todo!("initializer: {edge:?}");
        }

        Self(builder.build())
    }
}

fn take_attribute(attr: AttributeProto) -> (String, Attribute) {
    use AttributeType::*;
    let value = match AttributeType::from_i32(attr.r#type).unwrap() {
        Int => Attribute::Int(attr.i),
        Float => Attribute::Float(attr.f),
        String => Attribute::String(from_utf8(&attr.s).unwrap().into()),
        Ints => Attribute::Ints(attr.ints),
        Floats => Attribute::Floats(attr.floats),
        Strings => Attribute::Strings(
            attr.strings
                .into_iter()
                .map(|s| from_utf8(&s).unwrap().into())
                .collect(),
        ),
        _ => todo!(),
    };
    (attr.name, value)
}

fn build_tensor(value: ValueInfoProto) -> (String, Tensor) {
    use tensor_proto::DataType::*;
    use tensor_shape_proto::dimension::Value::DimValue;

    let mut tensor = Tensor::default();
    let type_proto::Value::TensorType(t) = value.r#type.unwrap().value.unwrap() else {
        todo!()
    };
    tensor.dt = match tensor_proto::DataType::from_i32(t.elem_type) {
        Some(Undefined) => DataType::UNDEFINED,
        Some(Float) => DataType::F32,
        Some(Uint8) => DataType::U8,
        Some(Int8) => DataType::I8,
        Some(Uint16) => DataType::U16,
        Some(Int16) => DataType::I16,
        Some(Int32) => DataType::I32,
        Some(Int64) => DataType::I64,
        Some(String) => DataType::STRING,
        Some(Bool) => DataType::BOOL,
        Some(Float16) => DataType::FP16,
        Some(Double) => DataType::F64,
        Some(Uint32) => DataType::U32,
        Some(Uint64) => DataType::U64,
        Some(Complex64) => DataType::COMPLEX64,
        Some(Complex128) => DataType::COMPLEX128,
        Some(Bfloat16) => DataType::BF16,
        _ => todo!(),
    };
    tensor.shape = Shape(
        t.shape
            .unwrap()
            .dim
            .into_iter()
            .map(|d| match d.value {
                Some(DimValue(d)) => d,
                _ => todo!(),
            })
            .collect(),
    );
    (value.name, tensor)
}

// #[test]
// fn test() {
//     let model =
//         load_model("*.onnx")
//             .unwrap();
//     println!("{:#?}", Graph::from(model));
// }
