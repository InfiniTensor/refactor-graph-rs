use common::DataType;
use computation::{Attribute, Blob, DimExpr, Edge, Node, Operator, Shape, Tensor};
use graph_topo::{Builder as GraphBuilder, Graph};
use internal::{
    attribute_proto::AttributeType, tensor_proto, tensor_proto::DataLocation, tensor_shape_proto,
    type_proto, AttributeProto, ModelProto, TensorProto, ValueInfoProto,
};
use prost::Message;
use std::{
    collections::{hash_map::Entry, HashMap},
    path::Path,
    rc::Rc,
    str::from_utf8,
};

#[allow(non_snake_case, clippy::enum_variant_names)]
mod internal {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/// Loads model from a file.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ModelProto, LoadError> {
    ModelProto::decode(std::fs::read(path).map_err(LoadError::Io)?.as_slice())
        .map_err(LoadError::Prost)
}

/// Saves model to a file.
pub fn save_model<P: AsRef<Path>>(path: P, model: &ModelProto) -> Result<(), SaveError> {
    let mut body = Vec::new();
    model.encode(&mut body).map_err(SaveError::Prost)?;
    std::fs::write(path, body).map_err(SaveError::Io)
}

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Prost(prost::DecodeError),
}

#[derive(Debug)]
pub enum SaveError {
    Io(std::io::Error),
    Prost(prost::EncodeError),
}

pub fn model_to_graph(model: ModelProto) -> Graph<Node, Edge> {
    let mut builder = GraphBuilder::default();
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
            Rc::new(Operator {
                ty: format!("onnx::{}", node.op_type).parse().unwrap(),
                attributes: node.attribute.into_iter().map(take_attribute).collect(),
            }),
        );
    }
    for edge in graph.input {
        let (name, tensor) = build_tensor_without_data(edge);
        builder.global_inputs.push(name.clone());
        builder.edges.insert(name, tensor);
    }
    for edge in graph.output {
        let (name, tensor) = build_tensor_without_data(edge);
        builder.global_outputs.push(name.clone());
        builder.edges.insert(name, tensor);
    }
    for edge in graph.initializer {
        let (name, tensor) = build_tensor_with_data(edge);
        match builder.edges.entry(name) {
            Entry::Occupied(mut entry) => {
                assert!(entry.get().info_equal(&tensor));
                assert!(!entry.get().has_data());
                *entry.get_mut() = tensor;
            }
            Entry::Vacant(entry) => {
                entry.insert(tensor);
            }
        }
    }

    builder.build()
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

fn build_tensor_without_data(value: ValueInfoProto) -> (String, Edge) {
    use tensor_shape_proto::dimension::Value::{DimParam, DimValue};
    let type_proto::Value::TensorType(t) = value.r#type.unwrap().value.unwrap() else {
        todo!()
    };
    (
        value.name,
        Rc::new(Tensor::without_data(
            match_dt(t.elem_type),
            Shape(
                t.shape
                    .unwrap()
                    .dim
                    .into_iter()
                    .map(|d| match d.value {
                        Some(DimValue(val)) => DimExpr::Value(val),
                        Some(DimParam(p)) => DimExpr::Variable(p),
                        None => unreachable!(),
                    })
                    .collect(),
            ),
        )),
    )
}

fn build_tensor_with_data(tensor: TensorProto) -> (String, Edge) {
    assert_eq!(tensor.data_location(), DataLocation::Default);
    let dt = match_dt(tensor.data_type);
    let size = tensor.dims.iter().product::<i64>() as usize;

    macro_rules! copy_data {
        ($data:ident) => {{
            let src = if tensor.raw_data.is_empty() {
                assert_eq!(tensor.$data.len(), size);
                tensor.$data.as_ptr() as *const u8
            } else {
                assert_eq!(tensor.raw_data.len(), size * dt.layout().size());
                tensor.raw_data.as_ptr()
            };
            let layout = dt.array_layout(size);
            let ans = Rc::new(Blob::new(layout, src));

            ans
        }};
    }

    (
        tensor.name,
        Rc::new(Tensor::with_data(
            dt,
            Shape(tensor.dims.into_iter().map(DimExpr::Value).collect()),
            match dt {
                DataType::F32 => copy_data!(float_data),
                DataType::I32 => copy_data!(int32_data),
                DataType::I64 => copy_data!(int64_data),
                DataType::F64 => copy_data!(double_data),
                DataType::U64 => copy_data!(uint64_data),
                dt => todo!("data type {dt:?} not supported yet"),
            },
        )),
    )
}

fn match_dt(dt: i32) -> DataType {
    use tensor_proto::DataType::*;
    match tensor_proto::DataType::from_i32(dt) {
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
    }
}

#[test]
fn test() -> std::io::Result<()> {
    use std::{env::current_dir, ffi::OsStr, fs::read_dir};
    let n = current_dir()
        .and_then(read_dir)?
        .filter_map(|res| res.ok())
        .filter(|file| file.path().extension() == Some(OsStr::new("onnx")))
        .map(|x| model_to_graph(load_model(x.path()).unwrap()))
        .count();
    println!("{n} onnx model(s) loaded");
    Ok(())
}
