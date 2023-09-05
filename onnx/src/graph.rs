﻿use computation::{smallvec::SmallVec, Edge, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Graph(pub(crate) graph_topo::Graph<Operator, Tensor>);

#[derive(Clone, Debug)]
pub struct Operator {
    pub(crate) ty: String,
    pub(crate) attributes: HashMap<String, Attribute>,
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

impl Graph {
    pub fn to_computation(self) -> computation::Graph {
        let Self(graph_topo::Graph {
            topology,
            nodes,
            edges,
        }) = self;
        computation::Graph(graph_topo::Graph {
            topology,
            nodes: nodes
                .into_iter()
                .map(|op| match op.ty.as_str() {
                    "Abs" => computation::abs(),
                    "Acos" => computation::acos(),
                    "Acosh" => computation::acosh(),
                    "Add" => computation::add(),
                    "AffineGrid" => computation::affine_grid(),
                    "And" => computation::and(),
                    "ArgMax" => computation::arg_max(),
                    "ArgMin" => computation::arg_min(),
                    "Asin" => computation::asin(),
                    "Asinh" => computation::asinh(),
                    "Atan" => computation::atan(),
                    "Atanh" => computation::atanh(),
                    "AveragePool" => {
                        macro_rules! take_ints {
                            () => {
                                |attr| match attr {
                                    Attribute::Ints(v) => SmallVec::from(v.as_slice()),
                                    _ => unreachable!(),
                                }
                            };
                        }
                        computation::average_pool(
                            !matches!(
                                op.attributes.get("ceil_mode"),
                                Some(Attribute::Int(0)) | None
                            ),
                            op.attributes.get("dilations").map(take_ints!()),
                            op.attributes.get("kernel_shape").map(take_ints!()).unwrap(),
                            op.attributes.get("pads").map(take_ints!()),
                            op.attributes.get("strides").map(take_ints!()),
                        )
                    }
                    "BatchNormalization" => computation::batch_normalization(),
                    _ => todo!(),
                })
                .collect(),
            edges: edges.into_iter().map(Edge::new).collect(),
        })
    }
}