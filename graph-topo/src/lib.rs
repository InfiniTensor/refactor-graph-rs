#![deny(warnings)]

mod container;
mod searcher;

pub use container::{EdgeRef, GraphTopo, NodeRef};
pub use searcher::{Node, Nodes, Searcher as GraphTopoSearcher};
