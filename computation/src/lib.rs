//! 硬件无关算子库。

#![deny(warnings, missing_docs)]

mod graph;
mod operator;
mod tensor;

use std::{ffi::OsStr, path::Path};

pub use graph::Graph;
pub use operator::*;
pub use tensor::*;

/// 加载图。
pub fn load_graph(path: impl AsRef<Path>, name: impl AsRef<OsStr>) -> Graph {
    let path = path.as_ref();
    let name = name.as_ref().to_str().unwrap();
    let info = std::fs::read_to_string(path.join(format!("{name}.info"))).unwrap();
    let data = std::fs::read(path.join(format!("{name}.data"))).unwrap();
    Graph::from((info.as_str(), data))
}

// #[test]
// fn test_load_graph() {
//     println!(
//         "{}",
//         load_graph(Path::new(env!("CARGO_MANIFEST_DIR")), "graph")
//     );
// }
