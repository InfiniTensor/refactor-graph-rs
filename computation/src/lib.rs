//! 硬件无关算子库。

#![deny(warnings, missing_docs)]

mod graph;
mod operator;
mod tensor;

use std::path::Path;

pub use graph::Graph;
pub use operator::*;
pub use tensor::*;

/// 表示形状里的数值的数据类型。
#[allow(non_camel_case_types)]
pub type udim = u32;

/// 表示有符号的 [udim]，例如用负数表示反向。
#[allow(non_camel_case_types)]
pub type sdim = i32;

/// 表示 [udim] 的差的数据类型。
#[allow(non_camel_case_types)]
pub type ddim = i16;

/// 加载图。
pub fn load_graph(path: impl AsRef<Path>, name: &str) -> Graph {
    let path = path.as_ref();
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
