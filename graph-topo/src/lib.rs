//! 图拓扑表示。
//!
//! 这个库定义了抽象的图拓扑结构。
//! 图拓扑结构是一种支持重边的有向无环图结构。
//! 与一般的有向无环图不同的是，图拓扑结构中连接到节点的边是有序的。
//! 换句话说，图拓扑结构中不能仅用入度、出度来描述节点的连接关系，而是可以定义某个节点的第几个入边，第几个出边等。
//! 这种结构可以很好地表示编译器中常见的数据流图（SSA IR）。
//!
//! 这个库提供 3 种重要的数据结构：
//! - [`Graph`]：包含节点和边信息的完整图表示，这个结构通过泛型支持任意类型的节点和边，可以作为具体图类型的基础。
//! - [`GraphTopo`]：一种轻量化的图拓朴表示，剥离了节点和边信息，专注于表示连接关系。可以用这个结构来复制和传递图拓朴。
//! - [`Builder`]：图建造者，包含容易操作的简单结构表示，开发者可以先填写建造者，由建造者构造出图。

#![deny(warnings, missing_docs)]

mod builder;
mod container;

/// 表示对象数量的基本类型。
#[allow(non_camel_case_types)]
pub type ucount = u32;
pub use builder::Builder;
pub use container::{EdgeIndices, EdgeRange, EdgeRc, Graph, GraphTopo, Iter};

#[test]
fn test() {
    use std::collections::HashMap;
    //  `a` (b)      `0` (1)
    //    \ /          \ /
    // ┌--[A]--┐    ┌--[0]--┐
    // └(c)-(d)┘(e) └(2)-(3)┘(4)
    //   |    \ /     |    \ /
    //   |   ┌[B]┐    |   ┌[1]┐
    //   |   └(f)┘    |   └(5)┘
    //    \   /        \   /
    //     \ /          \ /
    //    ┌[C]┐        ┌[2]┐
    //    └`z`┘        └`6`┘
    let Graph {
        topology,
        nodes,
        edges,
    } = Builder {
        topology: HashMap::from([
            ("A", (vec!["a", "b"], vec!["c", "d"])),
            ("B", (vec!["d", "e"], vec!["f"])),
            ("C", (vec!["f", "c"], vec!["z"])),
        ]),
        global_inputs: vec!["a"],
        global_outputs: vec!["z"],
        nodes: HashMap::from([("A", "*0"), ("B", "*1"), ("C", "*2")]),
        edges: HashMap::from([("a", "|0"), ("b", "|1"), ("e", "|4"), ("z", "!")]),
    }
    .build();

    // assert structure -----------------------------------

    assert_eq!(topology.global_inputs_len, 1); // a
    assert_eq!(topology.global_outputs_len, 1); // z

    // indices | 0 | 1 | 2
    //   nodes | A | B | C
    //  locals | 1 | 1 | 0
    //  inputs | 2 | 2 | 2
    // outputs | 2 | 1 | 1
    assert_eq!(nodes, vec!["*0", "*1", "*2"]);
    assert_eq!(
        topology.nodes,
        vec![
            container::Node {
                local_edges_len: 1,
                inputs_len: 2,
                outputs_len: 2,
            },
            container::Node {
                local_edges_len: 1,
                inputs_len: 2,
                outputs_len: 1,
            },
            container::Node {
                local_edges_len: 0,
                inputs_len: 2,
                outputs_len: 1,
            },
        ]
    );
    //     indices | 0 | 1 | 2 | 3 | 4 | 5 | 6
    //       edges | a | b | c | d | e | f | z
    // connections | z | a | b | d | e | f | c
    assert_eq!(edges, vec!["|0", "|1", "", "", "|4", "", "!"]);
    assert_eq!(topology.connections, vec![6, 0, 1, 3, 4, 5, 2,]);

    // assert api -----------------------------------
    assert_eq!(topology.nodes_len(), 3);
    assert_eq!(topology.calculate_edges_len(), 7);
    assert_eq!(topology.global_inputs_len(), 1);
    assert_eq!(topology.global_outputs_len(), 1);
    assert_eq!(topology.global_inputs(), 0..1);
    assert_eq!(&*topology.global_outputs(), &[6]);
    for (i, inputs, outputs) in topology.into_iter() {
        match i {
            0 => {
                assert_eq!(&*inputs, &[0, 1]);
                assert_eq!(outputs, 2..4);
            }
            1 => {
                assert_eq!(&*inputs, &[3, 4]);
                assert_eq!(outputs, 5..6);
            }
            2 => {
                assert_eq!(&*inputs, &[5, 2]);
                assert_eq!(outputs, 6..7);
            }
            _ => unreachable!(),
        }
    }
}
