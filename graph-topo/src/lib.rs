//! 图拓扑表示。
//!
//! 这个库定义了抽象的图拓扑结构。
//! 图拓扑结构是一种支持重边的有向无环图结构。
//! 与一般的有向无环图不同的是，图拓扑结构中连接到节点的边是有序的。
//! 换句话说，图拓扑结构中不能仅用入度、出度来描述节点的连接关系，而是可以定义某个节点的第几个入边，第几个出边等。
//! 这种结构可以很好地表示编译器中常见的数据流图（SSA IR）。
//!
//! 这个库提供 4 种重要的数据结构：
//! - [`GraphTopo`]：一种轻量化的图拓朴表示，剥离了节点和边信息，专注于表示连接关系。可以用这个结构来复制和传递图拓朴。
//! - [`Graph`]：包含节点和边信息的完整图表示，这个结构通过泛型支持任意类型的节点和边，可以作为具体图类型的基础。
//! - [`Searcher`]：对图拓扑的索引表示，其中缓存了最丰富的结构信息，用于快速查询复杂的连接关系。索引器可以从图拓朴中快速地构造出来。
//! - [`Builder`]：图建造者，包含容易操作的简单结构表示，开发者可以先填写建造者，由建造者构造出图。

#![deny(warnings, missing_docs)]

mod container;
mod searcher;

pub use container::{Graph, GraphTopo};
pub use searcher::{Edge, Node, Searcher};

// #[test]
// fn test() {
//     use std::collections::{HashMap, HashSet};

//     let graph = Builder {
//         topology: HashMap::from([
//             ("A", (vec!["a", "b"], vec!["c", "d"])),
//             ("B", (vec!["d", "e"], vec!["f"])),
//             ("C", (vec!["f", "c"], vec!["z"])),
//         ]),
//         global_inputs: vec!["a"],
//         global_outputs: vec!["z"],
//         nodes: HashMap::from([("A", "*0"), ("B", "*1"), ("C", "*2")]),
//         edges: HashMap::from([("a", "|0"), ("b", "|1"), ("e", "|4"), ("z", "!")]),
//     }
//     .build();

//     let Graph {
//         topology,
//         nodes: _,
//         edges,
//     } = graph;

//     let searcher = Searcher::from(&topology);
//     {
//         let inputs = searcher.global_inputs();
//         assert_eq!(inputs.len(), 1);
//         assert_eq!(inputs[0].index(), 0);
//         assert_eq!(edges[inputs[0].index()], "|0");

//         let outputs = searcher.global_outputs();
//         assert_eq!(outputs.len(), 1);
//         assert_eq!(outputs[0].index(), 6);
//         assert_eq!(edges[outputs[0].index()], "!");

//         let local_edges = searcher
//             .local_edges()
//             .iter()
//             .map(|x| x.index())
//             .collect::<HashSet<_>>();
//         assert_eq!(local_edges.len(), 2);
//         assert_eq!(local_edges, HashSet::from([1, 4]));
//         assert_eq!(
//             local_edges
//                 .iter()
//                 .map(|i| edges[*i])
//                 .collect::<HashSet<_>>(),
//             HashSet::from(["|1", "|4"])
//         );
//     }
//     {
//         let nodes = searcher.nodes();
//         assert_eq!(nodes.len(), 3);

//         let a = nodes.get(0);
//         assert_eq!(a.inputs().len(), 2);
//         assert_eq!(a.inputs()[0].index(), 0);
//         assert_eq!(a.inputs()[1].index(), 1);
//         assert_eq!(a.outputs().len(), 2);
//         assert_eq!(a.outputs()[0].index(), 2);
//         assert_eq!(a.outputs()[1].index(), 3);
//         assert_eq!(
//             a.predecessors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([])
//         );
//         assert_eq!(
//             a.successors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([1, 2])
//         );

//         let b = nodes.get(1);
//         assert_eq!(b.inputs().len(), 2);
//         assert_eq!(b.inputs()[0].index(), 3);
//         assert_eq!(b.inputs()[1].index(), 4);
//         assert_eq!(b.outputs().len(), 1);
//         assert_eq!(b.outputs()[0].index(), 5);
//         assert_eq!(
//             b.predecessors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([0])
//         );
//         assert_eq!(
//             b.successors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([2])
//         );

//         let c = nodes.get(2);
//         assert_eq!(c.inputs().len(), 2);
//         assert_eq!(c.inputs()[0].index(), 5);
//         assert_eq!(c.inputs()[1].index(), 2);
//         assert_eq!(c.outputs().len(), 1);
//         assert_eq!(c.outputs()[0].index(), 6);
//         assert_eq!(
//             c.predecessors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([0, 1])
//         );
//         assert_eq!(
//             c.successors()
//                 .iter()
//                 .map(Node::index)
//                 .collect::<HashSet<_>>(),
//             HashSet::from([usize::MAX])
//         );
//     }
//     {
//         let edges = searcher.edges();
//         assert_eq!(edges.len(), 7);
//     }
// }
