// use crate::{Graph, Node};
// use graph_topo::{GraphTopo, GraphTopoSearcher, Node as NodeRef};
// use std::{
//     collections::{HashMap, HashSet},
//     mem::take,
// };

// impl Graph {
//     /// 从原图中萃取一个节点的子图。
//     ///
//     /// 萃取一个节点的子图不需要改变当前图的拓扑结构。
//     ///
//     /// 返回当前图上子图索引的列表。
//     pub fn extract_one<I>(&mut self, subgraphs: I) -> HashSet<NodeRef>
//     where
//         I: IntoIterator<Item = NodeRef>,
//     {
//         let ans = subgraphs.into_iter().collect();
//         for node_ref in &ans {
//             assert!(self.topology.contains_node(node_ref));

//             let mut new_topo = GraphTopo::new();
//             let mut new_nodes = Vec::new();
//             let mut new_edges = Vec::new();

//             let mut inputs = Vec::new();
//             for edge in node_ref.inputs() {
//                 inputs.push(new_topo.add_edge());
//                 new_edges.push(self.edges[edge.index()].clone())
//             }
//             let node = new_topo.add_node(inputs, node_ref.outputs().len());
//             new_nodes.push(take(&mut self.nodes[node_ref.index()]));
//             for edge in node_ref.outputs() {
//                 new_topo.mark_output(node.get_output(edge.index()));
//                 new_edges.push(self.edges[edge.index()].clone())
//             }

//             self.nodes[node_ref.index()] = Node::from(Graph {
//                 topology: GraphTopoSearcher::from(new_topo),
//                 nodes: new_nodes,
//                 edges: new_edges,
//             });
//         }
//         ans
//     }

//     pub fn reduce(&mut self) {
//         let Self {
//             topology,
//             mut nodes,
//             mut edges,
//         } = std::mem::replace(
//             self,
//             Self {
//                 topology: GraphTopo::new().into(),
//                 nodes: Vec::new(),
//                 edges: Vec::new(),
//             },
//         );

//         let mut new_topo = GraphTopo::new();
//         let mut old_to_new = HashMap::new();

//         for edge in topology.global_inputs() {
//             self.edges.push(take(&mut edges[edge.index()]));
//             old_to_new.insert(edge, new_topo.add_edge());
//         }
//         for node in topology.nodes() {
//             match take(&mut nodes[node.index()]) {
//                 Node::Subgraph(mut g) => {
//                     let mut inner_to_new = HashMap::new();
//                     // 对齐子图输入
//                     node.inputs()
//                         .into_iter()
//                         .zip(g.topology.global_inputs())
//                         .for_each(|(outter, inner)| {
//                             inner_to_new.insert(inner, old_to_new[&outter]);
//                         });
//                     for node in g.topology.nodes() {
//                         self.nodes.push(take(&mut g.nodes[node.index()]));
//                         let new_node = new_topo.add_node(
//                             node.inputs().iter().map(|e| inner_to_new[e]),
//                             node.outputs().len(),
//                         );
//                         for (i, edge) in node.outputs().into_iter().enumerate() {
//                             self.edges.push(take(&mut g.edges[edge.index()]));
//                             old_to_new.insert(edge, new_node.get_output(i));
//                         }
//                     }
//                     // 对齐子图输出
//                     node.outputs()
//                         .into_iter()
//                         .zip(g.topology.global_outputs())
//                         .for_each(|(outter, inner)| {
//                             old_to_new.insert(outter, inner_to_new[&inner]);
//                         });
//                 }
//                 op => {
//                     self.nodes.push(op);
//                     let new_node = new_topo.add_node(
//                         node.inputs().iter().map(|e| old_to_new[e]),
//                         node.outputs().len(),
//                     );
//                     for (i, edge) in node.outputs().into_iter().enumerate() {
//                         self.edges.push(take(&mut edges[edge.index()]));
//                         old_to_new.insert(edge, new_node.get_output(i));
//                     }
//                 }
//             }
//         }

//         self.topology = new_topo.into();
//     }
// }
