use crate::{Graph, Node};
use graph_topo::{GraphTopo, GraphTopoSearcher, Node as NodeRef};
use std::collections::HashSet;

impl Graph {
    /// 从原图中萃取一个节点的子图。
    ///
    /// 萃取一个节点的子图不需要改变当前图的拓扑结构。
    ///
    /// 返回当前图上子图索引的列表。
    pub fn extract_one<I>(&mut self, subgraphs: I) -> HashSet<NodeRef>
    where
        I: IntoIterator<Item = NodeRef>,
    {
        let ans = subgraphs.into_iter().collect();
        for node_ref in &ans {
            assert!(self.topology.contains_node(node_ref));

            let mut new_topo = GraphTopo::new();
            let mut new_nodes = Vec::new();
            let mut new_edges = Vec::new();

            let mut inputs = Vec::new();
            for edge in node_ref.inputs() {
                inputs.push(new_topo.add_edge());
                new_edges.push(self.edges[edge.index()].clone())
            }
            let node = new_topo.add_node(inputs, node_ref.outputs().len());
            new_nodes.push(std::mem::take(&mut self.nodes[node_ref.index()]));
            for edge in node_ref.outputs() {
                new_topo.mark_output(node.get_output(edge.index()));
                new_edges.push(self.edges[edge.index()].clone())
            }

            self.nodes[node_ref.index()] = Node::from(Graph {
                topology: GraphTopoSearcher::from(new_topo),
                nodes: new_nodes,
                edges: new_edges,
            });
        }
        ans
    }
}
