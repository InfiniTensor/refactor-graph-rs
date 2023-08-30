use crate::{
    container::{Node, OutputEdge},
    GraphTopo,
};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

/// 图拓扑生成器。
pub struct Builder<NodeKey, Node, EdgeKey, Edge> {
    /// 拓扑结构记录：节点-入边-出边。
    pub topology: HashMap<NodeKey, (Vec<EdgeKey>, Vec<EdgeKey>)>,
    /// 全图输入边。
    pub global_inputs: Vec<EdgeKey>,
    /// 全图输出边。
    pub global_outputs: Vec<EdgeKey>,
    /// 已知的节点信息。
    ///
    /// 理论上说，每个节点都有部分信息是不可推断的。
    pub nodes: HashMap<NodeKey, Node>,
    /// 已知的边信息。
    ///
    /// 大部分边信息可能是推断出来的。
    pub edges: HashMap<EdgeKey, Edge>,
}

/// 用于保存构建结果的数据结构，对节点和边重新排序。
pub struct Graph<Node, Edge> {
    pub topology: GraphTopo,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl<KN, N, KE, E> Builder<KN, N, KE, E>
where
    KN: Eq + Hash,
    KE: Eq + Hash,
    E: Default,
{
    /// 消耗生成器构造图拓扑。
    pub fn build(mut self) -> Graph<N, E> {
        let mut topo_nodes = Vec::new();
        let mut connections = Vec::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 边和序号的映射关系。
        let mut key_to_idx = HashMap::new();
        // 可知来源的边，包括全图输入和节点输出。
        let mut generated_edges = HashSet::new();
        // 映射全图输入边。
        for edge in &self.global_inputs {
            key_to_idx.insert(edge, edges.len());
            edges.push(self.edges.remove(edge).unwrap_or_default());
            generated_edges.insert(edge);
        }
        // 记录节点输出边。
        for (_, outputs) in self.topology.values() {
            generated_edges.extend(outputs);
        }
        // 迭代拓扑结构。
        let mut mapped_nodes = HashSet::new();
        while mapped_nodes.len() < self.topology.len() {
            for (kn, (inputs, outputs)) in &self.topology {
                // 节点未映射，边已映射或局部。
                if !mapped_nodes.contains(kn)
                    && inputs
                        .iter()
                        .all(|i| key_to_idx.contains_key(i) || !generated_edges.contains(i))
                {
                    mapped_nodes.insert(kn);
                    // 找到未映射的局部边。
                    let new_local = inputs
                        .iter()
                        .filter(|e| !key_to_idx.contains_key(e) && !generated_edges.contains(e))
                        .collect::<HashSet<_>>();
                    // 添加节点。
                    topo_nodes.push(Node {
                        local_edges_len: new_local.len(),
                        inputs_len: inputs.len(),
                        outputs_len: outputs.len(),
                    });
                    nodes.push(self.nodes.remove(kn).expect("node inference not supported"));
                    // 映射局部边。
                    for edge in new_local {
                        key_to_idx.insert(edge, edges.len());
                        edges.push(self.edges.remove(edge).unwrap_or_default());
                    }
                    // 记录节点入边。
                    for edge in inputs {
                        connections.push(OutputEdge(key_to_idx[edge]));
                    }
                    // 映射节点出边。
                    for edge in outputs {
                        key_to_idx.insert(edge, edges.len());
                        edges.push(self.edges.remove(edge).unwrap_or_default());
                    }
                }
            }
        }
        // 映射全图输出边。
        for edge in &self.global_outputs {
            connections.push(OutputEdge(key_to_idx[edge]));
        }

        Graph {
            topology: GraphTopo {
                global_inputs_len: self.global_inputs.len(),
                nodes: topo_nodes,
                connections,
            },
            nodes,
            edges,
        }
    }
}
