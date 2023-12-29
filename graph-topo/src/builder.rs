use super::{
    container::{Graph, GraphTopo, Node},
    ucount,
};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::Add,
};

/// 图建造者。
#[derive(Debug)]
pub struct Builder<NodeKey, Node, EdgeKey, Edge> {
    /// 拓扑结构记录：节点-入边-出边。
    pub topology: HashMap<NodeKey, (Vec<EdgeKey>, Vec<EdgeKey>)>,
    /// 全图输入边。
    pub global_inputs: Vec<EdgeKey>,
    /// 全图输出边。
    pub global_outputs: Vec<EdgeKey>,
    /// 已知的节点信息。
    ///
    /// 理论上说节点信息是不可推断的。
    pub nodes: HashMap<NodeKey, Node>,
    /// 已知的边信息。
    ///
    /// 大部分边信息可能是推断出来的。
    pub edges: HashMap<EdgeKey, Edge>,
}

impl<KN, N, KE, E> Default for Builder<KN, N, KE, E> {
    /// 创建一个空的建造者。
    #[inline]
    fn default() -> Self {
        Self {
            topology: Default::default(),
            global_inputs: Default::default(),
            global_outputs: Default::default(),
            nodes: Default::default(),
            edges: Default::default(),
        }
    }
}

impl<KN, N, KE, E> Builder<KN, N, KE, E>
where
    KN: Eq + Hash + Clone,
    KE: Eq + Hash,
    E: Default,
{
    /// 消耗生成器构造图拓扑。
    pub fn build(mut self) -> Graph<N, E> {
        // 边和序号的映射关系。
        let mut key_to_idx = HashMap::new();
        // 非局部边表。
        let mut not_local = HashSet::new();
        // 映射全图输入边。
        let mut edges = self
            .global_inputs
            .iter()
            .enumerate()
            .map(|(i, edge)| {
                key_to_idx.insert(edge, i as ucount);
                not_local.insert(edge);
                self.edges.remove(edge).unwrap_or_default()
            })
            .collect::<Vec<_>>();
        // 直接遍历拓扑以构建非局部边表
        let mut connections = Vec::with_capacity(
            self.topology
                .values()
                .map(|(inputs, outputs)| {
                    not_local.extend(outputs);
                    inputs.len()
                })
                .sum::<usize>()
                .add(self.global_outputs.len()),
        );
        // 预留全图输出边的空间
        connections.resize(self.global_outputs.len(), ucount::MAX);
        // not_local 不再改变了
        let not_local = not_local;

        let mut topo_nodes = Vec::with_capacity(self.topology.len());
        let mut nodes = Vec::with_capacity(self.topology.len());
        let mut mapped = HashSet::<KN>::new();
        while mapped.len() < self.topology.len() {
            for (kn, (inputs, outputs)) in &self.topology {
                // 过滤映射过的节点
                if mapped.contains(kn) {
                    continue;
                }
                // 发现新局部边
                let new_local = inputs
                    .iter()
                    .filter(|k| !key_to_idx.contains_key(k))
                    .collect::<HashSet<_>>();
                // 局部边里有非局部边 === 有未知边
                if new_local.iter().any(|ke| not_local.contains(ke)) {
                    continue;
                }
                // 映射节点
                mapped.insert(kn.clone());
                nodes.push(self.nodes.remove(kn).unwrap());
                topo_nodes.push(Node {
                    local_edges_len: new_local.len() as _,
                    inputs_len: inputs.len() as _,
                    outputs_len: outputs.len() as _,
                });
                // 映射边
                for edge in new_local {
                    key_to_idx.insert(edge, edges.len() as ucount);
                    edges.push(self.edges.remove(edge).unwrap_or_default());
                }
                for edge in outputs {
                    key_to_idx.insert(edge, edges.len() as ucount);
                    edges.push(self.edges.remove(edge).unwrap_or_default());
                }
                // 映射连接
                connections.extend(inputs.iter().map(|ke| key_to_idx[ke]));
            }
        }
        // 映射全图输出边
        for (i, edge) in self.global_outputs.iter().enumerate() {
            connections[i] = key_to_idx[edge];
        }

        Graph {
            topology: GraphTopo {
                global_inputs_len: self.global_inputs.len() as _,
                global_outputs_len: self.global_outputs.len() as _,
                nodes: topo_nodes,
                connections,
            },
            nodes,
            edges,
        }
    }
}
