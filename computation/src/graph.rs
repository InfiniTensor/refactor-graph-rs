use crate::{operators::InferError, Edge, Node};

/// 计算图。
pub struct Graph(pub graph_topo::Graph<Node, Edge>);

impl Graph {
    /// 填写所有边。
    pub fn infer_edges(&mut self, check: bool) -> Result<(), InferError> {
        for (node_idx, inputs, outputs) in self.0.topology.traverse() {
            let inputs = inputs
                .into_iter()
                .map(|i| self.0.edges[i].clone())
                .collect::<Vec<_>>();
            let infered = self.0.nodes[node_idx].infer(&inputs)?;
            if infered.len() < outputs.len() {
                panic!("Operator {node_idx:?} returns too few outputs");
            }
            for (edge_idx, edge) in outputs.into_iter().zip(infered) {
                let t = &mut self.0.edges[edge_idx];
                if check && !t.is_unknown() {
                    assert!(t.info_equal(&edge));
                } else {
                    *t = edge;
                }
            }
        }
        Ok(())
    }
}
