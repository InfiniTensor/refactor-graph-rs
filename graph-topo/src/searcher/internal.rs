use super::GraphTopo;
use std::collections::HashSet;

pub(super) type NodeIdx = usize;
pub(super) type EdgeIdx = usize;
pub(super) const EXTERNAL: NodeIdx = NodeIdx::MAX;

#[derive(Clone, Debug)]
pub(super) struct Internal {
    pub(super) global_inputs: Vec<EdgeIdx>,
    pub(super) global_outputs: Vec<EdgeIdx>,
    pub(super) local_edges: HashSet<EdgeIdx>,
    pub(super) nodes: Vec<SeacherNode>,
    pub(super) edges: Vec<SeacherEdge>,
}

#[derive(Clone, Default, Debug)]
pub(super) struct SeacherNode {
    pub(super) inputs: Vec<EdgeIdx>,
    pub(super) outputs: Vec<EdgeIdx>,
    pub(super) predecessors: HashSet<NodeIdx>,
    pub(super) successors: HashSet<NodeIdx>,
}

#[derive(Clone, Debug)]
pub(super) struct SeacherEdge {
    pub(super) source: NodeIdx,
    pub(super) targets: HashSet<NodeIdx>,
}

impl Default for SeacherEdge {
    #[inline]
    fn default() -> Self {
        Self {
            source: EXTERNAL,
            targets: HashSet::new(),
        }
    }
}

impl Internal {
    pub(super) fn new(graph: &GraphTopo) -> Self {
        let nodes_len = graph.nodes.len();
        let global_inputs_len = graph.global_inputs_len;

        let global_inputs = (0..global_inputs_len).collect::<Vec<_>>();
        let mut global_outputs = Vec::new();
        let mut local_edges = HashSet::new();
        let mut nodes = vec![SeacherNode::default(); nodes_len];
        let mut edges = Vec::new();

        let mut pass_connections = 0;

        for _ in 0..global_inputs_len {
            edges.push(Default::default());
        }
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for _ in 0..node.local_edges_len {
                local_edges.insert(edges.len());
                edges.push(Default::default());
            }
            for _ in 0..node.outputs_len {
                nodes[node_idx].outputs.push(edges.len());
                edges.push(SeacherEdge {
                    source: node_idx,
                    targets: HashSet::new(),
                })
            }
            for _ in 0..node.inputs_len {
                let edge_idx = graph.connections[pass_connections].0;
                let edge = &mut edges[edge_idx];

                nodes[node_idx].inputs.push(edge_idx);
                edge.targets.insert(node_idx);

                if edge.source != EXTERNAL {
                    nodes[node_idx].predecessors.insert(edge.source);
                    nodes[edge.source].successors.insert(node_idx);
                }

                pass_connections += 1;
            }
        }
        for ouput in &graph.connections[pass_connections..] {
            let edge_idx = ouput.0;
            let edge = &mut edges[edge_idx];

            global_outputs.push(edge_idx);
            edge.targets.insert(EXTERNAL);

            if edge.source != EXTERNAL {
                nodes[edge.source].successors.insert(EXTERNAL);
            }
        }

        Self {
            global_inputs,
            global_outputs,
            local_edges,
            nodes,
            edges,
        }
    }
}
