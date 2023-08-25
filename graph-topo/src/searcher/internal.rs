use crate::container::GraphTopo;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub(super) struct Internal {
    pub(super) graph: GraphTopo,
    pub(super) global_inputs: Vec<usize>,
    pub(super) global_outputs: Vec<usize>,
    pub(super) nodes: Vec<SeacherNode>,
    pub(super) edges: Vec<SeacherEdge>,
}

#[derive(Clone, Default, Debug)]
pub(super) struct SeacherNode {
    pub(super) predecessors: HashSet<usize>,
    pub(super) successors: HashSet<usize>,
}

#[derive(Clone, Default, Debug)]
pub(super) struct SeacherEdge {
    pub(super) source: isize,
    pub(super) targets: Vec<usize>,
}

impl Internal {
    pub(super) fn new(mut graph: GraphTopo) -> Self {
        let nodes_len = graph.nodes.len();
        let edges_len = graph.edges.len();
        let mut global_inputs: Vec<usize>;
        let mut global_outputs = vec![];
        let mut nodes = vec![SeacherNode::default(); nodes_len];
        let mut edges = vec![
            SeacherEdge {
                source: -1,
                targets: vec![]
            };
            edges_len
        ];
        let mut global_input_candidatas = (0..edges_len).collect::<HashSet<_>>();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for edge_idx in node.outputs.clone() {
                edges[edge_idx].source = node_idx as _; // 填写边的起点
                global_input_candidatas.remove(&edge_idx); // 节点的出边不可能是全图入边
            }
            for target_idx in node.inputs.clone() {
                let edge_idx = graph.targets[target_idx].edge;
                edges[edge_idx].targets.push(node_idx); // 填写边的终点
            }
        }
        global_inputs = global_input_candidatas.into_iter().collect::<Vec<_>>();
        global_inputs.sort();

        for (edge_idx, edge) in graph.edges.iter().enumerate() {
            if edge.output_idx >= 0 {
                let output_idx = edge.output_idx as usize;
                if output_idx > global_outputs.len() {
                    global_outputs.resize(output_idx + 1, 0);
                }
                global_outputs[output_idx] = edge_idx;
            }
            let source = edges[edge_idx].source;
            if source >= 0 {
                let source = source as usize;
                for target in &edges[edge_idx].targets {
                    nodes[source].successors.insert(*target);
                    nodes[*target].predecessors.insert(source);
                }
            }
        }

        graph.nodes.shrink_to_fit();
        graph.edges.shrink_to_fit();
        graph.targets.shrink_to_fit();
        global_inputs.shrink_to_fit();
        global_outputs.shrink_to_fit();
        nodes.shrink_to_fit();
        edges.shrink_to_fit();

        Self {
            graph,
            global_inputs,
            global_outputs,
            nodes,
            edges,
        }
    }
}
