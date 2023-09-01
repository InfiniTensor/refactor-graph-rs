use std::fmt;

/// 图拓扑结构。
#[derive(Clone, Default, Debug)]
pub struct GraphTopo {
    /// 全图输入边的数量。
    pub(super) global_inputs_len: usize,
    /// 节点集合。
    ///
    /// 节点集合是图中节点的一种符合拓扑序的排列。
    pub(super) nodes: Vec<Node>,
    /// 边的目标集合。
    ///
    /// 该集合的排列顺序是由节点集合决定的。
    /// 首先是每个节点的输入边，最后是全图的输出边。
    ///
    /// 即：`connections.len() == nodes.sum(inputs_len) + global_outputs_lenws`。
    pub(super) connections: Vec<OutputEdge>,
}

/// 节点结构。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Node {
    pub(super) local_edges_len: usize,
    pub(super) inputs_len: usize,
    pub(super) outputs_len: usize,
}

/// 作为节点输入的边序号。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct OutputEdge(pub(super) usize);

impl GraphTopo {
    /// 遍历，返回全图的输入输出。
    pub fn traverse<F, E>(&self, mut f: F) -> Result<(Vec<usize>, Vec<usize>), E>
    where
        F: FnMut(usize, Vec<usize>, Vec<usize>) -> Result<(), E>,
    {
        let mut pass_connections = 0;
        let mut pass_edges = self.global_inputs_len;
        for (i, node) in self.nodes.iter().enumerate() {
            let first_input = pass_connections;
            pass_connections += node.inputs_len;

            pass_edges += node.local_edges_len;
            let first_edge = pass_edges;
            pass_edges += node.outputs_len;

            f(
                i,
                self.connections[first_input..pass_connections]
                    .iter()
                    .map(|e| e.0)
                    .collect(),
                (first_edge..pass_edges).collect(),
            )?;
        }
        Ok((
            (0..self.global_inputs_len).collect(),
            self.connections[pass_connections..]
                .iter()
                .map(|e| e.0)
                .collect(),
        ))
    }
}

impl fmt::Display for GraphTopo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (inputs, outputs) = self.traverse(|i, inputs, outputs| {
            for i in outputs {
                write!(f, "%{i} ")?;
            }
            write!(f, "<- _{i}")?;
            for i in inputs {
                write!(f, " %{i}")?;
            }
            writeln!(f)?;
            Ok(())
        })?;
        writeln!(f)?;
        for i in outputs {
            write!(f, "%{i} ")?;
        }
        write!(f, "<- _")?;
        for i in inputs {
            write!(f, " %{i}")?;
        }
        writeln!(f)
    }
}
