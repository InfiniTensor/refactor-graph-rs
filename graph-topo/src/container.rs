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
