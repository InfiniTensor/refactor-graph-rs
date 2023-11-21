use std::ops::{Add, Range};

/// 图拓扑结构。
#[derive(Clone, Default, Debug)]
pub struct GraphTopo {
    /// 全图输入边的数量。
    pub(super) global_inputs_len: usize,
    /// 全图输出边的数量。
    pub(super) global_outputs_len: usize,
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

/// 用于保存构建结果的数据结构，对节点和边重新排序。
#[derive(Clone, Debug)]
pub struct Graph<Node, Edge> {
    /// 节点和边的拓扑结构。
    pub topology: GraphTopo,
    /// 所有节点的信息。
    pub nodes: Vec<Node>,
    /// 所有边的信息。
    pub edges: Vec<Edge>,
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
pub struct OutputEdge(pub usize);

/// 用于遍历图拓扑的结构。
pub struct Iter<'a> {
    inner: &'a GraphTopo,
    i: usize,
    pass_connections: usize,
    pass_edges: usize,
}

impl GraphTopo {
    /// 遍历迭代器。
    #[inline]
    pub const fn traverse(&self) -> Iter {
        Iter {
            inner: self,
            i: 0,
            pass_connections: 0,
            pass_edges: self.global_inputs_len,
        }
    }

    /// 遍历迭代器。
    #[inline]
    pub const fn iter(&self) -> Iter {
        self.traverse()
    }

    /// 节点数量。
    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    /// 计算图中边的总数。
    pub fn calculate_edge_len(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.local_edges_len + node.outputs_len)
            .sum::<usize>()
            .add(self.global_outputs_len)
    }

    /// 全图输入边的数量。
    pub const fn global_inputs_len(&self) -> usize {
        self.global_inputs_len
    }

    /// 全图输出边的数量。
    pub const fn global_outputs_len(&self) -> usize {
        self.global_outputs_len
    }

    /// 全图输入边集。
    pub const fn global_inputs(&self) -> Range<usize> {
        0..self.global_inputs_len
    }

    /// 全图输出边集。
    pub fn global_outputs(&self) -> &[OutputEdge] {
        &self.connections[..self.global_outputs_len]
    }
}

impl<'a> IntoIterator for &'a GraphTopo {
    type Item = (usize, &'a [OutputEdge], Range<usize>);

    type IntoIter = Iter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.traverse()
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (usize, &'a [OutputEdge], Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        let node = self.inner.nodes.get(i)?;
        self.i += 1;

        let first_input = self.pass_connections;
        self.pass_connections += node.inputs_len;
        self.pass_edges += node.local_edges_len;
        let first_edge = self.pass_edges;
        self.pass_edges += node.outputs_len;

        Some((
            i,
            &self.inner.connections[first_input..self.pass_connections],
            first_edge..self.pass_edges,
        ))
    }
}
