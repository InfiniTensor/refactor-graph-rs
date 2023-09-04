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

/// 用于遍历图拓扑的结构。
pub struct Iter<'a> {
    inner: &'a GraphTopo,
    i: usize,
    pass_connections: usize,
    pass_edges: usize,
}

impl GraphTopo {
    /// 获取遍历迭代器。
    #[inline]
    pub const fn traverse(&self) -> Iter {
        Iter {
            inner: self,
            i: 0,
            pass_connections: 0,
            pass_edges: self.global_inputs_len,
        }
    }

    /// 获取遍历迭代器。
    #[inline]
    pub fn iter(&self) -> Iter {
        self.traverse()
    }
}

impl<'a> IntoIterator for &'a GraphTopo {
    type Item = (usize, Vec<usize>, Vec<usize>);

    type IntoIter = Iter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.traverse()
    }
}

impl Iter<'_> {
    /// 获取全图输入边序号。
    #[inline]
    pub fn global_inputs(&self) -> Vec<usize> {
        (0..self.inner.global_inputs_len).collect()
    }

    /// 获取全图输出边序号。
    ///
    /// 只能在遍历完成后调用。
    #[inline]
    pub fn global_outputs(&self) -> Vec<usize> {
        assert_eq!(self.i, self.inner.nodes.len());
        (self.pass_edges..self.inner.connections.len())
            .map(|i| self.inner.connections[i].0)
            .collect()
    }

    /// 消费迭代器以获取全图输出边序号。
    pub fn take_global_outputs(mut self) -> Vec<usize> {
        while self.next().is_some() {}
        self.global_outputs()
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (usize, Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        let node = self.inner.nodes.get(i)?;
        self.i += 1;

        let first_input = self.pass_connections;
        self.pass_connections += node.inputs_len;
        let inputs = self.inner.connections[first_input..self.pass_connections]
            .iter()
            .map(|e| e.0)
            .collect();

        self.pass_edges += node.local_edges_len;
        let first_edge = self.pass_edges;
        self.pass_edges += node.outputs_len;
        let outputs = (first_edge..self.pass_edges).collect();

        Some((i, inputs, outputs))
    }
}

// impl fmt::Display for GraphTopo {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         for (i, inputs, outputs) in self.traverse() {
//             for i in outputs {
//                 write!(f, "%{i} ")?;
//             }
//             write!(f, "<- _{i}")?;
//             for i in inputs {
//                 write!(f, " %{i}")?;
//             }
//             writeln!(f)?;
//         }
//         writeln!(f)?;
//         for i in outputs {
//             write!(f, "%{i} ")?;
//         }
//         write!(f, "<- _")?;
//         for i in 0..self.global_inputs_len {
//             write!(f, " %{i}")?;
//         }
//         writeln!(f)
//     }
// }
