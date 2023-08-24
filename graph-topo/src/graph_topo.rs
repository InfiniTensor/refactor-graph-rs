#[derive(Clone)]
pub struct GraphTopo {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    targets: Vec<Target>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub(super) struct Index(i64);

impl Default for Index {
    fn default() -> Self {
        Index(-1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub(super) struct NodeIdx(Index);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub(super) struct EdgeIdx(Index);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub(super) struct TargetIdx(Index);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub(super) struct OutputIdx(Index);

#[derive(Clone, Default)]
pub(super) struct Node {
    first_edge: EdgeIdx,
}
#[derive(Clone, Default)]
pub(super) struct Edge {
    next: EdgeIdx,
    first_target: TargetIdx,
    output: OutputIdx,
}
#[derive(Clone, Default)]
pub(super) struct Target {
    next: TargetIdx,
    to: NodeIdx,
}

#[derive(Clone, Copy)]
pub struct NodeRef(NodeIdx);
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct EdgeRef(EdgeIdx);

impl GraphTopo {
    #[inline]
    pub fn addEdge(&mut self) -> EdgeRef {
        self.edges.push(Default::default());
        EdgeRef(EdgeIdx(Index((self.edges.len() - 1) as _)))
    }

    pub fn addNode<I>(&mut self, inputs: I) -> NodeRef
    where
        I: IntoIterator<Item = EdgeRef>,
    {
        let node_idx = NodeIdx(Index(self.nodes.len() as _));
        // 添加节点
        self.nodes.push(Default::default());
        // 将节点添加到入边的目标
        for edge in inputs {
            self.targets.push(Target {
                next: std::mem::replace(
                    &mut self.edges[edge.0 .0 .0 as usize].first_target,
                    TargetIdx(Index(self.targets.len() as _)),
                ),
                to: node_idx,
            });
        }
        NodeRef(node_idx)
    }

    pub fn mark_output(&mut self, edge: EdgeRef) {
        let idx = edge.0 .0 .0 as usize;
        assert!((0..self.edges.len()).contains(&idx));
        assert!(self.edges[idx].output == Default::default());
        self.edges[idx].output.0 .0 = self.last_output().0 .0 + 1;
    }

    pub fn mark_outputs<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = EdgeRef>,
    {
        let mut out_idx = self.last_output().0 .0 + 1;
        for edge in edges {
            let idx = edge.0 .0 .0 as usize;
            assert!((0..self.edges.len()).contains(&idx));
            assert!(self.edges[idx].output == Default::default());
            self.edges[idx].output.0 .0 = out_idx;
            out_idx += 1;
        }
    }

    #[inline]
    fn last_output(&self) -> OutputIdx {
        self.edges
            .iter()
            .map(|x| x.output)
            .max()
            .unwrap_or_default()
    }
}
