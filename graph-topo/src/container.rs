use std::{fmt, ops::Range};

/// 图拓朴结构。
#[derive(Clone, Default, Debug)]
pub struct GraphTopo {
    /// 边的目标集合。
    pub(super) targets: Vec<Target>,
    /// 节点集合。
    pub(super) nodes: Vec<Node>,
    /// 边集合。
    pub(super) edges: Vec<Edge>,
}

/// 表示边的目标。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Target {
    /// 目标属于哪个边。
    pub(super) edge: usize,
}

/// 表示节点。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Node {
    /// 节点的输入边（有序）。
    pub(super) inputs: Range<usize>,
    /// 节点的输出边（有序）。
    pub(super) outputs: Range<usize>,
}

/// 表示边。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Edge {
    /// 边如果是全图出边，表示第几个全图出边，-1 表示不是全图出边。
    pub(super) output_idx: isize,
}

/// 一个已经添加到图中的节点。
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct NodeRef {
    /// 节点的第一个出边。
    first_output: usize,
    /// 节点的出边数量。
    outputs_len: usize,
}

impl NodeRef {
    /// 取出节点的第 `idx` 个出边。
    #[inline]
    pub fn get_output(&self, idx: usize) -> EdgeRef {
        assert!(idx < self.outputs_len);
        EdgeRef(self.first_output + idx)
    }
}

/// 一个已经添加到图中的边。
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct EdgeRef(usize);

impl GraphTopo {
    /// 向图中添加一条全图入边，返回指向该边的引用。
    #[inline]
    pub fn add_edge(&mut self) -> EdgeRef {
        self.edges.push(Edge { output_idx: -1 });
        EdgeRef(self.edges.len() - 1)
    }

    /// 向图中添加一个节点，节点的输入是 `inputs`，并将输出 `outputs_len` 条边，返回指向该节点的引用。
    pub fn add_node<I>(&mut self, inputs: I, outputs_len: usize) -> NodeRef
    where
        I: IntoIterator<Item = EdgeRef>,
    {
        // 为入边添加目标并统计入度
        let mut inputs_len = 0;
        for input in inputs {
            self.targets.push(Target { edge: input.0 });
            inputs_len += 1;
        }
        // 将出边添加到边集
        let first_output = self.edges.len();
        self.edges
            .resize(first_output + outputs_len, Edge { output_idx: -1 });
        // 将节点添加到点集
        self.nodes.push(Node {
            inputs: self.targets.len() - inputs_len..self.targets.len(),
            outputs: first_output..first_output + outputs_len,
        });
        NodeRef {
            first_output,
            outputs_len,
        }
    }

    /// 将一个边标记为按顺序的全图出边。
    #[inline]
    pub fn mark_output(&mut self, edge: EdgeRef) {
        assert_eq!(self.edges[edge.0].output_idx, -1);
        let max = self.edges.iter().map(|e| e.output_idx).max().unwrap_or(-1);
        self.edges[edge.0].output_idx = max + 1;
    }

    /// 将一组边标记为按顺序的全图出边。
    pub fn mark_outputs<I>(&mut self, edge: I)
    where
        I: IntoIterator<Item = EdgeRef>,
    {
        let mut max = self.edges.iter().map(|e| e.output_idx).max().unwrap_or(-1);
        for edge in edge {
            assert_eq!(self.edges[edge.0].output_idx, -1);
            max += 1;
            self.edges[edge.0].output_idx = max;
        }
    }
}

impl fmt::Display for GraphTopo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (node_idx, node) in self.nodes.iter().enumerate() {
            let mut it = self
                .targets
                .iter()
                .enumerate()
                .skip(node.inputs.start)
                .take(node.inputs.len());
            if let Some((target_idx, target)) = it.next() {
                write!(f, "|{}:t{} ", target.edge, target_idx)?;
                for (target_idx, target) in it {
                    write!(f, "+ |{}:t{} ", target.edge, target_idx)?;
                }
            }
            write!(f, "-> *{node_idx} -> ")?;
            let mut it = self
                .edges
                .iter()
                .enumerate()
                .skip(node.outputs.start)
                .take(node.outputs.len());
            if let Some((edge_idx, edge)) = it.next() {
                write!(f, "|{edge_idx}")?;
                if edge.output_idx >= 0 {
                    write!(f, ":<{} ", edge.output_idx)?;
                } else {
                    write!(f, " ")?;
                }
                for (edge_idx, edge) in it {
                    write!(f, "+ |{edge_idx}")?;
                    if edge.output_idx >= 0 {
                        write!(f, ":<{} ", edge.output_idx)?;
                    } else {
                        write!(f, " ")?;
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[test]
fn test_build() {
    let mut graph = GraphTopo::default();
    let e0 = graph.add_edge(); // |0
    let e1 = graph.add_edge(); // |1
    let n0 = graph.add_node([e0, e1], 2); // |0:t0 + |1:t1 -> *0 -> |2 + |3
    let e3 = n0.get_output(1); // |3
    let e4 = graph.add_edge(); // |4
    let n1 = graph.add_node([e3, e4], 1); // |3:t2 + |4:t3 -> *1 -> |5
    let e5 = n1.get_output(0); // |5
    let e2 = n0.get_output(0); // |2
    let n2 = graph.add_node([e5, e2], 1); // |5:t4 + |2:t5 -> *2 -> |6
    let e6 = n2.get_output(0); // |6
    graph.mark_outputs([e6, e4]); // |6 -> <0, |4 -> <1

    assert_eq!(graph.targets.len(), 6);
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.edges.len(), 7);

    assert_eq!(graph.targets[0], Target { edge: 0 });
    assert_eq!(graph.targets[1], Target { edge: 1 });
    assert_eq!(graph.targets[2], Target { edge: 3 });
    assert_eq!(graph.targets[3], Target { edge: 4 });
    assert_eq!(graph.targets[4], Target { edge: 5 });
    assert_eq!(graph.targets[5], Target { edge: 2 });

    assert_eq!(
        graph.nodes[0],
        Node {
            inputs: 0..2,
            outputs: 2..4
        }
    );
    assert_eq!(
        graph.nodes[1],
        Node {
            inputs: 2..4,
            outputs: 5..6
        }
    );
    assert_eq!(
        graph.nodes[2],
        Node {
            inputs: 4..6,
            outputs: 6..7
        }
    );

    assert_eq!(graph.edges[0], Edge { output_idx: -1 });
    assert_eq!(graph.edges[1], Edge { output_idx: -1 });
    assert_eq!(graph.edges[2], Edge { output_idx: -1 });
    assert_eq!(graph.edges[3], Edge { output_idx: -1 });
    assert_eq!(graph.edges[4], Edge { output_idx: 1 });
    assert_eq!(graph.edges[5], Edge { output_idx: -1 });
    assert_eq!(graph.edges[6], Edge { output_idx: 0 });
}
