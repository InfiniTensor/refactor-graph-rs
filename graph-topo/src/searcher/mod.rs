mod internal;

use crate::GraphTopo;
use internal::{Internal, NodeIdx};
use std::{
    cell::RefCell,
    collections::HashSet,
    hash::Hash,
    rc::{Rc, Weak},
};

/// 图拓扑索引器。
pub struct Searcher(Rc<RefCell<Internal>>);

impl Searcher {
    /// 获取节点集合。
    #[inline]
    pub fn nodes(&self) -> Nodes {
        Nodes(Rc::downgrade(&self.0))
    }

    /// 获取边集合。
    #[inline]
    pub fn edges(&self) -> Edges {
        Edges(Rc::downgrade(&self.0))
    }

    /// 获取全图输入边。
    pub fn global_inputs(&self) -> Vec<Edge> {
        let weak = Rc::downgrade(&self.0);
        let internal = self.0.borrow();
        internal
            .global_inputs
            .iter()
            .map(|i| Edge(weak.clone(), *i))
            .collect()
    }

    /// 获取全图输出边。
    pub fn global_outputs(&self) -> Vec<Edge> {
        let weak = Rc::downgrade(&self.0);
        let internal = self.0.borrow();
        internal
            .global_outputs
            .iter()
            .map(|i| Edge(weak.clone(), *i))
            .collect()
    }

    /// 获取局部边。
    pub fn local_edges(&self) -> HashSet<Edge> {
        let weak = Rc::downgrade(&self.0);
        let internal = self.0.borrow();
        internal
            .local_edges
            .iter()
            .map(|i| Edge(weak.clone(), *i))
            .collect()
    }

    /// 检查一个节点是否属于这个图。
    #[inline]
    pub fn contains_node(&self, node: &Node) -> bool {
        node.0.ptr_eq(&Rc::downgrade(&self.0))
    }

    /// 检查一个边是否属于这个图。
    #[inline]
    pub fn contains_edge(&self, edge: &Edge) -> bool {
        edge.0.ptr_eq(&Rc::downgrade(&self.0))
    }
}

impl Clone for Searcher {
    /// `Rc` 仅用于在相关索引类型之间共享，不在两个 `Searcher` 之间共享。
    #[inline]
    fn clone(&self) -> Self {
        Self(Rc::new(RefCell::new(self.0.borrow().clone())))
    }
}

impl From<&GraphTopo> for Searcher {
    #[inline]
    fn from(value: &GraphTopo) -> Self {
        Self(Rc::new(RefCell::new(Internal::new(value))))
    }
}

/// 节点集合。
pub struct Nodes(Weak<RefCell<Internal>>);
/// 边集合。
pub struct Edges(Weak<RefCell<Internal>>);
/// 节点索引器。
#[derive(Clone)]
pub struct Node(Weak<RefCell<Internal>>, usize);
/// 边索引器。
#[derive(Clone)]
pub struct Edge(Weak<RefCell<Internal>>, usize);
#[derive(Clone)]
pub struct NodeIter(Weak<RefCell<Internal>>, usize);
#[derive(Clone)]
pub struct EdgeIter(Weak<RefCell<Internal>>, usize);

impl Nodes {
    #[inline]
    pub fn get(&self, idx: usize) -> Node {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        assert!(idx < internal.nodes.len());
        Node(self.0.clone(), idx)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes.len()
    }
}

impl IntoIterator for Nodes {
    type Item = Node;
    type IntoIter = NodeIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        NodeIter(self.0, 0)
    }
}

impl Edges {
    #[inline]
    pub fn get(&self, idx: usize) -> Edge {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        assert!(idx < internal.edges.len());
        Edge(self.0.clone(), idx)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.edges.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.edges.len()
    }
}

impl IntoIterator for Edges {
    type Item = Edge;
    type IntoIter = EdgeIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        EdgeIter(self.0, 0)
    }
}

impl Iterator for NodeIter {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.upgrade().and_then(|internal| {
            let internal = internal.borrow();
            if self.1 < internal.nodes.len() {
                let idx = self.1;
                self.1 += 1;
                Some(Node(self.0.clone(), idx))
            } else {
                None
            }
        })
    }
}

impl Iterator for EdgeIter {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.upgrade().and_then(|internal| {
            let internal = internal.borrow();
            if self.1 < internal.edges.len() {
                let idx = self.1;
                self.1 += 1;
                Some(Edge(self.0.clone(), idx))
            } else {
                None
            }
        })
    }
}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr() && self.1 == other.1
    }
}

impl Eq for Node {}

impl Hash for Node {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
        self.1.hash(state);
    }
}

impl Node {
    /// 获取节点序号。
    #[inline]
    pub const fn index(&self) -> usize {
        self.1
    }

    /// 获取节点入边。
    pub fn inputs(&self) -> Vec<Edge> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes[self.1]
            .inputs
            .iter()
            .map(|i| Edge(self.0.clone(), *i))
            .collect()
    }

    /// 获取节点出边。
    pub fn outputs(&self) -> Vec<Edge> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes[self.1]
            .outputs
            .iter()
            .map(|i| Edge(self.0.clone(), *i))
            .collect()
    }

    /// 获取节点前驱。
    pub fn predecessors(&self) -> HashSet<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes[self.1]
            .predecessors
            .iter()
            .map(|i| Node(self.0.clone(), *i))
            .collect()
    }

    /// 获取节点后继。
    pub fn successors(&self) -> HashSet<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes[self.1]
            .successors
            .iter()
            .map(|i| Node(self.0.clone(), *i))
            .collect()
    }
}

impl PartialEq for Edge {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr() && self.1 == other.1
    }
}

impl Eq for Edge {}

impl Hash for Edge {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
        self.1.hash(state);
    }
}

impl Edge {
    /// 获取边序号。
    #[inline]
    pub const fn index(&self) -> usize {
        self.1
    }

    /// 获取边源节点。
    pub fn source(&self) -> Option<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        let idx = internal.edges[self.1].source;
        if idx == NodeIdx::MAX {
            None
        } else {
            Some(Node(self.0.clone(), idx as _))
        }
    }

    /// 获取边目标节点。
    pub fn targets(&self) -> Vec<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.edges[self.1]
            .targets
            .iter()
            .map(|i| Node(self.0.clone(), *i))
            .collect()
    }
}
