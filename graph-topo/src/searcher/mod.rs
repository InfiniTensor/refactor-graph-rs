mod internal;

use crate::GraphTopo;
use internal::Internal;
use std::{
    cell::RefCell,
    collections::HashSet,
    hash::Hash,
    rc::{Rc, Weak},
};

pub struct Searcher(Rc<RefCell<Internal>>);

impl Default for Searcher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Searcher {
    #[inline]
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(Internal {
            graph: GraphTopo::default(),
            global_inputs: vec![],
            global_outputs: vec![],
            nodes: vec![],
            edges: vec![],
        })))
    }

    #[inline]
    pub fn nodes(&self) -> Nodes {
        Nodes(Rc::downgrade(&self.0))
    }

    #[inline]
    pub fn edges(&self) -> Edges {
        Edges(Rc::downgrade(&self.0))
    }

    pub fn global_inputs(&self) -> Vec<Edge> {
        let weak = Rc::downgrade(&self.0);
        let internal = self.0.borrow();
        internal
            .global_inputs
            .iter()
            .map(|i| Edge(weak.clone(), *i))
            .collect()
    }

    pub fn global_outputs(&self) -> Vec<Edge> {
        let weak = Rc::downgrade(&self.0);
        let internal = self.0.borrow();
        internal
            .global_outputs
            .iter()
            .map(|i| Edge(weak.clone(), *i))
            .collect()
    }

    #[inline]
    pub fn contains_node(&self, node: &Node) -> bool {
        node.0.ptr_eq(&Rc::downgrade(&self.0))
    }

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

impl From<GraphTopo> for Searcher {
    #[inline]
    fn from(value: GraphTopo) -> Self {
        Self(Rc::new(RefCell::new(Internal::new(value))))
    }
}

pub struct Nodes(Weak<RefCell<Internal>>);

pub struct NodeIter(Weak<RefCell<Internal>>, usize);

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

pub struct Node(Weak<RefCell<Internal>>, usize);

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
    #[inline]
    pub const fn index(&self) -> usize {
        self.1
    }

    pub fn inputs(&self) -> Vec<Edge> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let graph = &internal.borrow().graph;
        graph.nodes[self.1]
            .inputs
            .clone()
            .map(|i| Edge(self.0.clone(), graph.targets[i].edge))
            .collect()
    }

    pub fn outputs(&self) -> Vec<Edge> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let graph = &internal.borrow().graph;
        graph.nodes[self.1]
            .outputs
            .clone()
            .map(|i| Edge(self.0.clone(), i))
            .collect()
    }

    pub fn predecessors(&self) -> HashSet<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        internal.nodes[self.1]
            .predecessors
            .iter()
            .map(|i| Node(self.0.clone(), *i))
            .collect()
    }

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

pub struct Edges(Weak<RefCell<Internal>>);

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

pub struct EdgeIter(Weak<RefCell<Internal>>, usize);

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

pub struct Edge(Weak<RefCell<Internal>>, usize);

impl Edge {
    #[inline]
    pub const fn index(&self) -> usize {
        self.1
    }

    pub fn source(&self) -> Option<Node> {
        let internal = self.0.upgrade().expect("Graph has been dropped");
        let internal = internal.borrow();
        let idx = internal.edges[self.1].source;
        if idx < 0 {
            None
        } else {
            Some(Node(self.0.clone(), idx as _))
        }
    }

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

#[test]
fn test() {
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

    let searcher = Searcher::from(graph);
    {
        let global_inputs = searcher.global_inputs();
        assert_eq!(global_inputs.len(), 3);
        assert_eq!(global_inputs[0].1, 0);
        assert_eq!(global_inputs[1].1, 1);
        assert_eq!(global_inputs[2].1, 4);

        let global_outputs = searcher.global_outputs();
        assert_eq!(global_outputs.len(), 2);
        assert_eq!(global_outputs[0].1, 6);
        assert_eq!(global_outputs[1].1, 4);
    }
    let nodes = searcher.nodes();
    {
        assert_eq!(nodes.len(), 3);
        let _0 = nodes.get(0);
        let _1 = nodes.get(1);
        let _2 = nodes.get(2);

        assert_eq!(_0.inputs().len(), 2);
        assert_eq!(_0.inputs()[0].1, 0);
        assert_eq!(_0.inputs()[1].1, 1);
        assert_eq!(_0.outputs().len(), 2);
        assert_eq!(_0.outputs()[0].1, 2);
        assert_eq!(_0.outputs()[1].1, 3);
        assert_eq!(
            _0.predecessors()
                .iter()
                .map(|x| x.1)
                .collect::<HashSet<_>>(),
            HashSet::from([])
        );
        assert_eq!(
            _0.successors().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );

        assert_eq!(_1.inputs().len(), 2);
        assert_eq!(_1.inputs()[0].1, 3);
        assert_eq!(_1.inputs()[1].1, 4);
        assert_eq!(_1.outputs().len(), 1);
        assert_eq!(_1.outputs()[0].1, 5);
        assert_eq!(
            _1.predecessors()
                .iter()
                .map(|x| x.1)
                .collect::<HashSet<_>>(),
            HashSet::from([0])
        );
        assert_eq!(
            _1.successors().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([2])
        );

        assert_eq!(_2.inputs().len(), 2);
        assert_eq!(_2.inputs()[0].1, 5);
        assert_eq!(_2.inputs()[1].1, 2);
        assert_eq!(_2.outputs().len(), 1);
        assert_eq!(_2.outputs()[0].1, 6);
        assert_eq!(
            _2.predecessors()
                .iter()
                .map(|x| x.1)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1])
        );
        assert_eq!(
            _2.successors().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([])
        );
    }
    let edges = searcher.edges();
    {
        assert_eq!(edges.len(), 7);
        let _0 = edges.get(0);
        let _1 = edges.get(1);
        let _2 = edges.get(2);
        let _3 = edges.get(3);
        let _4 = edges.get(4);
        let _5 = edges.get(5);
        let _6 = edges.get(6);

        assert_eq!(_0.source().map(|x| x.1), None);
        assert_eq!(
            _0.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([0])
        );
        assert_eq!(_1.source().map(|x| x.1), None);
        assert_eq!(
            _1.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([0])
        );
        assert_eq!(_2.source().map(|x| x.1), Some(0));
        assert_eq!(
            _2.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([2])
        );
        assert_eq!(_3.source().map(|x| x.1), Some(0));
        assert_eq!(
            _3.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([1])
        );
        assert_eq!(_4.source().map(|x| x.1), None);
        assert_eq!(
            _4.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([1])
        );
        assert_eq!(_5.source().map(|x| x.1), Some(1));
        assert_eq!(
            _5.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([2])
        );
        assert_eq!(_6.source().map(|x| x.1), Some(2));
        assert_eq!(
            _6.targets().iter().map(|x| x.1).collect::<HashSet<_>>(),
            HashSet::from([])
        );
    }
}
