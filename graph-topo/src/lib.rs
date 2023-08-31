#![deny(warnings)]

mod builder;
mod container;
mod searcher;

pub use builder::{Builder, Graph};
pub use container::GraphTopo;
pub use searcher::{Edge, Node, Searcher};

#[test]
fn test() {
    use std::collections::{HashMap, HashSet};

    let graph = Builder {
        topology: HashMap::from([
            ("A", (vec!["a", "b"], vec!["c", "d"])),
            ("B", (vec!["d", "e"], vec!["f"])),
            ("C", (vec!["f", "c"], vec!["z"])),
        ]),
        global_inputs: vec!["a"],
        global_outputs: vec!["z"],
        nodes: HashMap::from([("A", "*0"), ("B", "*1"), ("C", "*2")]),
        edges: HashMap::from([("a", "|0"), ("b", "|1"), ("e", "|4"), ("z", "!")]),
    }
    .build();

    let Graph {
        topology,
        nodes: _,
        edges,
    } = graph;

    let searcher = Searcher::from(topology);
    {
        let inputs = searcher.global_inputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].index(), 0);
        assert_eq!(edges[inputs[0].index()], "|0");

        let outputs = searcher.global_outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].index(), 6);
        assert_eq!(edges[outputs[0].index()], "!");

        let local_edges = searcher
            .local_edges()
            .iter()
            .map(|x| x.index())
            .collect::<HashSet<_>>();
        assert_eq!(local_edges.len(), 2);
        assert_eq!(local_edges, HashSet::from([1, 4]));
        assert_eq!(
            local_edges
                .iter()
                .map(|i| edges[*i])
                .collect::<HashSet<_>>(),
            HashSet::from(["|1", "|4"])
        );
    }
    {
        let nodes = searcher.nodes();
        assert_eq!(nodes.len(), 3);

        let a = nodes.get(0);
        assert_eq!(a.inputs().len(), 2);
        assert_eq!(a.inputs()[0].index(), 0);
        assert_eq!(a.inputs()[1].index(), 1);
        assert_eq!(a.outputs().len(), 2);
        assert_eq!(a.outputs()[0].index(), 2);
        assert_eq!(a.outputs()[1].index(), 3);
        assert_eq!(
            a.predecessors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([])
        );
        assert_eq!(
            a.successors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );

        let b = nodes.get(1);
        assert_eq!(b.inputs().len(), 2);
        assert_eq!(b.inputs()[0].index(), 3);
        assert_eq!(b.inputs()[1].index(), 4);
        assert_eq!(b.outputs().len(), 1);
        assert_eq!(b.outputs()[0].index(), 5);
        assert_eq!(
            b.predecessors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([0])
        );
        assert_eq!(
            b.successors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([2])
        );

        let c = nodes.get(2);
        assert_eq!(c.inputs().len(), 2);
        assert_eq!(c.inputs()[0].index(), 5);
        assert_eq!(c.inputs()[1].index(), 2);
        assert_eq!(c.outputs().len(), 1);
        assert_eq!(c.outputs()[0].index(), 6);
        assert_eq!(
            c.predecessors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1])
        );
        assert_eq!(
            c.successors()
                .iter()
                .map(Node::index)
                .collect::<HashSet<_>>(),
            HashSet::from([usize::MAX])
        );
    }
    {
        let edges = searcher.edges();
        assert_eq!(edges.len(), 7);
    }
}
