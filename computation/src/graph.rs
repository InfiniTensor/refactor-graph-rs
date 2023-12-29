use crate::{Blob, LayoutType, Operator, Shape, Tensor};
use common::DataType;
use graph_topo::ucount;
use std::{collections::HashMap, fmt, str::Lines, sync::Arc};

/// 计算图。
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Graph(pub graph_topo::Graph<(Operator, String), (Tensor, String)>);

impl From<(&str, Vec<u8>)> for Graph {
    fn from((text, data): (&str, Vec<u8>)) -> Self {
        let mut lines = text.lines();

        let (nodes, topology) = parse_operators(&mut lines);
        let (global_inputs, global_outputs) = parse_global(&mut lines);
        let edges = parse_tensors(&mut lines, data);

        let graph_topo::Graph {
            topology,
            nodes,
            edges: _,
        } = graph_topo::Builder {
            topology,
            global_inputs,
            global_outputs,
            nodes,
            edges: HashMap::<usize, ()>::new(),
        }
        .build();
        Graph(graph_topo::Graph {
            topology,
            nodes,
            edges,
        })
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (op, name)) in self.0.nodes.iter().enumerate() {
            writeln!(f, "*{i:>4}. {name:>32} {op:?}")?;
        }
        for (i, (tensor, name)) in self.0.edges.iter().enumerate() {
            writeln!(f, "%{i:>4}. {name:>32} {tensor}")?;
        }
        Ok(())
    }
}

type OpAndName = (Operator, String);
type InputAndOutput = (Vec<usize>, Vec<usize>);

fn parse_operators(
    lines: &mut Lines<'_>,
) -> (HashMap<usize, OpAndName>, HashMap<usize, InputAndOutput>) {
    let mut nodes = HashMap::new();
    let mut topology = HashMap::new();
    for i in 0usize.. {
        let Some(line) = lines.next() else {
            panic!("invalid graph");
        };
        if line.trim().is_empty() {
            break;
        }

        let (head, body) = line.split_once('(').unwrap();
        let (body, tail) = body.split_at(body.rfind(')').unwrap());

        let mut head = head.split_whitespace();
        let _ = head.next().unwrap();
        let name = head.next().unwrap();
        let op_type = head.next().unwrap();
        assert!(head.next().is_none());
        let args = body;

        assert!(nodes
            .insert(i, (Operator::from((op_type, args)), name.into()))
            .is_none());
        assert!(topology
            .insert(i, parse_topo(tail.trim_start_matches(')')))
            .is_none());
    }
    (nodes, topology)
}

fn parse_global(lines: &mut Lines<'_>) -> (Vec<usize>, Vec<usize>) {
    let Some(line) = lines.next() else {
        panic!("invalid graph");
    };
    assert!(lines.next().unwrap().is_empty());
    parse_topo(line.trim_start_matches("graph."))
}

fn parse_tensors(lines: &mut Lines<'_>, data: Vec<u8>) -> Vec<(Tensor, String)> {
    let data = Arc::new(data);
    let mut edges = Vec::new();
    for line in lines {
        let mut line = line.split_whitespace();

        let i = line
            .next()
            .unwrap()
            .trim_start_matches('%')
            .trim_end_matches('.')
            .parse::<ucount>()
            .unwrap();
        debug_assert_eq!(edges.len(), i as usize);

        let name = line.next().unwrap();
        let data_type = line.next().unwrap();
        let layout = line.next().unwrap();
        let begin =
            usize::from_str_radix(line.next().unwrap().trim_start_matches("0x"), 16).unwrap();
        let len =
            usize::from_str_radix(line.next().unwrap().trim_start_matches("+0x"), 16).unwrap();

        assert_eq!(line.next(), Some("["));
        let mut shape = line.collect::<Vec<_>>();
        assert_eq!(shape.pop(), Some("]"));

        let data_type = data_type.parse::<DataType>().unwrap();
        let shape = Shape(shape.into_iter().map(|x| x.parse().unwrap()).collect());
        let layout = layout.parse::<LayoutType>().unwrap();
        let blob = if len != 0 {
            debug_assert_eq!(
                shape.elements_len() as usize * data_type.layout().size(),
                len
            );
            Some(Blob {
                data: data.clone(),
                offset: begin,
            })
        } else {
            None
        };

        edges.push((
            Tensor {
                data_type,
                shape,
                layout,
                blob,
            },
            name.into(),
        ));
    }
    edges
}

fn parse_topo(line: &str) -> (Vec<usize>, Vec<usize>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    let mut line = line.split_whitespace();
    loop {
        match line.next().unwrap() {
            "<-" => break,
            num => outputs.push(num.trim_start_matches('%').parse().unwrap()),
        }
    }
    for num in line {
        inputs.push(num.trim_start_matches('%').parse().unwrap());
    }

    (inputs, outputs)
}
