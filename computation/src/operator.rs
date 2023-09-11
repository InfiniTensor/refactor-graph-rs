#![allow(missing_docs)]

use crate::{infer::InferResult, Edge};
use std::{cell::OnceCell, collections::HashMap, str::FromStr, sync::Mutex};

static MAP: Mutex<OnceCell<OpRepo>> = Mutex::new(OnceCell::new());

#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct Operator {
    pub ty: OpType,
    pub attributes: HashMap<String, Attribute>,
}

pub type InferFn = fn(&Operator, Vec<Edge>) -> InferResult;

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct OpType(usize);

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub struct UnknownOp;

pub struct Op {
    pub name: &'static str,
    pub inference: InferFn,
}

#[derive(Default)]
struct OpRepo {
    map: Vec<Op>,
    rev_map: HashMap<&'static str, usize>,
    known_list: HashMap<&'static str, InferFn>,
}

impl FromStr for OpType {
    type Err = UnknownOp;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut map = MAP.lock().unwrap();
        let repo = map.get_mut().unwrap();
        if let Some(id) = repo.rev_map.get(s) {
            Ok(OpType(*id))
        } else if let Some((name, inference)) = repo.known_list.remove_entry(s) {
            let id = repo.map.len();
            repo.map.push(Op {
                name: name.clone(),
                inference,
            });
            repo.rev_map.insert(name, id);
            Ok(OpType(id))
        } else {
            Err(UnknownOp)
        }
    }
}

impl OpType {
    pub fn register<I>(ops: I)
    where
        I: IntoIterator<Item = (&'static str, InferFn)>,
    {
        let mut map = MAP.lock().unwrap();
        let _ = map.get_or_init(Default::default);
        let repo = map.get_mut().unwrap();
        for (name, inference) in ops {
            if repo.rev_map.contains_key(&name) || repo.known_list.contains_key(&name) {
                panic!("Operator {name} already registered");
            }
            repo.known_list.insert(name, inference);
        }
    }
}
