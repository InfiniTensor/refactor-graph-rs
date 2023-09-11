#![allow(missing_docs)]

use crate::{infer::InferResult, Edge};
use once_cell::sync::OnceCell;
use std::{
    collections::HashMap,
    str::FromStr,
    sync::{Mutex, RwLock},
};

static MAP: OnceCell<OpRepo> = OnceCell::new();

#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
    Tensor(Edge),
    Tensors(Vec<Edge>),
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
    map: RwLock<Map>,
    known_list: Mutex<HashMap<&'static str, InferFn>>,
}

#[derive(Default)]
struct Map {
    map: Vec<Op>,
    rev_map: HashMap<&'static str, usize>,
}

impl FromStr for OpType {
    type Err = UnknownOp;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let repo = MAP.get().unwrap();
        let lock = repo.map.read().unwrap();
        if let Some(id) = lock.rev_map.get(s) {
            Ok(OpType(*id))
        } else {
            let known_list = &mut *repo.known_list.lock().unwrap();
            if let Some((name, inference)) = known_list.remove_entry(s) {
                let id = lock.map.len();
                drop(lock);
                let mut lock = repo.map.write().unwrap();
                lock.map.push(Op {
                    name: name.clone(),
                    inference,
                });
                lock.rev_map.insert(name, id);
                Ok(OpType(id))
            } else {
                Err(UnknownOp)
            }
        }
    }
}

impl OpType {
    pub fn register<I>(ops: I)
    where
        I: IntoIterator<Item = (&'static str, InferFn)>,
    {
        let repo = MAP.get_or_init(Default::default);
        let rev_map = &repo.map.read().unwrap().rev_map;
        let known_list = &mut *repo.known_list.lock().unwrap();
        for (name, inference) in ops {
            if rev_map.contains_key(&name) || known_list.contains_key(&name) {
                panic!("Operator {name} already registered");
            }
            known_list.insert(name, inference);
        }
    }
}

impl Operator {
    pub fn infer(&self, inputs: Vec<Edge>) -> InferResult {
        let repo = MAP.get().unwrap();
        let op = &repo.map.read().unwrap().map[self.ty.0];
        (op.inference)(self, inputs)
    }
}
