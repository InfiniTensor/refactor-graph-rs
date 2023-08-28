use common::DataType;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub enum Edge {
    Tensor(Tensor),
    ShapeVariable(ShapeVariable),
}

impl Default for Edge {
    fn default() -> Self {
        Self::Tensor(Tensor {
            dt: DataType::UNDEFINED,
            shape: Default::default(),
        })
    }
}

#[derive(Clone, Default, Debug)]
pub struct Shape(SmallVec<[i64; 4]>);

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dt: DataType,
    pub shape: Shape,
}

#[derive(Clone, Debug)]
pub struct ShapeVariable {
    pub shape: Shape,
}
