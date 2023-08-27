use common::DataType;
use smallvec::SmallVec;

pub enum Edge {
    Tensor(Tensor),
    ShapeVariable(ShapeVariable),
}

pub struct Shape(SmallVec<[i64; 4]>);

pub struct Tensor {
    pub dt: DataType,
    pub shape: Shape,
}

pub struct ShapeVariable {
    pub shape: Shape,
}
