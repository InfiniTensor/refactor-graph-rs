use std::any::Any;

pub trait OperatorEq: Downcast {
    fn op_eq(&self, other: &dyn OperatorEq) -> bool;
}

pub trait Downcast {
    fn as_any(&self) -> &dyn Any;
}
