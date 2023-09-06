use crate::Graph;

pub trait ForeignSubgrpah {
    fn to_kernel(self) -> Graph;
}
