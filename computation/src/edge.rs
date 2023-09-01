use common::DataType;

#[derive(Clone, Debug, PartialEq)]
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

impl Edge {
    pub fn tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }
}

// #[derive(Clone, Default, Debug)]
// pub struct Shape(SmallVec<[i64; 4]>);

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Shape([Option<u64>; 4]);

impl Shape {
    pub fn new(shape: Vec<Option<u64>>) -> Self {
        assert!(shape.len() == 4);
        Self([shape[0], shape[1], shape[2], shape[3]])
    }

    pub fn dims(&self) -> &[Option<u64>; 4] {
        &self.0
    }

    pub fn size(&self) -> usize {
        let mut size = 0;
        self.0.iter().for_each(|dim| {
            if dim.is_some() {
                size += 1
            }
        });
        size
    }

    pub fn reverse_dims(&self) -> [Option<u64>; 4] {
        let mut dims = self.0;
        dims.reverse();
        dims
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dt: DataType,
    pub shape: Shape,
}

impl Tensor {
    pub fn new(dt: DataType, shape: Shape) -> Self {
        Self { dt, shape }
    }

    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    pub fn data_type(&self) -> DataType {
        self.dt
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeVariable {
    pub shape: Shape,
}

impl ShapeVariable {
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }
}
