use crate::edge::{Edge, Shape};

pub type InferResult = Result<Vec<Edge>, InferError>;

#[derive(Debug, PartialEq)]
pub enum InferError {
    SizeMismatch,
    DataTypeMismatch,
    BroadcastError,
}

/// Infer multidirectional broadcast.
pub fn infer_multi_broadcast(shape_x: Shape, shape_y: Shape) -> Option<Shape> {
    let mut broadcasted_shape = vec![];

    let dims_x = shape_x.reverse_dims();
    let dims_y = shape_y.reverse_dims();

    for (dim_x, dim_y) in dims_x.iter().zip(dims_y.iter()) {
        match (dim_x, dim_y) {
            (Some(dim_x), Some(dim_y)) => {
                if *dim_x == *dim_y {
                    broadcasted_shape.push(Some(*dim_x))
                } else if *dim_x == 1 {
                    broadcasted_shape.push(Some(*dim_y))
                } else if *dim_y == 1 {
                    broadcasted_shape.push(Some(*dim_x))
                } else {
                    return None;
                }
            }
            (Some(dim_x), None) => {
                if *dim_x > 1 {
                    broadcasted_shape.push(Some(*dim_x))
                } else {
                    broadcasted_shape.push(None)
                }
            }
            (None, Some(dim_y)) => {
                if *dim_y > 1 {
                    broadcasted_shape.push(Some(*dim_y))
                } else {
                    broadcasted_shape.push(None)
                }
            }
            (None, None) => broadcasted_shape.push(None),
        }
    }

    broadcasted_shape.reverse();

    Some(Shape::new(broadcasted_shape))
}
