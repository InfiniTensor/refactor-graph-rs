use crate::{
    tensor::{DimExpr, Shape},
    Edge,
};

pub trait OutputInference {
    fn infer(&self, inputs: &[Edge]) -> InferResult;
}

pub type InferResult = Result<Vec<Edge>, InferError>;

#[derive(Debug, PartialEq)]
pub enum InferError {
    InputsLenMismatch,
    SizeMismatch,
    DataTypeMismatch,
    BroadcastError,
}

pub fn multidir_broadcast(shapes: &[&Shape]) -> Option<Shape> {
    let mut candidates = shapes.iter().map(|x| x.0.iter().rev()).collect::<Vec<_>>();

    let mut ans = smallvec::SmallVec::new();
    loop {
        let mut dim = None;
        let mut i = 0;
        while i < candidates.len() {
            match candidates[i].next() {
                Some(new) => {
                    dim = match dim.take() {
                        Some(DimExpr::Value(1)) | None => Some(new.clone()),
                        Some(e) if *new == e || *new == DimExpr::Value(1) => Some(e),
                        _ => return None,
                    };
                    i += 1;
                }
                None => {
                    let _ = candidates.swap_remove(i);
                }
            }
        }
        if let Some(dim) = dim {
            ans.push(dim);
        } else {
            break;
        }
    }
    ans.reverse();
    Some(Shape(ans))
}

pub fn uinidir_broadcast(target: &Shape, test: &Shape) -> bool {
    target.0.len() >= test.0.len() && {
        let mut target = target.0.iter().rev();
        for b in test.0.iter().rev() {
            let a = target.next().unwrap();
            if b != a && *b != DimExpr::Value(1) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn shape(val: &[i64]) -> Shape {
        Shape(
            val.iter()
                .map(|&x| DimExpr::Value(x))
                .collect::<smallvec::SmallVec<_>>(),
        )
    }

    #[test]
    fn test_multidir() {
        const CASES: [(&[i64], &[i64], &[i64]); 5] = [
            (&[2, 3, 4, 5], &[2, 3, 4, 5], &[]),
            (&[2, 3, 4, 5], &[2, 3, 4, 5], &[5]),
            (&[2, 3, 4, 5], &[4, 5], &[2, 3, 4, 5]),
            (&[2, 3, 4, 5], &[1, 4, 5], &[2, 3, 1, 1]),
            (&[2, 3, 4, 5], &[3, 4, 5], &[2, 1, 1, 1]),
        ];

        for (i, (ans, a, b)) in CASES.iter().enumerate() {
            let result = multidir_broadcast(&[&shape(a), &shape(b)]);
            assert_eq!(
                Some(shape(ans)),
                result,
                "Multidirectional broadcast test case #{i} failed."
            );
        }
    }

    #[test]
    fn test_unidir() {
        const CASES: [(&[i64], &[i64]); 4] = [
            (&[2, 3, 4, 5], &[]),
            (&[2, 3, 4, 5], &[5]),
            (&[2, 3, 4, 5], &[2, 1, 1, 5]),
            (&[2, 3, 4, 5], &[3, 1, 5]),
        ];

        for (i, (a, b)) in CASES.iter().enumerate() {
            assert!(
                uinidir_broadcast(&shape(a), &shape(b)),
                "Unidirectional broadcast test case #{i} failed."
            );
        }
    }
}
