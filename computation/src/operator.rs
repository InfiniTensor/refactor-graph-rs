use common::{ddim, sdim, udim};
use smallvec::SmallVec;

/// 算子。
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum Operator {
    BatchNormalization {
        epsilon: f32,
    },
    Broadcast,
    Cast,
    Clip,
    Concat(Axis),
    Conv(Box<PoolModifier>),
    Gather(Axis),
    GlobalPool,
    MatMul {
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
    },
    Pool(Box<PoolAttributes>),
    Reduce(Box<ReduceAttributes>),
    Select(SelectType),
    SimpleBinary(SimpleBinaryType),
    SimpleUnary(SimpleUnaryType),
    Slice(Box<SliceAttribute>),
    Softmax(Axis),
    Split(Axis),
    Transpose(Box<Permutation>),
    Where,
}

static_assertions::const_assert!(std::mem::size_of::<Operator>() <= 16);

/// 轴。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Axis(pub udim);

/// 池化属性。
#[derive(Clone, Debug)]
pub struct PoolAttributes {
    /// 池化的类型。
    pub ty: PoolType,
    /// 池化核的形状。
    pub kernel_shape: SmallVec<[ddim; 2]>,
    /// 池化修饰器。
    pub modifier: PoolModifier,
}

/// 池化修饰器。
#[derive(Clone, Debug)]
pub struct PoolModifier(SmallVec<[ddim; 16]>);

/// 池化类型。
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PoolType {
    Max,
    Average,
    Lp,
}

/// 规约类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ReduceType {
    /// 均值。
    Mean,
    /// 一范数。
    L1,
    /// 二范数。
    L2,
    /// 对数和。
    LogSum,
    /// 对数和的 e 指数。
    LogSumExp,
    /// 最大值。
    Max,
    /// 最小值。
    Min,
    /// 乘积。
    Prod,
    /// 求和。
    Sum,
    /// 平方和。
    SumSquare,
}

/// 规约属性。
#[derive(Clone, Debug)]
pub struct ReduceAttributes {
    /// 规约类型。
    pub ty: ReduceType,
    /// 轴。
    pub axes: SmallVec<[Axis; 8]>,
}

/// 选择类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum SelectType {
    /// 最小值。
    Min,
    /// 最大值。
    Max,
}

/// 简单双目运算类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SimpleBinaryType {
    /// 加。
    Add,
    /// 减。
    Sub,
    /// 乘。
    Mul,
    /// 除。
    Div,
    /// 幂。
    Pow,
    /// 与。
    And,
    /// 或。
    Or,
    /// 异或。
    Xor,
    /// 位与。
    BitAnd,
    /// 位或。
    BitOr,
    /// 位异或。
    BitXor,
    /// Equal.
    EQ,
    /// Not equal.
    NE,
    /// Less than.
    LT,
    /// Less than or equal.
    LE,
    /// Greater than.
    GT,
    /// Greater than or equal.
    GE,
}

/// 简单单目运算类型。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SimpleUnaryType {
    /// 绝对值。
    Abs,
    /// 反余弦。
    Acos,
    /// 反双曲余弦。
    Acosh,
    /// 反正弦。
    Asin,
    /// 反双曲正弦。
    Asinh,
    /// 反正切。
    Atan,
    /// 反双曲正切。
    Atanh,
    /// 余弦。
    Cos,
    /// 双曲余弦。
    Cosh,
    /// 正弦。
    Sin,
    /// 双曲正弦。
    Sinh,
    /// 正切。
    Tan,
    /// 双曲正切。
    Tanh,
    /// Relu.
    Relu,
    /// 开方。
    Sqrt,
    /// Sigmoid.
    Sigmoid,
    /// Erf。
    Erf,
    /// 相反数。
    Neg,
    /// 取反。
    Not,
    /// 位取反。
    BitNot,
}

/// 切片属性的一个维度。
#[derive(Clone, Debug)]
pub struct SliceDim {
    /// 起始维度。
    pub start: udim,
    /// 步长，可以为负。
    pub step: sdim,
    /// 切后的维度长度。
    pub len: udim,
}

/// 切片属性。
#[derive(Clone, Debug)]
pub struct SliceAttribute(pub SmallVec<[SliceDim; 4]>);

/// 转置排列。
#[derive(Clone, Debug)]
pub struct Permutation(pub SmallVec<[Axis; 8]>);

impl From<(&str, &str)> for Operator {
    fn from((op_type, args): (&str, &str)) -> Self {
        macro_rules! binary {
            ($t:ident) => {
                Self::SimpleBinary(SimpleBinaryType::$t)
            };
        }
        macro_rules! unary {
            ($t:ident) => {
                Self::SimpleUnary(SimpleUnaryType::$t)
            };
        }

        match op_type {
            "MatMul" => {
                let mut args = args.split(',');
                let parse_factor = |s: &str| {
                    let (_, alpha) = s.split_once("=0x").unwrap();
                    u32::from_str_radix(alpha, 16).map(f32::from_bits)
                };
                Self::MatMul {
                    alpha: parse_factor(args.next().unwrap()).unwrap(),
                    beta: parse_factor(args.next().unwrap()).unwrap(),
                    transpose_a: args.next().unwrap().contains('T'),
                    transpose_b: args.next().unwrap().contains('T'),
                }
            }
            "Gather" => Self::Gather(Axis::from_args(args)),
            "ReduceMean" => parse_reduce(op_type, args),

            #[rustfmt::skip]    "Add" => binary!(Add    ),
            #[rustfmt::skip]    "Sub" => binary!(Sub    ),
            #[rustfmt::skip]    "Mul" => binary!(Mul    ),
            #[rustfmt::skip]    "Div" => binary!(Div    ),
            #[rustfmt::skip]    "Pow" => binary!(Pow    ),
            #[rustfmt::skip]    "And" => binary!(And    ),
            #[rustfmt::skip]     "Or" => binary!(Or     ),
            #[rustfmt::skip]    "Xor" => binary!(Xor    ),
            #[rustfmt::skip] "BitAnd" => binary!(BitAnd ),
            #[rustfmt::skip]  "BitOr" => binary!(BitOr  ),
            #[rustfmt::skip] "BitXor" => binary!(BitXor ),
            #[rustfmt::skip]     "EQ" => binary!(EQ     ),
            #[rustfmt::skip]     "NE" => binary!(NE     ),
            #[rustfmt::skip]     "LT" => binary!(LT     ),
            #[rustfmt::skip]     "LE" => binary!(LE     ),
            #[rustfmt::skip]     "GT" => binary!(GT     ),
            #[rustfmt::skip]     "GE" => binary!(GE     ),

            #[rustfmt::skip]     "Abs" => unary!(Abs    ),
            #[rustfmt::skip]    "Acos" => unary!(Acos   ),
            #[rustfmt::skip]   "Acosh" => unary!(Acosh  ),
            #[rustfmt::skip]    "Asin" => unary!(Asin   ),
            #[rustfmt::skip]   "Asinh" => unary!(Asinh  ),
            #[rustfmt::skip]    "Atan" => unary!(Atan   ),
            #[rustfmt::skip]   "Atanh" => unary!(Atanh  ),
            #[rustfmt::skip]     "Cos" => unary!(Cos    ),
            #[rustfmt::skip]    "Cosh" => unary!(Cosh   ),
            #[rustfmt::skip]     "Sin" => unary!(Sin    ),
            #[rustfmt::skip]    "Sinh" => unary!(Sinh   ),
            #[rustfmt::skip]     "Tan" => unary!(Tan    ),
            #[rustfmt::skip]    "Tanh" => unary!(Tanh   ),
            #[rustfmt::skip]    "Relu" => unary!(Relu   ),
            #[rustfmt::skip]    "Sqrt" => unary!(Sqrt   ),
            #[rustfmt::skip] "Sigmoid" => unary!(Sigmoid),
            #[rustfmt::skip]     "Erf" => unary!(Erf    ),
            #[rustfmt::skip]     "Neg" => unary!(Neg    ),
            #[rustfmt::skip]     "Not" => unary!(Not    ),
            #[rustfmt::skip]  "BitNot" => unary!(BitNot ),

            "Softmax" => Self::Softmax(Axis::from_args(args)),
            "Split" => Self::Split(Axis::from_args(args)),
            "Transpose" => Self::Transpose(Box::new(Permutation(Axis::vec_from_args(args)))),
            "Where" => Self::Where,

            _ => todo!("unsupported operator \"{op_type}\""),
        }
    }
}

impl Axis {
    #[inline]
    fn from_args(args: &str) -> Self {
        let (axis, _) = args.split_once('/').unwrap();
        Self(axis.parse().unwrap())
    }

    fn vec_from_args<T: FromIterator<Self>>(args: &str) -> T {
        args.trim_start_matches('[')
            .trim_end_matches(']')
            .split_whitespace()
            .map(|d| Axis(d.parse::<udim>().unwrap()))
            .collect()
    }
}

fn parse_reduce(op_type: &str, args: &str) -> Operator {
    let (axes, _) = args.split_once('/').unwrap();
    Operator::Reduce(Box::new(ReduceAttributes {
        ty: match op_type {
            "ReduceMean" => ReduceType::Mean,
            "ReduceL1" => ReduceType::L1,
            "ReduceL2" => ReduceType::L2,
            "ReduceLogSum" => ReduceType::LogSum,
            "ReduceLogSumExp" => ReduceType::LogSumExp,
            "ReduceMax" => ReduceType::Max,
            "ReduceMin" => ReduceType::Min,
            "ReduceProd" => ReduceType::Prod,
            "ReduceSum" => ReduceType::Sum,
            "ReduceSumSquare" => ReduceType::SumSquare,
            _ => unreachable!("unsupported reduce \"{op_type}\""),
        },
        axes: Axis::vec_from_args(axes),
    }))
}
