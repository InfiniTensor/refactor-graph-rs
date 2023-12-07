use crate::{ddim, sdim, udim};
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
    pub axes: SmallVec<[Axis; 4]>,
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
pub struct Permutation(pub SmallVec<[udim; 4]>);
