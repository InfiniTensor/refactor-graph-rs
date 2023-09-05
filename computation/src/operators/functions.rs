use super::{
    binary::{Binary, BinaryOpType},
    pool::{Pool, PoolAttributes, PoolOpType},
    unary::{Unary, UnaryOpType},
};
use crate::Node;
use smallvec::SmallVec;
use std::rc::Rc;

/// See <https://onnx.ai/onnx/operators/onnx__Abs.html>.
#[inline]
pub fn abs() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Abs,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Acos.html>.
#[inline]
pub fn acos() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Acos,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Acosh.html>.
#[inline]
pub fn acosh() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Acosh,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Add.html>.
#[inline]
pub fn add() -> Node {
    Rc::new(Binary {
        ty: BinaryOpType::Add,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__AffineGrid.html>.
#[inline]
pub fn affine_grid() -> Node {
    unimplemented!()
}

/// See <https://onnx.ai/onnx/operators/onnx__And.html>.
#[inline]
pub fn and() -> Node {
    unimplemented!()
}

/// See <https://onnx.ai/onnx/operators/onnx__ArgMax.html>.
#[inline]
pub fn arg_max() -> Node {
    unimplemented!()
}

/// See <https://onnx.ai/onnx/operators/onnx__ArgMin.html>.
#[inline]
pub fn arg_min() -> Node {
    unimplemented!()
}

/// See <https://onnx.ai/onnx/operators/onnx__Asin.html>.
#[inline]
pub fn asin() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Asin,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Asinh.html>.
#[inline]
pub fn asinh() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Asinh,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Atan.html>.
#[inline]
pub fn atan() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Atan,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__Atanh.html>.
#[inline]
pub fn atanh() -> Node {
    Rc::new(Unary {
        ty: UnaryOpType::Atanh,
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__AveragePool.html>.
#[inline]
pub fn average_pool(
    ceil_mode: bool,
    dilations: Option<SmallVec<[i64; 2]>>,
    kernel_shape: SmallVec<[i64; 2]>,
    pads: Option<SmallVec<[i64; 4]>>,
    strides: Option<SmallVec<[i64; 2]>>,
) -> Node {
    Rc::new(Pool {
        ty: PoolOpType::Average,
        ceil_mode,
        kernel_shape,
        attrs: PoolAttributes {
            dilations,
            pads,
            strides,
        },
    })
}

/// See <https://onnx.ai/onnx/operators/onnx__BatchNormalization.html>.
#[inline]
pub fn batch_normalization() -> Node {
    unimplemented!()
}

//     Bernoulli,
//     BitShift,
//     BitwiseAnd,
//     BitwiseNot,
//     BitwiseOr,
//     BitwiseXor,
//     BlackmanWindow,
//     Cast,
//     CastLike,
//     Ceil,
//     Celu,
//     CenterCropPad,
//     Clip,
//     Col2Im,
//     Compress,
//     Concat,
//     ConcatFromSequence,
//     Constant,
//     ConstantOfShape,
//     Conv,
//     ConvInteger,
//     ConvTranspose,
//     Cos,
//     Cosh,
//     CumSum,
//     DFT,
//     DeformConv,
//     DepthToSpace,
//     DequantizeLinear,
//     Det,
//     Div,
//     Dropout,
//     DynamicQuantizeLinear,
//     Einsum,
//     Elu,
//     Equal,
//     Erf,
//     Exp,
//     Expand,
//     EyeLike,
//     Flatten,
//     Floor,
//     GRU,
//     Gather,
//     GatherElements,
//     GatherND,
//     Gelu,
//     Gemm,
//     GlobalAveragePool,
//     GlobalLpPool,
//     GlobalMaxPool,
//     Greater,
//     GreaterOrEqual,
//     GridSample,
//     GroupNormalization,
//     HammingWindow,
//     HannWindow,
//     HardSigmoid,
//     HardSwish,
//     Hardmax,
//     Identity,
//     If,
//     ImageDecoder,
//     InstanceNormalization,
//     IsInf,
//     IsNaN,
//     LRN,
//     LSTM,
//     LayerNormalization,
//     LeakyRelu,
//     Less,
//     LessOrEqual,
//     Log,
//     LogSoftmax,
//     Loop,
//     LpNormalization,
//     LpPool,
//     MatMul,
//     MatMulInteger,
//     Max,
//     MaxPool,
//     MaxRoiPool,
//     MaxUnpool,
//     Mean,
//     MeanVarianceNormalization,
//     MelWeightMatrix,
//     Min,
//     Mish,
//     Mod,
//     Mul,
//     Multinomial,
//     Neg,
//     NegativeLogLikelihoodLoss,
//     NonMaxSuppression,
//     NonZero,
//     Not,
//     OneHot,
//     Optional,
//     OptionalGetElement,
//     OptionalHasElement,
//     Or,
//     PRelu,
//     Pad,
//     Pow,
//     QLinearConv,
//     QLinearMatMul,
//     QuantizeLinear,
//     RNN,
//     RandomNormal,
//     RandomNormalLike,
//     RandomUniform,
//     RandomUniformLike,
//     Range,
//     Reciprocal,
//     ReduceL1,
//     ReduceL2,
//     ReduceLogSum,
//     ReduceLogSumExp,
//     ReduceMax,
//     ReduceMean,
//     ReduceMin,
//     ReduceProd,
//     ReduceSum,
//     ReduceSumSquare,
//     RegexFullMatch,
//     Relu,
//     Reshape,
//     Resize,
//     ReverseSequence,
//     RoiAlign,
//     Round,
//     STFT,
//     Scan,
//     Scatter,
//     ScatterElements,
//     ScatterND,
//     Selu,
//     SequenceAt,
//     SequenceConstruct,
//     SequenceEmpty,
//     SequenceErase,
//     SequenceInsert,
//     SequenceLength,
//     SequenceMap,
//     Shape,
//     Shrink,
//     Sigmoid,
//     Sign,
//     Sin,
//     Sinh,
//     Size,
//     Slice,
//     Softmax,
//     SoftmaxCrossEntropyLoss,
//     Softplus,
//     Softsign,
//     SpaceToDepth,
//     Split,
//     SplitToSequence,
//     Sqrt,
//     Squeeze,
//     StringConcat,
//     StringNormalizer,
//     StringSplit,
//     Sub,
//     Sum,
//     Tan,
//     Tanh,
//     TfIdfVectorizer,
//     ThresholdedRelu,
//     Tile,
//     TopK,
//     Transpose,
//     Trilu,
//     Unique,
//     Unsqueeze,
//     Upsample,
//     Where,
//     Xor,
