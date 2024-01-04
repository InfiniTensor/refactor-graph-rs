use bitvec::vec::BitVec;
use common::udim;
use graph_topo::ucount;

/// 广播器，支持任意数量输入形状相互广播的优化表示。
#[derive(Clone, Debug)]
pub struct Broadcaster {
    /// 所有输入的各维度步长，形如 `[[udim; n]; m]`。
    ///
    /// - `n` 是压缩后的张量维度；
    /// - `m` 是 `inputs_count + 1`；
    strides: Vec<udim>,
    /// 输入张量的数量。
    inputs_count: ucount,
    /// 输出张量的元素数量。
    output_size: udim,
}

impl Broadcaster {
    /// 从所有输入的形状构造广播器。
    pub fn from_inputs_shape(mut inputs: Vec<&[udim]>) -> Self {
        let mut state = BitVec::<usize>::repeat(false, inputs.len());
        let mut factors = vec![1; inputs.len()];
        let mut output_size = 1;
        let mut strides = Vec::new();

        loop {
            let mut next = BitVec::<usize>::repeat(false, inputs.len());
            let shape = match inputs
                .iter_mut()
                // 为所有 input 标号
                .enumerate()
                // 取出最后一维
                .filter_map(|(i, input)| {
                    input.split_last().map(|(&dim, head)| {
                        *input = head;
                        next.set(i, dim != 1);
                        dim
                    })
                })
                // 更新形状
                .fold(None, |acc, dim| match acc {
                    Some(1) | None => Some(dim),
                    Some(shape) => {
                        assert!(dim == 1 || dim == shape);
                        Some(shape)
                    }
                }) {
                Some(1) => continue,
                Some(shape) => shape,
                None => break,
            };
            if next != state {
                state = next;
                strides.resize(strides.len() + inputs.len() + 1, 0);

                for ((state, factor), dim) in state
                    .iter()
                    .zip(factors.iter_mut())
                    .zip(strides.iter_mut().rev())
                {
                    if *state {
                        *dim = *factor;
                        *factor *= shape;
                    }
                }
            } else {
                for (state, factor) in state.iter().zip(factors.iter_mut()) {
                    if *state {
                        *factor *= shape;
                    }
                }
            }
            output_size *= shape;
        }
        if strides.len() == inputs.len() + 1 && strides.iter().all(|&x| x == 1) {
            strides.clear();
        } else {
            strides.reverse();
        }

        Self {
            strides,
            inputs_count: inputs.len() as _,
            output_size,
        }
    }

    /// 从输出元素序号定位输入元素序号。
    pub fn locate(&self, mut k: udim, ans: &mut [udim]) {
        debug_assert_eq!(ans.len(), self.inputs_count as usize);

        let each = self.inputs_count as usize + 1;
        for i in 0..self.strides.len() / each {
            let dim = &self.strides[each * i..][..each];
            let (div, dim) = dim.split_last().unwrap();

            let quot = k / div;
            k %= div;
            for (ans, dim) in ans.iter_mut().zip(dim) {
                *ans += dim * quot;
            }
        }
    }

    /// 输出张量的元素数量。
    #[inline]
    pub fn output_size(&self) -> udim {
        self.output_size
    }

    /// 判断广播器是否表示需要广播。
    #[inline]
    pub fn need_broadcast(&self) -> bool {
        !self.strides.is_empty()
    }
}
