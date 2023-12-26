use crate::{align, Calculator};
use bitvec::vec::BitVec;
use std::alloc::Layout;

/// 平铺对象的栈计算器。
pub struct FlatCalculator;

impl Calculator for FlatCalculator {
    fn calculate(
        self,
        topology: &graph_topo::GraphTopo,
        manager: &mut impl crate::Manager,
    ) -> usize {
        let flags = {
            let mut flags = BitVec::<usize>::new();
            for i in topology.connections() {
                let i = i.0 as usize;
                if i >= flags.len() {
                    flags.resize(i + 1, false);
                }
                flags.set(i, true);
            }
            for i in topology.global_outputs() {
                flags.set(i.0 as usize, false);
            }
            flags
        };

        let mut ans = 0;
        for (i, _inputs, outputs) in topology {
            for i in outputs {
                if flags[i] {
                    manager.set_tensor_offset(i, put_obj(&mut ans, manager.tensor_layout(i)));
                }
            }
            manager.set_workspace_offset(i, put_obj(&mut ans, manager.workspace_layout(i)));
        }
        ans
    }
}

#[inline(always)]
fn put_obj(size: &mut usize, obj: Layout) -> usize {
    if obj.size() == 0 {
        *size
    } else {
        let offset = align(*size, obj.align());
        *size = offset + obj.size();
        offset
    }
}
