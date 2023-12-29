use crate::{align, Calculator};
use std::{alloc::Layout, collections::HashSet};

/// 平铺对象的栈计算器。
pub struct FlatCalculator;

impl Calculator for FlatCalculator {
    fn calculate(
        self,
        topology: &graph_topo::GraphTopo,
        manager: &mut impl crate::Manager,
    ) -> usize {
        let global_outputs = HashSet::<usize>::from_iter(topology.global_outputs());

        let mut ans = 0;
        for (i, _inputs, outputs) in topology {
            for i in outputs {
                if !global_outputs.contains(&i) {
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
