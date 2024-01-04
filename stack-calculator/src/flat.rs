//! 平铺对象的栈计算器，包括一个实时的版本和一个非实时的版本。

use crate::RealtimeCalculator as _;
use std::{alloc::Layout, collections::HashSet, ops::Range};

/// 平铺对象的栈计算器。
pub struct Calculator;

impl crate::Calculator for Calculator {
    fn calculate(
        self,
        topology: &graph_topo::GraphTopo,
        manager: &mut impl crate::Manager,
    ) -> usize {
        let global_outputs = HashSet::<usize>::from_iter(topology.global_outputs());

        let mut rt_cal = RealtimeCalculator::default();
        for (i, _inputs, outputs) in topology {
            for i in outputs {
                if !global_outputs.contains(&i) {
                    manager.set_tensor_offset(i, rt_cal.alloc(manager.tensor_layout(i)).start);
                }
            }
            manager.set_workspace_offset(i, rt_cal.alloc(manager.workspace_layout(i)).start);
        }
        rt_cal.peak()
    }
}

/// 实时的平铺对象的栈计算器。
#[derive(Default, Debug)]
pub struct RealtimeCalculator {
    pos: usize,
}

impl crate::RealtimeCalculator for RealtimeCalculator {
    fn alloc(&mut self, obj: Layout) -> Range<usize> {
        if obj.size() == 0 {
            return 0..0;
        }

        let start = crate::align(self.pos, obj.align());
        self.pos = start + obj.size();

        start..self.pos
    }

    #[inline]
    fn free(&mut self, _range: Range<usize>) {
        // Nothing to do.
    }

    #[inline]
    fn peak(&self) -> usize {
        self.pos
    }
}
