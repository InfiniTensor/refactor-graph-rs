//! 栈计算抽象。
//!
//! 这个 crate 定义了 [`Calculator`] 和 [`Manager`] 两个 trait，用于抽象栈计算的过程。
//! [`Calculator`] 用于分配栈空间并计算栈的容量需求，[`Manager`] 用于管理栈计算过程中的信息。

#![deny(warnings, missing_docs)]

mod flat;

pub use flat::FlatCalculator;
use graph_topo::GraphTopo;
use std::{alloc::Layout, collections::HashSet};

/// 栈计算器。
pub trait Calculator {
    /// 与 `manager` 交互，根据给定的图拓扑计算每个对象在栈上的偏移并返回栈容量需求。
    fn calculate(self, topology: &GraphTopo, manager: &mut impl Manager) -> usize;
}

/// 栈计算管理器。
pub trait Manager {
    /// `i` 号节点需要的工作空间布局。
    fn workspace_layout(&self, i: usize) -> Layout;

    /// `i` 号边需要的栈空间布局。
    fn tensor_layout(&self, i: usize) -> Layout;

    /// 设置 `i` 号节点工作空间在栈空间的偏移。
    fn set_workspace_offset(&mut self, i: usize, offset: usize);

    /// 设置 `i` 号边在栈空间的偏移。
    fn set_tensor_offset(&mut self, i: usize, offset: usize);
}

#[inline(always)]
const fn align(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

#[inline(always)]
fn global_outputs_set(topology: &GraphTopo) -> HashSet<usize> {
    topology
        .global_outputs()
        .into_iter()
        .map(|i| i.0 as usize)
        .collect()
}
