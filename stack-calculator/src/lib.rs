//! 栈计算抽象。
//!
//! 这个 crate 定义了 [`Calculator`] 和 [`Manager`] 两个 trait，用于抽象栈计算的过程。
//! [`Calculator`] 用于分配栈空间并计算栈的容量需求，[`Manager`] 用于管理栈计算过程中的信息。
//!
//! crate 还提供了 2 种预定义的 [`Calculator`] 实现：
//!
//! - [`FlatCalculator`]：平铺所有工作空间和张量，不复用空间。
//! - [`UnidirCalculator`]：单向扩容，最适匹配的计算器，分割满足对齐和容量要求的最小空闲空间进行分配，没有合适的空闲空间时从尾部扩容。

#![deny(warnings, missing_docs)]

pub mod flat;
pub mod unidir;

use graph_topo::GraphTopo;
use std::{alloc::Layout, ops::Range};

/// 栈计算器。
pub trait Calculator {
    /// 与 `manager` 交互，根据给定的图拓扑计算每个对象在栈上的偏移并返回栈容量需求。
    fn calculate(self, topology: &GraphTopo, manager: &mut impl Manager) -> usize;
}

/// 实时栈计算器。
pub trait RealtimeCalculator {
    /// 分配满足 `obj` 要求的空间。
    fn alloc(&mut self, obj: Layout) -> Range<usize>;

    /// 释放 `range` 范围内的空间。
    fn free(&mut self, range: Range<usize>);

    /// 获取栈空间的历史峰值。
    fn peak(&self) -> usize;
}

/// 栈计算管理器。
pub trait Manager {
    /// 获取张量的数量。
    fn tensors_len(&self) -> usize;

    /// `i` 号节点需要的工作空间布局。
    fn workspace_layout(&self, i: usize) -> Layout;

    /// `i` 号边需要的栈空间布局。
    fn tensor_layout(&self, i: usize) -> Layout;

    /// 获取 `i` 号边在栈空间的偏移。
    ///
    /// 如果这个偏移没有设置过，返回 `None`。
    fn tensor_offset(&self, i: usize) -> Option<usize>;

    /// 设置 `i` 号节点工作空间在栈空间的偏移。
    fn set_workspace_offset(&mut self, i: usize, offset: usize);

    /// 设置 `i` 号边在栈空间的偏移。
    fn set_tensor_offset(&mut self, i: usize, offset: usize);
}

#[inline(always)]
const fn align(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}
