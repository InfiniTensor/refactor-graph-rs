use crate::{align, Calculator};
use std::{
    alloc::Layout,
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    ops::Range,
};

/// 单向扩容的栈计算器。
pub struct UnidirCalculator;

impl Calculator for UnidirCalculator {
    fn calculate(
        self,
        topology: &graph_topo::GraphTopo,
        manager: &mut impl crate::Manager,
    ) -> usize {
        let mut rc = Vec::<usize>::new();
        for i in topology.connections() {
            let i = i.0 as usize;
            if i >= rc.len() {
                rc.resize(i + 1, 0);
            }
            rc[i] += 1;
        }
        for i in topology.global_outputs() {
            rc[i.0 as usize] = 0;
        }

        let mut rt_cal = RealtimeCalculator::default();
        for (i, inputs, outputs) in topology {
            for i in outputs {
                if rc[i] > 0 {
                    manager.set_tensor_offset(i, rt_cal.alloc(manager.tensor_layout(i)).start);
                }
            }
            {
                let workspace = rt_cal.alloc(manager.workspace_layout(i));
                manager.set_workspace_offset(i, workspace.start);
                rt_cal.free(workspace);
            }
            for i in inputs {
                let i = i.0 as usize;
                let rc = &mut rc[i];
                debug_assert_ne!(*rc, 0);
                *rc -= 1;
                if *rc == 0 {
                    let offset = manager.tensor_offset(i).unwrap();
                    rt_cal.free(offset..offset + manager.tensor_layout(i).size());
                }
            }
        }
        rt_cal.peak()
    }
}

#[derive(Default, Debug)]
struct RealtimeCalculator {
    used: usize,
    peak: usize,

    free_headtails: BTreeSet<HeadTail>,
    free_head_tail: HashMap<usize, usize>,
    free_tail_head: HashMap<usize, usize>,
}

impl RealtimeCalculator {
    fn alloc(&mut self, obj: Layout) -> Range<usize> {
        if obj.size() == 0 {
            return 0..0;
        }
        self.used += obj.size();

        if let Some(&HeadTail(Range { start, end })) = self
            .free_headtails
            .range(HeadTail(0..obj.size())..)
            .filter(|&HeadTail(r)| r.end - align(r.start, obj.align()) >= obj.size())
            .next()
        {
            self.free_headtails.remove(&HeadTail(start..end));
            self.free_head_tail.remove(&start);
            self.free_tail_head.remove(&end);

            let (head, tail) = Self::head_tail(start, obj);
            self.insert(start, head);
            self.insert(tail, end);
            head..tail
        } else if let Some(start) = self.free_tail_head.remove(&self.peak) {
            self.free_headtails.remove(&HeadTail(start..self.peak));
            self.free_head_tail.remove(&start);

            let (head, tail) = Self::head_tail(start, obj);
            self.insert(start, head);
            self.peak = tail;
            head..tail
        } else {
            let (head, tail) = Self::head_tail(self.peak, obj);
            self.insert(self.peak, head);
            self.peak = tail;
            head..tail
        }
    }

    fn free(&mut self, obj: Range<usize>) {
        if obj.is_empty() {
            return;
        }
        self.used -= obj.len();

        let Range { mut start, mut end } = obj;
        if let Some((tail, head)) = self.free_tail_head.remove_entry(&start) {
            self.free_head_tail.remove(&head);
            self.free_headtails.remove(&HeadTail(head..tail));
            start = head;
        }
        if let Some((head, tail)) = self.free_head_tail.remove_entry(&end) {
            self.free_tail_head.remove(&tail);
            self.free_headtails.remove(&HeadTail(head..tail));
            end = tail;
        }
        self.insert(start, end);
    }

    #[inline]
    const fn peak(&self) -> usize {
        self.peak
    }

    #[inline]
    fn insert(&mut self, start: usize, end: usize) {
        if end > start {
            self.free_head_tail.insert(start, end);
            self.free_tail_head.insert(end, start);
            self.free_headtails.insert(HeadTail(start..end));
        }
    }

    #[inline(always)]
    const fn head_tail(start: usize, obj: Layout) -> (usize, usize) {
        let head = align(start, obj.align());
        (head, head + obj.size())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct HeadTail(Range<usize>);

impl PartialOrd for HeadTail {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0
            .len()
            .partial_cmp(&other.0.len())
            .or_else(|| self.0.start.partial_cmp(&other.0.start))
    }
}

impl Ord for HeadTail {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .partial_cmp(&other.0.len())
            .unwrap_or_else(|| self.0.start.cmp(&other.0.start))
    }
}
