use crate::{uninit_vec, Blob, RoutineWorkspace};
use graph_topo::GraphTopo;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::alloc::Layout;

const UNINIT_OFFSET: usize = usize::MAX;

pub(crate) fn calculate(
    topology: &GraphTopo,
    routines: &mut [RoutineWorkspace],
    tensors: &[(computation::Tensor, String)],
    calculator: impl stack_calculator::Calculator,
) -> (Vec<Blob>, usize) {
    let mut manager = StackManager {
        routines,
        tensors,
        blobs: {
            let mut blobs = unsafe { uninit_vec(tensors.len()) };
            blobs
                .par_iter_mut()
                .for_each(|blob| *blob = Blob::OnStack(UNINIT_OFFSET));
            blobs
        },
    };
    let stack_len = calculator.calculate(topology, &mut manager);
    (manager.build(topology), stack_len)
}

struct StackManager<'a> {
    routines: &'a mut [RoutineWorkspace],
    tensors: &'a [(computation::Tensor, String)],
    blobs: Vec<Blob>,
}

impl StackManager<'_> {
    fn build(mut self, topology: &GraphTopo) -> Vec<Blob> {
        for i in topology.global_inputs() {
            self.blobs[i] = Blob::empty_extern();
        }
        for i in topology.global_outputs() {
            let size = self.tensors[i].0.blob_mem_layout().size();
            self.blobs[i] = Blob::variable(size);
        }
        for (i, blob) in self.blobs.iter_mut().enumerate() {
            match *blob {
                Blob::OnStack(usize::MAX) => {
                    *blob = Blob::Constant(self.tensors[i].0.blob.as_ref().unwrap().clone());
                }
                Blob::OnStack(_) | Blob::Variable(_) | Blob::Extern(_) => {}
                Blob::Constant(_) => unreachable!(),
            }
        }
        self.blobs
    }
}

impl stack_calculator::Manager for StackManager<'_> {
    #[inline]
    fn tensors_len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn workspace_layout(&self, i: usize) -> Layout {
        const ALIGN: usize = std::mem::align_of::<usize>();
        Layout::from_size_align(self.routines[i].workspace, ALIGN).unwrap()
    }

    #[inline]
    fn tensor_layout(&self, i: usize) -> Layout {
        self.tensors[i].0.blob_mem_layout()
    }

    #[inline]
    fn tensor_offset(&self, i: usize) -> Option<usize> {
        match self.blobs[i] {
            Blob::OnStack(UNINIT_OFFSET) => None,
            Blob::OnStack(offset) => Some(offset),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn set_workspace_offset(&mut self, i: usize, offset: usize) {
        self.routines[i].workspace = offset;
    }

    #[inline]
    fn set_tensor_offset(&mut self, i: usize, offset: usize) {
        self.blobs[i] = Blob::OnStack(offset);
    }
}
