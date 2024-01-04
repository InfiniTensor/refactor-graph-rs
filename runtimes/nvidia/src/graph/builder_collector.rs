use crate::{
    driver::ContextGuard,
    kernel::{GraphBuilder, GraphUser, Resources},
};
use graph_topo::{EdgeIndices, EdgeRange};
use std::alloc::Layout;

pub(super) struct BuilderCollector<'a> {
    builders: Vec<Box<dyn GraphBuilder>>,
    resources: Resources,
    src: &'a computation::Graph,
}

impl<'a> BuilderCollector<'a> {
    #[inline]
    pub fn new(src: &'a computation::Graph) -> Self {
        Self {
            builders: Vec::new(),
            resources: Resources::default(),
            src,
        }
    }
}

impl BuilderCollector<'_> {
    pub fn push(
        &mut self,
        node_idx: usize,
        inputs: &EdgeIndices,
        outputs: &EdgeRange,
        ctx: &ContextGuard,
    ) -> Vec<Layout> {
        let tensors = inputs
            .iter()
            .map(|i| &self.src.0.edges[*i as usize].0)
            .chain(outputs.clone().map(|i| &self.src.0.edges[i].0))
            .collect::<Vec<_>>();
        let (inputs, outputs) = tensors.split_at(inputs.len());

        let (op, _) = &self.src.0.nodes[node_idx];
        let builder = op.builder(inputs, outputs, &mut self.resources, ctx);
        let workspace = builder.worksapce();
        self.builders.push(builder);

        std::slice::from_ref(&workspace)
            .iter()
            .cloned()
            .chain(inputs.iter().map(|t| t.blob_mem_layout()))
            .chain(outputs.iter().map(|t| t.blob_mem_layout()))
            .map(cuda_align)
            .collect()
    }

    #[inline]
    pub fn take(self) -> (Vec<Box<dyn GraphBuilder>>, Resources) {
        (self.builders, self.resources)
    }
}

#[inline]
fn cuda_align(layout: Layout) -> Layout {
    const MIN_ALIGN: usize = 32;
    const MAX_ALIGN: usize = 256;
    layout
        .align_to(
            layout
                .size()
                .next_power_of_two()
                .max(MIN_ALIGN)
                .min(MAX_ALIGN),
        )
        .unwrap()
}
