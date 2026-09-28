//! Embedding gather (`kernels/embed.metal`).
use super::{buf, bytes, Arg};
use crate::config::HIDDEN;
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

pub struct Embed {
    gather: ComputePipelineState,
}

impl Embed {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self { gather: dev.pipeline("embed_gather_bf16")? })
    }

    /// out[t] = embed[ids[t]] (bf16 → f32), rows [T, HIDDEN].
    pub fn embed(&self, enc: &ComputeCommandEncoderRef, ids: Arg, table: Arg, out: Arg, t: usize) {
        enc.set_compute_pipeline_state(&self.gather);
        buf(enc, 0, ids); buf(enc, 1, table); buf(enc, 2, out);
        bytes(enc, 3, &(HIDDEN as u32));
        enc.dispatch_threads(MTLSize::new(HIDDEN as u64, t as u64, 1), MTLSize::new(256, 1, 1));
    }
}
