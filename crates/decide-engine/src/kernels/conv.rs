//! Causal depthwise conv + SiLU + per-head L2 norm, and the conv state update
//! (`kernels/conv.metal`).
use super::{buf, bytes, Arg};
use crate::config::CONV_DIM;
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

#[repr(C)]
struct ConvParams { t: u32, ldp: u32 }

pub struct Conv {
    conv: ComputePipelineState,
    state: ComputePipelineState,
}

impl Conv {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self { conv: dev.pipeline("conv_silu_l2")?, state: dev.pipeline("conv_state_update")? })
    }

    /// `cpu_ref::conv_silu_l2`. Reads the qkv columns (0..CONV_DIM) of `t` rows
    /// of `p` (fp32, row stride `ldp`) and `prev`, the 3 raw rows before them
    /// (fp32 `[3, CONV_DIM]`, oldest first); `w` is the fp32 `[CONV_DIM, 4]` conv
    /// weight. Writes fp32 `q`, `k` `[t, 16, 128]` (L2-normed, `q` also scaled
    /// by DK^-½) and `v` `[t, 32, 128]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_silu_l2(&self, enc: &ComputeCommandEncoderRef, p: Arg, prev: Arg, w: Arg,
                        q: Arg, k: Arg, v: Arg, t: usize, ldp: usize) {
        enc.set_compute_pipeline_state(&self.conv);
        buf(enc, 0, p); buf(enc, 1, prev); buf(enc, 2, w);
        buf(enc, 3, q); buf(enc, 4, k); buf(enc, 5, v);
        bytes(enc, 6, &ConvParams { t: t as u32, ldp: ldp as u32 });
        // 64 head slots: 0–15 q heads, 16–31 k heads, 32–63 v heads.
        enc.dispatch_thread_groups(MTLSize::new(64, t as u64, 1), MTLSize::new(128, 1, 1));
    }

    /// `cpu_ref::conv_state_update`: writes to `prev_out` the 3 raw rows that
    /// end this chunk of `t` rows of `p`, drawing on `prev_in` when `t < 3`.
    /// `prev_out` may be `prev_in`.
    pub fn conv_state_update(&self, enc: &ComputeCommandEncoderRef, p: Arg, prev_in: Arg, prev_out: Arg,
                             t: usize, ldp: usize) {
        enc.set_compute_pipeline_state(&self.state);
        buf(enc, 0, p); buf(enc, 1, prev_in); buf(enc, 2, prev_out);
        bytes(enc, 3, &ConvParams { t: t as u32, ldp: ldp as u32 });
        enc.dispatch_threads(MTLSize::new(CONV_DIM as u64, 1, 1), MTLSize::new(256, 1, 1));
    }
}
