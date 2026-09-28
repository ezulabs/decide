//! Row RMSNorm, fp32 in, fp16 out (`kernels/rmsnorm.metal`).
use super::{buf, bytes, Arg};
use crate::config::EPS;
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

#[repr(C)]
struct RowNormParams { h: u32, ld_in: u32, ld_out: u32, eps: f32 }

pub struct Norm {
    plain: ComputePipelineState,
    add1: ComputePipelineState,
}

impl Norm {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self {
            plain: dev.pipeline_with_consts_u32("rmsnorm_rows", &[(0, 0)])?,
            add1: dev.pipeline_with_consts_u32("rmsnorm_rows", &[(0, 1)])?,
        })
    }

    /// y[r] = rmsnorm(x[rowmap[r]]) · (w or 1+w) as fp16; `rows` threadgroups.
    /// Full passes bind an identity `rowmap`.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(&self, enc: &ComputeCommandEncoderRef, x: Arg, w: Arg, y: Arg, rowmap: Arg,
                   rows: usize, h: usize, ld_in: usize, ld_out: usize, add_one: bool) {
        assert!(h <= ld_in && h <= ld_out, "rmsnorm: H = {h} exceeds a row stride (ld_in = {ld_in}, ld_out = {ld_out})");
        enc.set_compute_pipeline_state(if add_one { &self.add1 } else { &self.plain });
        buf(enc, 0, x); buf(enc, 1, w); buf(enc, 2, y); buf(enc, 3, rowmap);
        bytes(enc, 4, &RowNormParams { h: h as u32, ld_in: ld_in as u32, ld_out: ld_out as u32, eps: EPS });
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }
}
