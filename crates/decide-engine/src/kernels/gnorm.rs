//! Gated RMSNorm × SiLU(z) after the recurrence (`kernels/gnorm.metal`).
use super::{buf, bytes, Arg};
use crate::config::NV;
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

#[repr(C)]
struct GNormParams { ldz: u32 }

pub struct GNorm {
    gnorm: ComputePipelineState,
}

impl GNorm {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self { gnorm: dev.pipeline("gated_rmsnorm_silu")? })
    }

    /// `cpu_ref::gated_rmsnorm_silu`: per (row, value head), RMSNorm of `o`
    /// (fp32 `[t, 32, 128]`) over 128 dims with weight `w` (fp32 `[128]`, no +1),
    /// times `silu(z)`; `z` is read from rows of stride `ldz` (a byte offset into
    /// `P` at column `LIN_Z`). Writes fp16 `y` `[t, 4096]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_silu(&self, enc: &ComputeCommandEncoderRef, o: Arg, z: Arg, w: Arg, y: Arg,
                              t: usize, ldz: usize) {
        enc.set_compute_pipeline_state(&self.gnorm);
        buf(enc, 0, o); buf(enc, 1, z); buf(enc, 2, w); buf(enc, 3, y);
        bytes(enc, 4, &GNormParams { ldz: ldz as u32 });
        // One simdgroup per (value head, row).
        enc.dispatch_threads(MTLSize::new(32, NV as u64, t as u64), MTLSize::new(32, 1, 1));
    }
}
