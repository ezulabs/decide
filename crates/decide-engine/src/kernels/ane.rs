//! The GPU's half of the GPU+ANE MLP split: folding the Neural Engine's
//! partial `down_proj` output into the residual stream (`kernels/ane.metal`).
use super::{buf, bytes, Arg};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

#[repr(C)]
struct AnePartialParams { h: u32, ld_x: u32, ld_p: u32, inv_scale: f32 }

pub struct Ane {
    add: ComputePipelineState,
}

impl Ane {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self { add: dev.pipeline("ane_add_partial")? })
    }

    /// `x[r] += partial[r] · inv_scale` over `rows` rows of `h` columns:
    /// fp32 `x` (row stride `ld_x`) and the ANE's fp16 `partial` (row stride
    /// `ld_p`), whose values are `1 / inv_scale` times the true ones.
    #[allow(clippy::too_many_arguments)]
    pub fn add_partial(&self, enc: &ComputeCommandEncoderRef, x: Arg, partial: Arg, rows: usize, h: usize,
                       ld_x: usize, ld_p: usize, inv_scale: f32) {
        assert!(h <= ld_x && h <= ld_p, "ane_add_partial: H = {h} exceeds a row stride ({ld_x}, {ld_p})");
        enc.set_compute_pipeline_state(&self.add);
        buf(enc, 0, x); buf(enc, 1, partial);
        bytes(enc, 2, &AnePartialParams { h: h as u32, ld_x: ld_x as u32, ld_p: ld_p as u32, inv_scale });
        enc.dispatch_threads(MTLSize::new(h as u64, rows as u64, 1), MTLSize::new(64, 1, 1));
    }
}
