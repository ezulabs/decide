//! Row gathers and the candidate-only final decision head (`kernels/head.metal`).
use super::{buf, bytes, Arg};
use crate::config::{EPS, HIDDEN};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

// final_head's row width is a compile-time constant in the kernel.
const _: () = assert!(HIDDEN == 2560, "kernels/head.metal hard-codes HIDDEN = 2560");

#[repr(C)]
struct HeadParams { eps: f32 }

pub struct Head {
    gather_f32: ComputePipelineState,
    gather_f16: ComputePipelineState,
    head: ComputePipelineState,
}

impl Head {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self {
            gather_f32: dev.pipeline("gather_rows_f32")?,
            gather_f16: dev.pipeline("gather_rows_f16")?,
            head: dev.pipeline("final_head")?,
        })
    }

    /// out[i] = x[rowmap[i]] for `rows` fp32 rows of `h` elements.
    pub fn gather_rows_f32(&self, enc: &ComputeCommandEncoderRef, x: Arg, rowmap: Arg, out: Arg, rows: usize, h: usize) {
        Self::gather(&self.gather_f32, enc, x, rowmap, out, rows, h);
    }

    /// out[i] = x[rowmap[i]] for `rows` fp16 rows of `h` elements.
    pub fn gather_rows_f16(&self, enc: &ComputeCommandEncoderRef, x: Arg, rowmap: Arg, out: Arg, rows: usize, h: usize) {
        Self::gather(&self.gather_f16, enc, x, rowmap, out, rows, h);
    }

    fn gather(pso: &ComputePipelineState, enc: &ComputeCommandEncoderRef, x: Arg, rowmap: Arg, out: Arg,
              rows: usize, h: usize) {
        enc.set_compute_pipeline_state(pso);
        buf(enc, 0, x); buf(enc, 1, rowmap); buf(enc, 2, out);
        bytes(enc, 3, &(h as u32));
        enc.dispatch_threads(MTLSize::new(h as u64, rows as u64, 1), MTLSize::new(256, 1, 1));
    }

    /// `cpu_ref::final_head`: the final zero-centred RMSNorm (weight `w`, fp32
    /// `[HIDDEN]`) of each of `rows` fp32 rows of `x` `[rows, HIDDEN]`, then its
    /// logits against the bf16 embedding rows of `table` named by its
    /// candidates: `cand` is u32 `[rows, MAX_CANDIDATES]`, of which row r uses
    /// the first `ncand[r]` (u32 `[rows]`; the kernel clamps it to
    /// MAX_CANDIDATES). Writes fp32 `logits` `[rows, MAX_CANDIDATES]`; entries
    /// past `ncand[r]` are not written.
    #[allow(clippy::too_many_arguments)]
    pub fn final_head(&self, enc: &ComputeCommandEncoderRef, x: Arg, w: Arg, table: Arg, cand: Arg, ncand: Arg,
                      logits: Arg, rows: usize) {
        enc.set_compute_pipeline_state(&self.head);
        buf(enc, 0, x); buf(enc, 1, w); buf(enc, 2, table); buf(enc, 3, cand); buf(enc, 4, ncand); buf(enc, 5, logits);
        bytes(enc, 6, &HeadParams { eps: EPS });
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }
}
