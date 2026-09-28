//! Gated-delta recurrence. By default the fast kernel of
//! `kernels/delta_fast.metal` (`gated_delta_fast`); with
//! `DECIDE_DELTA=baseline` (read once, at construction) the reference
//! `kernels/delta.metal` (`gated_delta`, MLX's algorithm), so parity problems
//! can be bisected. The method signature is the same either way.
use super::{buf, bytes, Arg};
use crate::config::{DV, NV};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

/// Which kernel [`Delta`] dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaImpl {
    /// `kernels/delta_fast.metal` (`gated_delta_fast`): a simdgroup per 16 value
    /// columns, lane l holding dk [4l, 4l + 4) of each; 1.4 ms/layer at T = 493.
    Fast,
    /// `kernels/delta.metal` (`gated_delta`): a simdgroup per value column, two
    /// `simd_sum`s per token; 7.4 ms/layer at T = 493.
    Baseline,
}

/// Value columns per simdgroup of the fast kernel (its `R`).
const FAST_COLS_PER_SIMDGROUP: usize = 16;

#[repr(C)]
struct DeltaParams { t: u32, ldab: u32, write_state: u32 }

pub struct Delta {
    imp: DeltaImpl,
    pso: ComputePipelineState,
}

impl Delta {
    /// The implementation is `DECIDE_DELTA`: unset or `fast` → the fast kernel,
    /// `baseline` → the reference kernel.
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        let imp = match std::env::var("DECIDE_DELTA").as_deref() {
            Err(_) | Ok("") | Ok("fast") => DeltaImpl::Fast,
            Ok("baseline") => DeltaImpl::Baseline,
            Ok(other) => anyhow::bail!("DECIDE_DELTA={other}: expected `fast` or `baseline`"),
        };
        Self::with_impl(dev, imp)
    }

    /// Explicit choice, for bisecting.
    pub fn with_impl(dev: &DecideDevice, imp: DeltaImpl) -> anyhow::Result<Self> {
        let name = match imp { DeltaImpl::Fast => "gated_delta_fast", DeltaImpl::Baseline => "gated_delta" };
        Ok(Self { imp, pso: dev.pipeline(name)? })
    }

    pub fn implementation(&self) -> DeltaImpl { self.imp }

    /// `cpu_ref::gated_delta` over `t` rows, fp32 throughout. `q`, `k` are
    /// `[t, 16, 128]`, `v` and the output `o` `[t, 32, 128]`; `a`, `b` are read
    /// from rows of stride `ldab` (value head h at column h), i.e. byte offsets
    /// into `P` at columns `LIN_A` and `LIN_B`. The state is `[32][128 dv][128 dk]`;
    /// `state_in` may be `state_out` (an in-place chunk continuation), and
    /// with `write_state` false `state_out` is not written. Chunk-continuation
    /// dispatches that share a state buffer must be encoded on one serial
    /// compute encoder: the engine's buffers are untracked, so nothing else
    /// orders one chunk's state write before the next chunk's read.
    ///
    /// Requires `t ≥ 1`. The fast kernel reads `q`, `k` and the state as
    /// `float4`, so their byte offsets must be multiples of 16; every engine
    /// offset (whole rows of `q`/`k`, whole states) is. Row indices `t · ldab`
    /// are 32-bit in the fast kernel, which `MAX_TOKENS` rows of `P` satisfy.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta(&self, enc: &ComputeCommandEncoderRef, q: Arg, k: Arg, v: Arg, a: Arg, b: Arg,
                       a_log: Arg, dt_bias: Arg, state_in: Arg, state_out: Arg, o: Arg,
                       t: usize, ldab: usize, write_state: bool) {
        assert!(t >= 1, "delta: T must be at least 1");
        assert!(ldab >= NV, "delta: ldab = {ldab} must cover the {NV} value heads");
        for (name, arg) in [("q", q), ("k", k), ("v", v), ("a", a), ("b", b), ("a_log", a_log),
                            ("dt_bias", dt_bias), ("state_in", state_in), ("state_out", state_out), ("o", o)] {
            assert!(arg.1 % 4 == 0, "delta: {name} byte offset {} must be a multiple of 4", arg.1);
        }
        enc.set_compute_pipeline_state(&self.pso);
        buf(enc, 0, q); buf(enc, 1, k); buf(enc, 2, v); buf(enc, 3, a); buf(enc, 4, b);
        buf(enc, 5, a_log); buf(enc, 6, dt_bias); buf(enc, 7, state_in); buf(enc, 8, state_out); buf(enc, 9, o);
        bytes(enc, 10, &DeltaParams { t: t as u32, ldab: ldab as u32, write_state: write_state as u32 });
        match self.imp {
            DeltaImpl::Baseline => {
                // One simdgroup per (value head, value column); lane l owns dk ∈ [4l, 4l + 4).
                enc.dispatch_threads(MTLSize::new(32, DV as u64, NV as u64), MTLSize::new(32, 4, 1));
            }
            DeltaImpl::Fast => {
                for (name, arg) in [("q", q), ("k", k), ("state_in", state_in), ("state_out", state_out)] {
                    assert!(arg.1 % 16 == 0, "delta (fast): {name} byte offset {} must be a multiple of 16", arg.1);
                }
                assert!((t as u64) * (ldab as u64) + (NV as u64) <= u32::MAX as u64,
                        "delta (fast): t · ldab = {} · {ldab} must fit in 32 bits", t);
                // One simdgroup (threadgroup) per 16 value columns of one value head:
                // threadgroup (x, h) owns columns [16x, 16x + 16) of head h.
                enc.dispatch_thread_groups(
                    MTLSize::new((DV / FAST_COLS_PER_SIMDGROUP) as u64, NV as u64, 1),
                    MTLSize::new(32, 1, 1),
                );
            }
        }
    }
}
