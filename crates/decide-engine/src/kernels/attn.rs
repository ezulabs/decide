//! Causal GQA attention with the sigmoid output gate. By default the fast
//! steel-derived kernel of `kernels/attn_fast.metal` (`attn_hd256_fast`, Task
//! 37); with `DECIDE_ATTN=baseline` (read once, at construction) the reference
//! `kernels/attn.metal` (`attn_hd256`: one simdgroup per (row, head), online
//! softmax), so parity problems can be bisected. The method signature is the
//! same either way.
use super::{buf, bytes, Arg};
use crate::config::{NKV, NQ};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

/// Which kernel [`Attn`] dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnImpl {
    /// `kernels/attn_fast.metal` (`attn_hd256_fast`): a threadgroup per (kv
    /// head, 8·RG query rows), simdgroup-matrix QKᵀ and PV over staged K/V
    /// tiles, fp32 online softmax in fragments. The softmax weights are
    /// rounded to fp16 once, as the PV MMA's operand (the row sum uses the
    /// rounded weights, so the output is exactly normalised).
    Fast,
    /// `kernels/attn.metal` (`attn_hd256`): a simdgroup per (row, head)
    /// walking the cache one position at a time, fp32 throughout.
    Baseline,
}

#[repr(C)]
struct AttnParams { ldq: u32 }
#[repr(C)]
struct AttnFastParams { ldq: u32, rows: u32 }

pub struct Attn {
    imp: AttnImpl,
    pso: ComputePipelineState,
    /// The fast kernel's threadgroup size, 128·RG, read from the pipeline
    /// (its `max_total_threads_per_threadgroup` attribute), and the query rows
    /// each threadgroup covers, 8·RG.
    threads: u64,
    rows_per_tg: usize,
    /// The fast kernel's K/V tile bytes, bound as its dynamic threadgroup
    /// memory. Read from the metallib, not mirrored: `attn_hd256_fast_smem`
    /// is a probe kernel whose static threadgroup array is sized by the same
    /// `AttnCfg` instantiation, and its pipeline reports that size. (Static
    /// tiles in the kernel itself compile to code ~23% slower, so the tiles
    /// stay host-sized.)
    smem_bytes: u64,
}

impl Attn {
    /// The implementation is `DECIDE_ATTN`: unset or `fast` → the fast kernel,
    /// `baseline` → the reference kernel.
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        let imp = match std::env::var("DECIDE_ATTN").as_deref() {
            Err(_) | Ok("") | Ok("fast") => AttnImpl::Fast,
            Ok("baseline") => AttnImpl::Baseline,
            Ok(other) => anyhow::bail!("DECIDE_ATTN={other}: expected `fast` or `baseline`"),
        };
        Self::with_impl(dev, imp)
    }

    /// Explicit choice, for bisecting.
    pub fn with_impl(dev: &DecideDevice, imp: AttnImpl) -> anyhow::Result<Self> {
        let name = match imp { AttnImpl::Fast => "attn_hd256_fast", AttnImpl::Baseline => "attn_hd256" };
        let pso = dev.pipeline(name)?;
        let (threads, smem_bytes) = match imp {
            AttnImpl::Baseline => (32, 0),
            AttnImpl::Fast => {
                let threads = pso.max_total_threads_per_threadgroup();
                anyhow::ensure!(threads == 128 || threads == 256,
                                "{name}: expected 128·RG threads per threadgroup, the pipeline says {threads}");
                let smem = dev.pipeline("attn_hd256_fast_smem")?.static_threadgroup_memory_length();
                anyhow::ensure!(smem > 0 && smem <= 32768 && smem % 16 == 0,
                                "attn_hd256_fast_smem: implausible tile size {smem} bytes");
                (threads, smem)
            }
        };
        Ok(Self { imp, pso, threads, rows_per_tg: threads as usize / 16, smem_bytes })
    }

    pub fn implementation(&self) -> AttnImpl { self.imp }

    /// `cpu_ref::attention`. Queries `q` are fp16 `[rows, 16, 256]`; row r at
    /// position `qpos[r]` (u32) attends to rows `0..=qpos[r]` of the fp16 K/V
    /// caches `[.., 4, 256]`. The gate is read from `pq` (the rows' fused
    /// projection, fp32, row stride `ldq`) at column `ATT_GATE`. Writes fp16
    /// `y` `[rows, 4096]`. Requires `rows ≥ 1`, and cache rows `0..=max(qpos)`
    /// written (finite) before the dispatch.
    ///
    /// The fast kernel reads `q` and the caches as 16-byte vectors and the gate
    /// as `float2`, so it also requires 16-byte-aligned `q`/`kc`/`vc` byte
    /// offsets, an 8-byte-aligned `pq` offset with even `ldq`, and 4-byte-aligned
    /// `qpos`/`y` offsets. Every engine offset (whole rows, whole caches) is.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, enc: &ComputeCommandEncoderRef, q: Arg, kc: Arg, vc: Arg, pq: Arg, qpos: Arg,
                     y: Arg, rows: usize, ldq: usize) {
        assert!(rows >= 1, "attention: rows must be at least 1");
        assert!(ldq >= 2 * NQ * 256, "attention: ldq = {ldq} must cover the gate columns");
        enc.set_compute_pipeline_state(&self.pso);
        buf(enc, 0, q); buf(enc, 1, kc); buf(enc, 2, vc); buf(enc, 3, pq); buf(enc, 4, qpos); buf(enc, 5, y);
        match self.imp {
            AttnImpl::Baseline => {
                bytes(enc, 6, &AttnParams { ldq: ldq as u32 });
                enc.dispatch_threads(MTLSize::new(32, NQ as u64, rows as u64), MTLSize::new(32, 1, 1));
            }
            AttnImpl::Fast => {
                for (name, arg) in [("q", q), ("kc", kc), ("vc", vc)] {
                    assert!(arg.1 % 16 == 0, "attention (fast): {name} byte offset {} must be a multiple of 16", arg.1);
                }
                assert!(pq.1 % 8 == 0 && ldq % 2 == 0, "attention (fast): pq offset {} must be a multiple of 8 and ldq = {ldq} even", pq.1);
                assert!(qpos.1 % 4 == 0 && y.1 % 4 == 0, "attention (fast): qpos/y offsets {}/{} must be multiples of 4", qpos.1, y.1);
                bytes(enc, 6, &AttnFastParams { ldq: ldq as u32, rows: rows as u32 });
                enc.set_threadgroup_memory_length(0, self.smem_bytes);
                enc.dispatch_thread_groups(
                    MTLSize::new(rows.div_ceil(self.rows_per_tg) as u64, NKV as u64, 1),
                    MTLSize::new(self.threads, 1, 1),
                );
            }
        }
    }
}
