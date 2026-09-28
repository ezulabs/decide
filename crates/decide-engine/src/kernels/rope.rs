//! Attention prep: zero-centred q/k head norm + partial RoPE, and the K/V cache
//! write (`kernels/rope.metal`). Two kernels, so that the pruned last layer can
//! feed q and k/v from different row sets. `inv_freq` is an fp32 buffer of
//! `config::inv_freq()`, uploaded once.
use super::{buf, bytes, Arg};
use crate::config::{MAX_TOKENS, NKV, NQ};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;

#[repr(C)]
struct QRopeParams { ldq: u32 }

#[repr(C)]
struct KVRopeParams { ldkv: u32, kcol: u32, vcol: u32, pos0: u32 }

pub struct Rope {
    q: ComputePipelineState,
    kv: ComputePipelineState,
}

impl Rope {
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        Ok(Self { q: dev.pipeline("q_norm_rope")?, kv: dev.pipeline("kv_norm_rope")? })
    }

    /// `cpu_ref::head_norm_rope` of the 16 q heads of `rows` rows of `pq` (fp32,
    /// q at column 0, row stride `ldq`); row r is at absolute position
    /// `qpos[r]` (u32). Writes fp16 `qo` `[rows, 16, 256]`.
    #[allow(clippy::too_many_arguments)]
    pub fn q_norm_rope(&self, enc: &ComputeCommandEncoderRef, pq: Arg, qw: Arg, inv_freq: Arg, qpos: Arg,
                       qo: Arg, rows: usize, ldq: usize) {
        enc.set_compute_pipeline_state(&self.q);
        buf(enc, 0, pq); buf(enc, 1, qw); buf(enc, 2, inv_freq); buf(enc, 3, qpos); buf(enc, 4, qo);
        bytes(enc, 5, &QRopeParams { ldq: ldq as u32 });
        enc.dispatch_thread_groups(MTLSize::new(NQ as u64, rows as u64, 1), MTLSize::new(256, 1, 1));
    }

    /// For `rows` rows of `pkv` (fp32, row stride `ldkv`), row r at position
    /// `pos0 + r`: writes `cpu_ref::head_norm_rope` of the 4 k heads (at column
    /// `kcol`) to the K cache `kc` and copies the 4 v heads (at column `vcol`)
    /// to the V cache `vc`; both caches are fp16 `[MAX_TOKENS, 4, 256]`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_norm_rope(&self, enc: &ComputeCommandEncoderRef, pkv: Arg, kw: Arg, inv_freq: Arg, kc: Arg,
                        vc: Arg, rows: usize, ldkv: usize, kcol: usize, vcol: usize, pos0: usize) {
        assert!(pos0 + rows <= MAX_TOKENS, "kv_norm_rope: positions {pos0}..{} overrun the {MAX_TOKENS}-row cache", pos0 + rows);
        enc.set_compute_pipeline_state(&self.kv);
        buf(enc, 0, pkv); buf(enc, 1, kw); buf(enc, 2, inv_freq); buf(enc, 3, kc); buf(enc, 4, vc);
        bytes(enc, 5, &KVRopeParams { ldkv: ldkv as u32, kcol: kcol as u32, vcol: vcol as u32, pos0: pos0 as u32 });
        // Slots 0–3 are the k heads, 4–7 the v heads.
        enc.dispatch_thread_groups(MTLSize::new(2 * NKV as u64, rows as u64, 1), MTLSize::new(256, 1, 1));
    }
}
