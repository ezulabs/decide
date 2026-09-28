//! GEMM with fused epilogues. By default the fast steel-derived kernels of
//! `kernels/gemm_fast.metal`, with the tile config chosen per shape by
//! [`decide_metal::gemm_configs::plan`]; with `DECIDE_GEMM=baseline` (read
//! once, at construction) the reference `kernels/gemm.metal`, so parity
//! problems can be bisected. The method signature is the same either way.
use super::{buf, bytes, Arg};
use crate::config::SWIGLU_BLOCK;
use decide_metal::gemm_configs::{self, GemmDispatch};
use decide_metal::metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use decide_metal::DecideDevice;
use std::collections::HashMap;
use std::sync::Mutex;

/// What the GEMM does with its fp32 accumulators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// Writes fp32 `[M, N]`.
    StoreF32 = 0,
    /// Writes fp16 `[M, N]`.
    StoreF16 = 1,
    /// `Y += acc` into fp32, for the residual adds.
    AddF32 = 2,
    /// Treats the N columns as interleaved gate|up blocks of `SWIGLU_BLOCK` and
    /// writes fp16 `[M, N/2]` = `silu(gate)·up`.
    SwigluF16 = 3,
}

/// Which kernels [`Gemm`] dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmImpl {
    /// `kernels/gemm_fast.metal` (`gemm_nt_fast*` / `gemm_nt_small`), per-shape configs.
    Fast,
    /// `kernels/gemm.metal` (`gemm_nt`), 32×32 tiles of 128 threads.
    Baseline,
}

// The fast kernel hard-codes the 16-column gate|up interleave.
const _: () = assert!(SWIGLU_BLOCK == 16, "gemm_fast.metal pairs SwiGLU fragments 16 columns apart");

#[repr(C)]
struct GemmParams { m: u32, n: u32, k: u32, ldx: u32, ldw: u32, ldy: u32, swiglu_block: u32 }

/// Pipeline cache key for the fast path: (kernel name, config id, grid order, epilogue).
type FastKey = (&'static str, u32, u32, u32);

pub struct Gemm {
    dev: &'static DecideDevice,
    imp: GemmImpl,
    /// One baseline pipeline per [`Epilogue`], indexed by its value.
    baseline: [ComputePipelineState; 4],
    /// Fast pipelines, built on first use per (kernel, config, order, epilogue).
    fast: Mutex<HashMap<FastKey, ComputePipelineState>>,
}

impl Gemm {
    /// The implementation is `DECIDE_GEMM`: unset or `fast` → the fast kernels,
    /// `baseline` → the reference kernel.
    pub fn new(dev: &DecideDevice) -> anyhow::Result<Self> {
        let imp = match std::env::var("DECIDE_GEMM").as_deref() {
            Err(_) | Ok("") | Ok("fast") => GemmImpl::Fast,
            Ok("baseline") => GemmImpl::Baseline,
            Ok(other) => anyhow::bail!("DECIDE_GEMM={other}: expected `fast` or `baseline`"),
        };
        Self::with_impl(dev, imp)
    }

    /// Explicit choice, for bisecting.
    pub fn with_impl(dev: &DecideDevice, imp: GemmImpl) -> anyhow::Result<Self> {
        let shared = DecideDevice::shared()?;
        anyhow::ensure!(std::ptr::eq(shared, dev), "Gemm: expected the process-wide DecideDevice");
        let pso = |e: Epilogue| dev.pipeline_with_consts_u32("gemm_nt", &[(0, e as u32)]);
        Ok(Self {
            dev: shared,
            imp,
            baseline: [
                pso(Epilogue::StoreF32)?,
                pso(Epilogue::StoreF16)?,
                pso(Epilogue::AddF32)?,
                pso(Epilogue::SwigluF16)?,
            ],
            fast: Mutex::new(HashMap::new()),
        })
    }

    pub fn implementation(&self) -> GemmImpl { self.imp }

    /// Builds the fast pipelines of every recommended config × epilogue, so a
    /// timed forward pass never compiles a pipeline.
    pub fn warm_up(&self) -> anyhow::Result<()> {
        if self.imp != GemmImpl::Fast {
            return Ok(());
        }
        let mut kernels: Vec<gemm_configs::GemmKernel> = Vec::new();
        let all = gemm_configs::RECOMMENDED.iter().map(|r| r.kernel);
        for kernel in all.chain([gemm_configs::DEFAULT_SMALL, gemm_configs::DEFAULT_TILE]) {
            if !kernels.contains(&kernel) {
                kernels.push(kernel);
            }
        }
        for kernel in kernels {
            let d = gemm_configs::dispatch_for(kernel, 64, 64, 1);
            for epi in [Epilogue::StoreF32, Epilogue::StoreF16, Epilogue::AddF32, Epilogue::SwigluF16] {
                self.fast_pipeline(&d, epi)?;
            }
        }
        Ok(())
    }

    /// Fast pipelines built so far. After [`Self::warm_up`] this must not
    /// grow: growth means a GEMM compiled a pipeline mid-request.
    pub fn fast_pipelines(&self) -> usize {
        self.fast.lock().unwrap().len()
    }

    fn fast_pipeline(&self, d: &GemmDispatch, epi: Epilogue) -> anyhow::Result<ComputePipelineState> {
        let key: FastKey = (d.kernel, d.cfg, d.order, epi as u32);
        let mut cache = self.fast.lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let p = self.dev.pipeline_with_consts_u32(d.kernel, &d.function_constants(epi as u32))?;
        cache.insert(key, p.clone());
        Ok(p)
    }

    /// `Y = X · Wᵀ`, then `epi`. `X` is fp16 `[M, K]` with row stride `ldx`, `W`
    /// fp16 `[N, K]` with row stride `ldw`, and `ldy` is Y's row stride in
    /// elements of its output type. Accumulation is fp32. Requires
    /// `N % 32 == 0`, `K % 32 == 0` and `M ≥ 1`. A slice of W's rows is a byte
    /// offset into `w`: `row_start · ldw · 2`.
    ///
    /// The fast kernels read 16-byte row vectors and store `float2`/`half2`
    /// pairs, so they also require `ldx % 8 == 0`, `ldw % 8 == 0`, `ldy` even,
    /// x/w byte offsets that are multiples of 16 and a y offset that is a
    /// multiple of 8 (fp32 outputs) or 4 (fp16 outputs). Every engine shape and
    /// slice satisfies these; the asserts catch the rest.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, enc: &ComputeCommandEncoderRef, x: Arg, w: Arg, y: Arg,
                m: usize, n: usize, k: usize, ldx: usize, ldw: usize, ldy: usize, epi: Epilogue) {
        assert!(n % 32 == 0 && k % 32 == 0, "gemm: N = {n} and K = {k} must be multiples of 32");
        assert!(m >= 1, "gemm: M must be at least 1");
        assert!(ldx >= k && ldw >= k, "gemm: row strides ldx = {ldx}, ldw = {ldw} must cover K = {k}");
        let y_cols = if epi == Epilogue::SwigluF16 { n / 2 } else { n };
        assert!(ldy >= y_cols, "gemm: row stride ldy = {ldy} must cover the {y_cols} output columns");
        let params = GemmParams {
            m: m as u32, n: n as u32, k: k as u32,
            ldx: ldx as u32, ldw: ldw as u32, ldy: ldy as u32,
            swiglu_block: SWIGLU_BLOCK as u32,
        };

        if self.imp == GemmImpl::Baseline {
            enc.set_compute_pipeline_state(&self.baseline[epi as usize]);
            buf(enc, 0, x); buf(enc, 1, w); buf(enc, 2, y);
            bytes(enc, 3, &params);
            enc.dispatch_thread_groups(MTLSize::new((n / 32) as u64, m.div_ceil(32) as u64, 1), MTLSize::new(128, 1, 1));
            return;
        }

        assert!(ldx % 8 == 0 && ldw % 8 == 0, "gemm (fast): ldx = {ldx} and ldw = {ldw} must be multiples of 8");
        assert!(x.1 % 16 == 0 && w.1 % 16 == 0, "gemm (fast): x/w byte offsets {}/{} must be multiples of 16", x.1, w.1);
        assert!(ldy % 2 == 0, "gemm (fast): ldy = {ldy} must be even");
        let y_align = match epi { Epilogue::StoreF32 | Epilogue::AddF32 => 8, Epilogue::StoreF16 | Epilogue::SwigluF16 => 4 };
        assert!(y.1 % y_align == 0, "gemm (fast): y byte offset {} must be a multiple of {y_align}", y.1);

        let d = gemm_configs::plan(m, n, k, epi as u32);
        let pso = self.fast_pipeline(&d, epi).expect("gemm (fast): pipeline");
        enc.set_compute_pipeline_state(&pso);
        buf(enc, 0, x); buf(enc, 1, w); buf(enc, 2, y);
        bytes(enc, 3, &params);
        // Tile configs stage through threadgroup memory; small configs use it
        // only for split-K (ks > 1). Bind it always (at least 16 bytes) so a
        // kernel that declares the argument never sees an unbound one.
        enc.set_threadgroup_memory_length(0, d.smem_bytes.max(16));
        enc.dispatch_thread_groups(MTLSize::new(d.grid.0, d.grid.1, 1), MTLSize::new(d.threads, 1, 1));
    }
}
