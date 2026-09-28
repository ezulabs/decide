//! The forward pass: embedding, the 32 decoder layers and the candidate-only
//! decision head, encoded on one serial compute encoder per request.
//!
//! **Chunks.** A span of positions `start..end` runs in chunks of at most
//! `chunk` rows, chunk-major: all 32 layers of one chunk, then the next. Each
//! linear layer carries its conv and recurrent state from chunk to chunk
//! (`conv`, `state`), and each attention layer writes its chunk's K/V rows to
//! the cache before attending. Chunk 0 reads its state from a [`Source`]:
//! zeros for a run from position 0, or the prefix snapshot.
//!
//! **Requests.** Every prompt starts with the same constant prefix (107
//! tokens), so [`Forward::build_snapshot`] runs it once and keeps the state
//! after it: each linear layer's recurrent and conv state, and each attention
//! layer's K/V rows. A request whose ids start with the prefix computes only
//! positions `prefix ..= last answer row` from the snapshot; one that doesn't
//! falls back to the full path from position 0 with zero state. Either way
//! the rows after the last answer row never run (tail truncation). The K/V
//! caches keep the snapshot's rows at positions `0..prefix` between requests;
//! a full-path run overwrites them, and the next snapshot run puts them back.
//!
//! **Follow-ups.** A small LRU cache holds the state at the
//! end of recent prompts' state sections (`…## State\n{state json}\n##
//! Decision schema\n`): the linear layers' recurrent and conv state, and the
//! attention layers' K/V rows from the prefix to there. A request about a
//! new state saves it on the way, at no extra pass over the weights: the
//! chunk that holds the end of the state splits only its recurrence there.
//! A request whose ids start with a cached state's ids then computes only
//! the rows after it, putting its K/V rows back first if another request
//! has overwritten them.
//!
//! **Ordering.** Every buffer here is untracked (as are the weights), so the
//! only thing that orders dependent dispatches is the single serial encoder:
//! a serial encoder runs its dispatches one after another, in encode order.
//! The CPU writes per-request data (staged embedding and candidate rows,
//! answer rows) before encoding and reads results after the command buffer
//! completes, so it never races the GPU. Everything a chunk reads lives at
//! that chunk's offset of a whole-span buffer; nothing is rewritten between
//! chunks of one command buffer.
//!
//! **The embedding table is never bound on the GPU.** A lazily mapped no-copy
//! buffer gets wired almost whole on its first GPU use, which would
//! pin 1.27 GB. The CPU copies the span's rows into `embed_stage`, which
//! `embed_gather_bf16` reads with ids `0..rows`, and each answer's candidate
//! rows into `cand_stage`, which `final_head` reads through the constant
//! remap in `cand`.
use crate::config::*;
use crate::convert::AneConfig;
use crate::kernels::{Arg, Delta, DeltaImpl, Epilogue, Gemm, GemmImpl, Kernels};
use crate::weights::{DType, Weights};
use anyhow::{bail, ensure, Context, Result};
use decide_ane::AneSession;
use block::ConcreteBlock;
use decide_metal::metal::foreign_types::ForeignTypeRef;
use decide_metal::metal::{
    self, Buffer, CommandBufferRef, CommandQueueRef, ComputeCommandEncoderRef, MTLCommandBufferStatus,
    MTLResourceOptions, SharedEvent, SharedEventRef,
};
use decide_metal::DecideDevice;
use std::cell::Cell;
use std::path::Path;

/// Most answers (fields) per request: InternLM accepts 1 to 16 questions.
pub const MAX_ANSWERS: usize = 16;

/// The token ids of `"## Decision schema\n"`, the line that ends a prompt's
/// state (`compile_row`: `{preamble}{state json}\n## Decision schema\n…`).
/// They do not depend on what precedes the line (the newline before `##`
/// always ends the previous pre-token).
/// [`Forward::set_state_marker`] overrides them.
pub const STATE_MARKER: [u32; 4] = [550, 39087, 10485, 198];
/// Most entries the state cache holds (each ~53 MB + 32 KB per state token).
pub const MAX_STATE_ENTRIES: usize = 4;
/// The state cache's capacity at construction.
pub const DEFAULT_STATE_ENTRIES: usize = 2;

/// How long one run's ANE jobs may take before the run is declared hung
/// (`submit_run`): the longest request is a few seconds; a stuck Core ML
/// predict is the failure this guards against.
const ANE_JOB_TIMEOUT_S: f64 = 60.0;

/// Bytes per bf16 embedding row.
const EMBED_ROW: usize = HIDDEN * 2;
/// Byte sizes of the element types.
const F32: usize = 4;
const F16: usize = 2;
const U32: usize = 4;

/// A weight tensor, resolved once at construction: its blob's buffer and the
/// tensor's byte offset in it.
struct W {
    buf: Buffer,
    off: u64,
}

impl W {
    fn arg(&self) -> Arg<'_> {
        (&self.buf, self.off)
    }
}

/// One decoder layer's weights.
struct Layer {
    in_norm: W,
    post_norm: W,
    /// Fused input projection: linear `qkv|z|b|a` [12352, 2560], attention
    /// `q|gate|k|v` [10240, 2560].
    w_in: W,
    w_out: W,
    /// Interleaved gate|up [18432, 2560].
    w_gu: W,
    w_down: W,
    mixer: Mixer,
}

enum Mixer {
    /// A gated-delta layer; `i` indexes the per-linear-layer state buffers.
    Linear { i: usize, conv_w: W, a_log: W, dt_bias: W, gnorm: W },
    /// A full-attention layer; `a` indexes the K/V caches.
    Attention { a: usize, q_norm: W, k_norm: W },
}

/// Buffers allocated once in [`Forward::new`], all shared and untracked.
/// `cap` is the rows per chunk they hold.
struct Buffers {
    /// u32 [MAX_TOKENS] = 0, 1, 2, …: the identity rowmap, and (at byte
    /// offset 4·pos0) the absolute positions of a chunk starting at pos0.
    iota: Buffer,
    /// f32 [ROT_DIM / 2].
    inv_freq: Buffer,
    /// bf16 [MAX_TOKENS, HIDDEN]: the span's embedding rows, staged by the CPU.
    embed_stage: Buffer,
    /// f32 [cap, HIDDEN]: the residual stream.
    x: Buffer,
    /// f16 [cap, HIDDEN]: normed GEMM input.
    h: Buffer,
    /// f32 [cap, LIN_IN_N]: fused projection output (attention layers use row stride ATT_IN_N).
    p: Buffer,
    /// f32 [cap, KEY_DIM] each.
    q: Buffer,
    k: Buffer,
    /// f32 [cap, VALUE_DIM] each.
    v: Buffer,
    o: Buffer,
    /// f16 [cap, VALUE_DIM]: gated-norm / gated-attention output.
    g: Buffer,
    /// f16 [cap, INTER]: SwiGLU output.
    mlp: Buffer,
    /// f16 [cap, Q_DIM]: normed, rotated queries.
    qo: Buffer,
    /// f16 [MAX_TOKENS, NKV, HEAD_DIM] per attention layer.
    kc: Vec<Buffer>,
    vc: Vec<Buffer>,
    /// f32 [NV, DV, DK] per linear layer: the recurrent state.
    state: Vec<Buffer>,
    /// f32 [3, CONV_DIM] per linear layer: the last 3 raw qkv rows. The
    /// update may run in place (`conv_state_update` reads before it writes).
    conv: Vec<Buffer>,
    /// All-zero sources for chunk 0 of a run from position 0.
    zero_state: Buffer,
    zero_conv: Buffer,
    /// u32 [MAX_ANSWERS]: each answer's row within its chunk.
    ans_rows: Buffer,
    /// u32 [MAX_ANSWERS]: each answer's absolute position.
    ans_pos: Buffer,
    /// f32 [MAX_ANSWERS, HIDDEN]: the answer rows' residual stream.
    xs: Buffer,
    /// f16 [MAX_ANSWERS, HIDDEN]: the answer rows' normed GEMM inputs (pruned layer).
    hs: Buffer,
    /// f32 [MAX_ANSWERS, ATT_K]: the answer rows' q|gate projection (pruned layer).
    ps: Buffer,
    /// u32 [MAX_ANSWERS, MAX_CANDIDATES] = 0, 1, 2, …: answer i's candidate c
    /// is row i·MAX_CANDIDATES + c of `cand_stage`.
    cand: Buffer,
    /// u32 [MAX_ANSWERS]: candidates per answer.
    ncand: Buffer,
    /// bf16 [MAX_ANSWERS · MAX_CANDIDATES, HIDDEN]: candidate rows, staged by the CPU.
    cand_stage: Buffer,
    /// f32 [MAX_ANSWERS, MAX_CANDIDATES].
    logits: Buffer,
}

/// Where chunk 0 of a run reads its conv and recurrent state from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// Zero state: a run from position 0.
    Zero,
    /// The prefix snapshot: a run from position `prefix`.
    Snapshot,
    /// State cache entry `i` (an index into `Forward::states`): a run from
    /// the end of that entry's state.
    State(usize),
}

/// How [`Forward::logits`] computed a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// From the prefix snapshot: positions `prefix ..= last answer row`. A
    /// request with a new state also leaves it in the state cache.
    Snapshot,
    /// The full path: positions `0 ..= last answer row` from zero state.
    Full,
    /// A follow-up: the state is in the state cache, so only positions
    /// `end of state ..= last answer row` run.
    StateCache,
}

/// What a request's run starts from.
#[derive(Clone, Copy, Debug)]
enum Plan {
    Full,
    Snapshot,
    /// The snapshot path; the request's (new) state, which ends at
    /// `boundary`, is captured into the state cache on the way.
    Capture { boundary: usize },
    /// State cache entry `i`.
    Hit(usize),
}

/// The state at the end of a prompt's state section: what a
/// follow-up question set about the same state starts from.
struct StateEntry {
    /// Unique per capture, for `Forward::kv_owner`.
    id: u64,
    /// The prompt's ids through the state marker; a request uses the entry
    /// only if its ids start with these.
    ids: Vec<u32>,
    /// Per linear layer: the recurrent state and conv state after `ids`.
    state: Vec<Buffer>,
    conv: Vec<Buffer>,
    /// Per attention layer: the K and V cache rows `prefix..ids.len()`
    /// (fp16), room for `kv_rows` rows.
    k: Vec<Buffer>,
    v: Vec<Buffer>,
    kv_rows: usize,
}

/// The state after the constant prompt prefix.
struct Snapshot {
    ids: Vec<u32>,
    /// Per linear layer: the recurrent state and the conv state after the prefix.
    state: Vec<Buffer>,
    conv: Vec<Buffer>,
    /// Per attention layer: the prefix's K and V cache rows (fp16 bytes), put
    /// back by the CPU after a full-path run has overwritten them.
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
}

/// One chunk of a run.
#[derive(Clone, Copy, Debug)]
struct Chunk {
    /// Absolute position of the chunk's first row.
    pos0: usize,
    rows: usize,
    /// The chunk's first row in `embed_stage` (pos0 − the run's start).
    stage_row: usize,
    /// Chunk 0 of its run: reads state from the run's [`Source`].
    first: bool,
    /// Write the conv and recurrent state after this chunk: a later chunk
    /// (or a snapshot) reads it.
    keep_state: bool,
    /// The request's state ends in this chunk, after its first `t` rows
    /// (1 ≤ t ≤ rows): capture the state there.
    capture: Option<usize>,
    /// The chunk's answers are sorted answers `a0..a0 + n`.
    a0: usize,
    n: usize,
}

/// The Neural Engine's half of the MLP, present when the weight set
/// was built by `decide ane-convert` and the engine was opened with the ANE on.
///
/// Per split layer the GPU signals `event` once the post-attention norm has
/// written `h`, the ANE computes `partial` (intermediate dims `0..n_inter` of
/// the MLP, pre-scaled by `out_scale`) while the GPU runs its own slice, and
/// the next command buffer waits for the ANE's value at its head before
/// folding `partial / out_scale` into the residual stream.
struct AneSplit {
    session: AneSession,
    cfg: AneConfig,
    /// `DECIDE_ANE_SYNC=1`: wait for the norm on the CPU before the ANE runs.
    sync: bool,
    /// f16 [cap, HIDDEN]: the ANE's partial `down_proj` output, bound once.
    partial: Buffer,
    event: SharedEvent,
    /// The last event value used; every job takes the next two.
    used: Cell<u64>,
}

impl AneSplit {
    /// (value the GPU signals after writing `h`, value the ANE signals after
    /// writing `partial`). Strictly increasing, so jobs fire in order.
    fn next_pair(&self) -> (u64, u64) {
        let v = self.used.get() + 1;
        self.used.set(v + 1);
        (v, v + 1)
    }

    /// Intermediate dims the GPU keeps.
    fn n_gpu(&self) -> usize {
        INTER - self.cfg.n_inter
    }
}

/// One run's command buffers. With the ANE off this is the single buffer and
/// serial encoder the engine has always used; with it on, [`Run::split`] ends
/// the current buffer (optionally signalling) and starts the next (optionally
/// waiting at its head — a wait encoded mid-buffer stalls the GPU for
/// milliseconds, see the spike).
struct Run<'p> {
    queue: &'p CommandQueueRef,
    cmd: &'p CommandBufferRef,
    enc: &'p ComputeCommandEncoderRef,
    event: Option<&'p SharedEventRef>,
    /// Every buffer committed so far, for the status check and GPU timing.
    committed: Vec<&'p CommandBufferRef>,
    /// The first error an encoder hit; `submit_run` returns it after the
    /// run has been committed and waited for (so the queue stays consistent).
    failed: Option<anyhow::Error>,
}

impl<'p> Run<'p> {
    /// Record an error. Encoding continues (the buffers already committed
    /// must be balanced), but the run's result is discarded.
    fn fail(&mut self, e: anyhow::Error) {
        if self.failed.is_none() {
            self.failed = Some(e);
        }
    }

    /// The encoder to encode into. Never hold it across a [`Run::split`].
    fn enc(&self) -> &'p ComputeCommandEncoderRef {
        self.enc
    }

    /// Block until every command buffer committed so far has completed. Used
    /// by the ANE's synchronous mode (`DECIDE_ANE_SYNC=1`), which trades the
    /// overlap for a hand-off that cannot race.
    fn wait_committed(&self) {
        if let Some(cmd) = self.committed.last() {
            cmd.wait_until_completed();
        }
    }

    fn split(&mut self, signal: Option<u64>, wait: Option<u64>) {
        self.enc.end_encoding();
        if let (Some(ev), Some(v)) = (self.event, signal) {
            // Signalled from the completion handler, not by `encodeSignalEvent`:
            // the ANE reads `h` on another processor, and an encoded signal
            // fires inside the GPU pipeline, before this buffer's writes to an
            // untracked, shared buffer are necessarily visible there. (Measured:
            // an encoded signal gives a wrong, run-to-run varying MLP; the
            // completion handler matches the GPU-only path to 7e-5.) The event
            // outlives every run: it belongs to `AneSplit`.
            let ptr = ev.as_ptr() as usize;
            let handler = ConcreteBlock::new(move |_cmd: &CommandBufferRef| {
                // SAFETY: `ptr` is the `MTLSharedEvent` owned by this engine's
                // `AneSplit`, which outlives every command buffer it commits.
                unsafe { SharedEventRef::from_ptr(ptr as *mut _) }.set_signaled_value(v);
            })
            .copy();
            self.cmd.add_completed_handler(&handler);
        }
        self.cmd.commit();
        self.committed.push(self.cmd);
        let cmd = self.queue.new_command_buffer();
        if let (Some(ev), Some(v)) = (self.event, wait) {
            cmd.encode_wait_for_event(ev, v);
        }
        self.enc = cmd.new_compute_command_encoder();
        self.cmd = cmd;
    }
}

/// An answer to compute: its index in the caller's list, its logit row and its candidates.
struct Answer<'a> {
    index: usize,
    row: usize,
    candidates: &'a [u32],
}

pub struct Forward {
    dev: &'static DecideDevice,
    k: Kernels,
    layers: Vec<Layer>,
    /// Read by the CPU only (see the module docs).
    embed: W,
    final_norm: W,
    /// The GPU+ANE MLP split, when this weight set and this engine use it.
    /// Declared before `b`: fields drop in declaration order, and the
    /// session's arrays alias `b.h` (see [`Forward::open_ane`]).
    ane: Option<AneSplit>,
    b: Buffers,
    /// Rows the chunk buffers hold.
    cap: usize,
    /// Rows per prefill chunk, ≤ `cap`.
    chunk: usize,
    /// Run the last layer on the answer rows only (on by default).
    prune: bool,
    snapshot: Option<Snapshot>,
    /// The K/V caches' rows `0..prefix` hold the snapshot's rows.
    prefix_clean: bool,
    /// The state cache, most recently used first.
    states: Vec<StateEntry>,
    /// Entries whose state was forgotten: buffers kept for the next
    /// capture. `states.len() + free.len() ≤ state_capacity`.
    free: Vec<StateEntry>,
    /// Its capacity (0 disables it).
    state_capacity: usize,
    /// The token ids that end a prompt's state.
    state_marker: Vec<u32>,
    /// The entry whose K/V rows `prefix..end of its state` the K/V caches hold.
    kv_owner: Option<u64>,
    next_entry_id: u64,
    last_route: Option<Route>,
    /// The first position the last `logits`/`logits_full` call computed.
    last_start: Option<usize>,
    // Last, so that it drops after every tensor handle above.
    weights: Weights,
}

impl Forward {
    /// Open the converted weights in `weights_dir`, resolve every tensor and
    /// allocate the buffers for `CHUNK`-row chunks. The Neural Engine split is
    /// used when `DECIDE_ANE=1` (see [`Forward::open`]).
    pub fn new(weights_dir: &Path) -> Result<Self> {
        Self::open(weights_dir, matches!(std::env::var("DECIDE_ANE").as_deref(), Ok("1")))
    }

    /// [`Forward::new`] with the Neural Engine split explicitly on or off.
    ///
    /// A weight set built by `decide ane-convert` holds only the GPU's share
    /// of each split layer's MLP, so it *must* run with `ane = true`; a plain
    /// weight set has no ANE models and must run with `ane = false`. Either
    /// mismatch is refused rather than quietly computing the wrong thing.
    pub fn open(weights_dir: &Path, ane: bool) -> Result<Self> {
        let weights = Weights::open(weights_dir)?;
        let dev = DecideDevice::shared()?;
        let k = Kernels::new(dev)?;
        // Build the fast GEMM pipelines now, not during the first request.
        k.gemm.warm_up()?;
        let layers = (0..N_LAYERS).map(|l| resolve_layer(&weights, l)).collect::<Result<Vec<_>>>()?;
        let embed = tensor(&weights, "embed", DType::BF16, &[VOCAB, HIDDEN])?;
        let final_norm = tensor(&weights, "final_norm", DType::F32, &[HIDDEN])?;
        let cap = CHUNK;
        let b = Buffers::new(dev, cap, ane && weights.ane().is_some());
        let ane = match (weights.ane().cloned(), ane) {
            (Some(cfg), true) => Some(Self::open_ane(dev, weights_dir, cfg, &b, cap)?),
            (None, false) => None,
            (Some(_), false) => bail!(
                "{} is an ANE weight set (its split layers hold only the GPU's share of each MLP); \
                 open it with the Neural Engine on (DECIDE_ANE=1)",
                weights_dir.display()
            ),
            (None, true) => bail!(
                "{} has no ANE models; build them with `decide ane-convert --weights {} --out <dir>` \
                 and open that directory instead",
                weights_dir.display(),
                weights_dir.display()
            ),
        };
        Ok(Self {
            dev,
            k,
            layers,
            embed,
            final_norm,
            ane,
            b,
            cap,
            chunk: cap,
            prune: true,
            snapshot: None,
            prefix_clean: false,
            states: Vec::new(),
            free: Vec::new(),
            state_capacity: DEFAULT_STATE_ENTRIES,
            state_marker: STATE_MARKER.to_vec(),
            kv_owner: None,
            next_entry_id: 0,
            last_route: None,
            last_start: None,
            weights,
        })
    }

    /// A buffer allocated the way this engine's others are (see
    /// [`Buffers::new`]): tracked while the ANE splits runs across command
    /// buffers, untracked otherwise.
    fn buffer(&self, bytes: usize) -> Buffer {
        if self.ane.is_some() { tracked(self.dev, bytes) } else { shared(self.dev, bytes) }
    }

    /// The converted weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Rows per prefill chunk.
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// Set the rows per prefill chunk, at most the `CHUNK` rows the buffers
    /// hold. For tests: results must not depend on it beyond rounding.
    pub fn set_chunk(&mut self, rows: usize) -> Result<()> {
        ensure!((1..=self.cap).contains(&rows), "chunk of {rows} rows; the buffers hold 1 to {}", self.cap);
        self.chunk = rows;
        Ok(())
    }

    /// Whether the last layer runs on the answer rows only.
    pub fn pruning(&self) -> bool {
        self.prune
    }

    /// Prune the last layer to the answer rows (the default), or run it on
    /// every row. For tests: results must not depend on it beyond rounding.
    /// (`debug_layers` and `debug_layer` never prune.)
    pub fn set_pruning(&mut self, on: bool) {
        self.prune = on;
    }

    /// Load the per-layer Core ML models next to the weights, bind them to the
    /// `h` and `partial` buffers and run each once so the ANE programs are
    /// resident before the first request.
    fn open_ane(dev: &'static DecideDevice, weights_dir: &Path, cfg: AneConfig, b: &Buffers, cap: usize)
                -> Result<AneSplit> {
        ensure!(
            cfg.n_inter % 128 == 0 && cfg.n_inter < INTER && (INTER - cfg.n_inter) % 32 == 0,
            "an ANE split of {} of {INTER} intermediate dims",
            cfg.n_inter
        );
        // Layer N_LAYERS - 1 runs its MLP on gathered answer rows (the pruned
        // path), not on the whole-chunk `h` the session is bound to.
        ensure!(
            cfg.layers < N_LAYERS,
            "{} split layers; at most {} (the last layer stays on the GPU)",
            cfg.layers,
            N_LAYERS - 1
        );
        ensure!(cfg.out_scale.is_finite() && cfg.out_scale > 0.0, "an ANE output scale of {}", cfg.out_scale);
        ensure!(
            cfg.chunk_rows > 0 && cap % cfg.chunk_rows == 0,
            "ANE chunks of {} rows do not tile the {cap}-row buffers",
            cfg.chunk_rows
        );
        check_ane_manifest(weights_dir, &cfg)?;
        let session = AneSession::open(weights_dir, cfg.layers, cfg.chunk_rows, HIDDEN, HIDDEN)
            .with_context(|| format!("loading the ANE models in {}", weights_dir.display()))?;
        // Hazard-TRACKED, unlike every other buffer here: the engine's blanket
        // `Untracked` is sound only because one serial encoder orders every
        // access, and this buffer is written by another processor entirely.
        let partial = tracked(dev, cap * HIDDEN * F16);
        let event = dev.device().new_shared_event();
        let mut session = session;
        // SAFETY: the session's arrays alias `b.h` and `partial`, each `cap`
        // rows of HIDDEN fp16 with stride HIDDEN, for as long as the session
        // lives. The invariant is that no job is in flight when either buffer
        // goes away: `AneSession::drop` waits for its jobs to go idle before
        // it releases the arrays, and `Forward` declares `ane` before `b`, so
        // the session (and that wait) drops before `Buffers` does; `partial`
        // lives in the same `AneSplit` as the session. Within a run, the
        // GPU/ANE ordering of reads and writes is encoded in `encode_mlp`.
        unsafe {
            session.bind(&event, b.h.contents(), HIDDEN, partial.contents(), HIDDEN, cap / cfg.chunk_rows)?;
        }
        session.warm_up()?;
        let sync = matches!(std::env::var("DECIDE_ANE_SYNC").as_deref(), Ok("1"));
        Ok(AneSplit { session, cfg, partial, event, sync, used: Cell::new(0) })
    }

    /// Debug: the normed rows in `h` (what a split layer's ANE job reads) and
    /// the ANE's partial output, both as f32 over `rows` rows. For debugging
    /// the split only; valid right after a run that ended in a split layer.
    pub fn debug_ane_tensors(&self, rows: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let ane = self.ane.as_ref()?;
        let f16s = |buf: &Buffer, n: usize| -> Vec<f32> {
            read_vec::<u16>(buf, n).into_iter().map(|b| half::f16::from_bits(b).to_f32()).collect()
        };
        Some((f16s(&self.b.h, rows * HIDDEN), f16s(&ane.partial, rows * HIDDEN)))
    }

    /// The ANE split in use: its config, or `None` on the GPU-only path.
    pub fn ane(&self) -> Option<&AneConfig> {
        self.ane.as_ref().map(|a| &a.cfg)
    }

    /// Predicts the ANE has run and the seconds they took (cumulative).
    pub fn ane_stats(&self) -> Option<(u64, f64)> {
        self.ane.as_ref().map(|a| a.session.stats())
    }

    /// Milliseconds each ANE model took to load.
    pub fn ane_load_ms(&self) -> Vec<f64> {
        match &self.ane {
            Some(a) => (0..a.cfg.layers).map(|l| a.session.load_ms(l)).collect(),
            None => Vec::new(),
        }
    }

    /// The GEMM implementation in use: `DECIDE_GEMM` at construction.
    pub fn gemm(&self) -> GemmImpl {
        self.k.gemm.implementation()
    }

    /// Switch the GEMM implementation: the fast kernels (the default), or the
    /// baseline reference, for tests and for bisecting parity. The fast GEMM
    /// picks its kernel by M, so a row's rounding can depend on how many rows
    /// share its GEMM (the chunk, or the pruned layer's answer rows); the
    /// baseline computes every row the same way whatever M. A snapshot built
    /// before the switch stays valid, rounded the old way.
    pub fn set_gemm(&mut self, imp: GemmImpl) -> Result<()> {
        if self.gemm() != imp {
            let gemm = Gemm::with_impl(self.dev, imp)?;
            gemm.warm_up()?;
            self.k.gemm = gemm;
        }
        Ok(())
    }

    /// Fast-GEMM pipelines built so far (see `Gemm::fast_pipelines`).
    pub fn gemm_pipelines(&self) -> usize {
        self.k.gemm.fast_pipelines()
    }

    /// The gated-delta implementation in use: `DECIDE_DELTA` at construction.
    pub fn delta(&self) -> DeltaImpl {
        self.k.delta.implementation()
    }

    /// Switch the gated-delta implementation: the fast kernel (the default)
    /// or the baseline reference, for tests and for bisecting parity.
    pub fn set_delta(&mut self, imp: DeltaImpl) -> Result<()> {
        if self.delta() != imp {
            self.k.delta = Delta::with_impl(self.dev, imp)?;
        }
        Ok(())
    }

    /// Run the constant prompt prefix (`tokenizer.prefix_ids()`) from
    /// position 0 with zero state, and keep the state after it: every
    /// request that starts with these ids then runs from the snapshot.
    /// Replaces any earlier snapshot, and empties the state cache.
    pub fn build_snapshot(&mut self, prefix_ids: &[u32]) -> Result<()> {
        check_ids(prefix_ids)?;
        let len = prefix_ids.len();
        ensure!(len < MAX_TOKENS, "a {len}-token prefix leaves no room for a request");
        self.snapshot = None;
        self.prefix_clean = false;
        self.forget_states();
        self.stage_embeddings(prefix_ids);
        let chunks = self.chunks(0, len, true, &[]);
        self.submit_run(|run| chunks.iter().for_each(|ch| self.encode_chunk(run, ch, Source::Zero, None)))?;
        let copy = |buf: &Buffer| {
            let snap = self.buffer(buf.length() as usize);
            // SAFETY: equal lengths; the command buffer has completed.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.contents() as *const u8, snap.contents() as *mut u8, buf.length() as usize)
            };
            snap
        };
        let rows = |cache: &Buffer| read_vec::<u8>(cache, len * KV_DIM * F16);
        self.snapshot = Some(Snapshot {
            ids: prefix_ids.to_vec(),
            state: self.b.state.iter().map(copy).collect(),
            conv: self.b.conv.iter().map(copy).collect(),
            k: self.b.kc.iter().map(rows).collect(),
            v: self.b.vc.iter().map(rows).collect(),
        });
        self.prefix_clean = true;
        Ok(())
    }

    /// The snapshot's prefix length, once built.
    pub fn prefix_len(&self) -> Option<usize> {
        self.snapshot.as_ref().map(|s| s.ids.len())
    }

    /// How the last `logits` or `logits_full` call computed its request.
    pub fn last_route(&self) -> Option<Route> {
        self.last_route
    }

    /// The first position the last `logits` or `logits_full` call computed:
    /// 0 on the full route, the prefix length on the snapshot route, the end
    /// of the cached state on a follow-up. It computed positions from there
    /// to its last answer row.
    pub fn last_start(&self) -> Option<usize> {
        self.last_start
    }

    /// Logits of each answer's candidates. Each answer is (absolute logit
    /// row, candidate token ids); the result has one row of logits per
    /// answer, in the given order.
    ///
    /// Runs from the prefix snapshot when the ids start with the prefix and
    /// every answer row lies after it; otherwise falls back to the full
    /// path. On the snapshot path, a request whose ids start with a cached
    /// state's ids (and whose answer rows lie after it) is a follow-up: it
    /// runs from the end of that state. A request with a new state leaves
    /// the state at its end in the cache, evicting the least recently used
    /// entry.
    pub fn logits<C: AsRef<[u32]>>(&mut self, ids: &[u32], answers: &[(usize, C)]) -> Result<Vec<Vec<f32>>> {
        check_ids(ids)?;
        let answers = sorted_answers(ids.len(), answers)?;
        let first_row = answers[0].row;
        let plan = match &self.snapshot {
            Some(s) if ids.starts_with(&s.ids) && first_row >= s.ids.len() => {
                let prefix = s.ids.len();
                // A follow-up: the longest cached state the ids start with.
                let hit = self
                    .states
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.ids.len() <= first_row && ids.starts_with(&e.ids))
                    .max_by_key(|(_, e)| e.ids.len())
                    .map(|(i, _)| i);
                match hit {
                    Some(i) => Plan::Hit(i),
                    None if self.state_capacity > 0 => match state_boundary(ids, prefix, &self.state_marker, first_row) {
                        Some(boundary) => Plan::Capture { boundary },
                        None => Plan::Snapshot,
                    },
                    None => Plan::Snapshot,
                }
            }
            _ => Plan::Full,
        };
        self.run_answers(ids, &answers, plan)
    }

    /// Hold up to `entries` (≤ MAX_STATE_ENTRIES) states in the state cache;
    /// 0 turns it off. Shrinking evicts the least recently used entries.
    pub fn set_state_cache(&mut self, entries: usize) -> Result<()> {
        ensure!(entries <= MAX_STATE_ENTRIES, "a state cache of {entries} entries; at most {MAX_STATE_ENTRIES}");
        self.state_capacity = entries;
        self.states.truncate(entries);
        self.free.truncate(entries - self.states.len());
        if !self.states.iter().any(|e| Some(e.id) == self.kv_owner) {
            self.kv_owner = None;
        }
        Ok(())
    }

    /// Forget every cached state (the next request about any state is
    /// cold), keeping the entries' buffers for the next captures. For
    /// timing cold requests with the cache on.
    pub fn forget_states(&mut self) {
        self.free.append(&mut self.states);
        self.kv_owner = None;
    }

    /// The state cache's capacity (0: off).
    pub fn state_cache_capacity(&self) -> usize {
        self.state_capacity
    }

    /// The length in tokens (prefix, state and marker) of each cached state,
    /// most recently used first.
    pub fn cached_states(&self) -> Vec<usize> {
        self.states.iter().map(|e| e.ids.len()).collect()
    }

    /// The token ids that end a prompt's state (default [`STATE_MARKER`]),
    /// e.g. `tokenizer.encode("## Decision schema\n")`. Empties the cache.
    pub fn set_state_marker(&mut self, marker: &[u32]) -> Result<()> {
        ensure!(!marker.is_empty(), "an empty state marker");
        self.state_marker = marker.to_vec();
        self.forget_states();
        Ok(())
    }

    /// Full prefill from position 0 with zero state, committing after every
    /// layer and returning the residual stream after each of the 32 layers
    /// ([T, HIDDEN] each). Runs in chunks like any request, so T may exceed
    /// the chunk size. For parity work only.
    pub fn debug_layers(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        check_ids(ids)?;
        let t = ids.len();
        self.prefix_clean = false;
        self.kv_owner = None;
        self.stage_embeddings(ids);
        let mut out = vec![vec![0.0f32; t * HIDDEN]; N_LAYERS];
        for ch in self.chunks(0, t, false, &[]) {
            for (l, layer_out) in out.iter_mut().enumerate() {
                self.submit_run(|run| {
                    if l == 0 {
                        self.encode_embed(run.enc(), &ch);
                    }
                    self.encode_layer(run, l, &ch, Source::Zero, None);
                })?;
                read_into(&self.b.x, &mut layer_out[ch.pos0 * HIDDEN..(ch.pos0 + ch.rows) * HIDDEN]);
            }
        }
        Ok(out)
    }

    /// Layer `l` alone over a whole sequence: `x` is its input residual
    /// stream ([T, HIDDEN], e.g. the gold output of layer l − 1), run from
    /// position 0 with zero state; returns the layer's output. Isolates one
    /// layer's error from the drift of the layers before it. For parity work.
    pub fn debug_layer(&mut self, l: usize, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(l < N_LAYERS, "layer {l} of {N_LAYERS}");
        let t = x.len() / HIDDEN;
        ensure!(x.len() == t * HIDDEN && (1..=MAX_TOKENS).contains(&t), "x must be [T ≤ {MAX_TOKENS}, {HIDDEN}]");
        self.prefix_clean = false;
        self.kv_owner = None;
        let mut out = vec![0.0f32; x.len()];
        for ch in self.chunks(0, t, false, &[]) {
            let span = ch.pos0 * HIDDEN..(ch.pos0 + ch.rows) * HIDDEN;
            write(&self.b.x, 0, &x[span.clone()]);
            self.submit_run(|run| self.encode_layer(run, l, &ch, Source::Zero, None))?;
            read_into(&self.b.x, &mut out[span]);
        }
        Ok(out)
    }

    /// [`Self::logits`] by the full path whatever the ids: every row from
    /// position 0 to the last answer row, from zero state.
    pub fn logits_full<C: AsRef<[u32]>>(&mut self, ids: &[u32], answers: &[(usize, C)]) -> Result<Vec<Vec<f32>>> {
        check_ids(ids)?;
        let answers = sorted_answers(ids.len(), answers)?;
        self.run_answers(ids, &answers, Plan::Full)
    }

    /// Run positions `start..=last answer row` as `plan` says, and read the
    /// answers' logits.
    fn run_answers(&mut self, ids: &[u32], answers: &[Answer], plan: Plan) -> Result<Vec<Vec<f32>>> {
        // Tail truncation: the rows after the last answer never run.
        let end = answers.last().expect("at least one answer").row + 1;
        let (source, start, route) = match plan {
            Plan::Full => (Source::Zero, 0, Route::Full),
            Plan::Snapshot | Plan::Capture { .. } => (Source::Snapshot, self.prefix_len().expect("a snapshot"), Route::Snapshot),
            Plan::Hit(i) => (Source::State(i), self.states[i].ids.len(), Route::StateCache),
        };
        self.last_route = Some(route);
        self.last_start = Some(start);
        if start == 0 {
            // This run overwrites the prefix's K/V rows.
            self.prefix_clean = false;
        } else if !self.prefix_clean {
            self.restore_prefix();
        }
        // A new state's entry: new buffers, or the least recently used entry's.
        let capture = match plan {
            Plan::Capture { boundary } => {
                let slot = match self.free.pop() {
                    Some(slot) => Some(slot),
                    None if self.states.len() >= self.state_capacity => self.states.pop(),
                    None => None,
                };
                Some((self.state_entry(slot, &ids[..boundary], start), boundary))
            }
            _ => None,
        };
        // A follow-up needs its entry's K/V rows back in the caches, unless
        // no run has overwritten them since they were there.
        let restore = match plan {
            Plan::Hit(i) if self.kv_owner != Some(self.states[i].id) => Some(i),
            _ => None,
        };
        self.stage_embeddings(&ids[start..end]);
        let rows: Vec<usize> = answers.iter().map(|a| a.row).collect();
        let mut chunks = self.chunks(start, end, false, &rows);
        if let Some(&(_, boundary)) = capture.as_ref() {
            let ch = chunks.iter_mut().find(|c| c.pos0 < boundary && boundary <= c.pos0 + c.rows);
            let ch = ch.expect("the state ends inside the span");
            ch.capture = Some(boundary - ch.pos0);
        }
        self.stage_answers(&chunks, answers);
        let prefix = self.prefix_len().unwrap_or(0);
        let submitted = self.submit_run(|run| {
            if let Some(i) = restore {
                self.encode_state_rows(run.enc(), &self.states[i], prefix, false);
            }
            for ch in &chunks {
                let entry = capture.as_ref().filter(|_| ch.capture.is_some()).map(|c| &c.0);
                self.encode_chunk(run, ch, source, entry);
                if let Some(entry) = entry {
                    // Rows prefix..boundary of every K/V cache are final now.
                    self.encode_state_rows(run.enc(), entry, prefix, true);
                }
            }
        });
        // Whose K/V rows, from the prefix to the end of its state, the caches
        // now hold: a run writes every row from its start on.
        self.kv_owner = match plan {
            Plan::Full | Plan::Snapshot => None,
            Plan::Capture { .. } => capture.as_ref().map(|c| c.0.id),
            Plan::Hit(i) => Some(self.states[i].id),
        };
        if let Err(e) = submitted {
            // The caches' contents and the new entry are unknown.
            self.kv_owner = None;
            return Err(e);
        }
        match (plan, capture) {
            (Plan::Capture { .. }, Some((entry, _))) => self.states.insert(0, entry),
            (Plan::Hit(i), _) => {
                let entry = self.states.remove(i);
                self.states.insert(0, entry);
            }
            _ => {}
        }
        let all = read_vec::<f32>(&self.b.logits, answers.len() * MAX_CANDIDATES);
        let mut out = vec![Vec::new(); answers.len()];
        for (slot, a) in answers.iter().enumerate() {
            let row = &all[slot * MAX_CANDIDATES..slot * MAX_CANDIDATES + a.candidates.len()];
            ensure!(row.iter().all(|x| x.is_finite()), "non-finite logits for answer {} (row {})", a.index, a.row);
            out[a.index] = row.to_vec();
        }
        Ok(out)
    }

    /// Split `start..end` into chunks; `answer_rows` (sorted) assigns each
    /// chunk its answers. `keep_final_state` keeps the state after the last
    /// chunk too.
    fn chunks(&self, start: usize, end: usize, keep_final_state: bool, answer_rows: &[usize]) -> Vec<Chunk> {
        let mut out = Vec::new();
        let (mut pos0, mut a) = (start, 0);
        while pos0 < end {
            let rows = self.chunk.min(end - pos0);
            let last = pos0 + rows == end;
            let a0 = a;
            while a < answer_rows.len() && answer_rows[a] < pos0 + rows {
                a += 1;
            }
            out.push(Chunk {
                pos0,
                rows,
                stage_row: pos0 - start,
                first: pos0 == start,
                keep_state: !last || keep_final_state,
                capture: None,
                a0,
                n: a - a0,
            });
            pos0 += rows;
        }
        out
    }

    /// Copy the embedding rows of `ids` into `embed_stage` rows 0.. .
    fn stage_embeddings(&self, ids: &[u32]) {
        let dst = self.b.embed_stage.contents() as *mut u8;
        for (i, &id) in ids.iter().enumerate() {
            // SAFETY: ids are checked < VOCAB, so the source row lies inside
            // the embedding tensor, and i < MAX_TOKENS rows fit the stage.
            unsafe { std::ptr::copy_nonoverlapping(self.embed_row(id), dst.add(i * EMBED_ROW), EMBED_ROW) };
        }
    }

    /// The CPU address of embedding row `id` (< VOCAB) in the mapped blob.
    fn embed_row(&self, id: u32) -> *const u8 {
        // A no-copy buffer's contents are the mapping it was made over.
        let base = self.embed.buf.contents() as *const u8;
        // SAFETY: the tensor is [VOCAB, HIDDEN] bf16 at `off` (checked at
        // construction) and the caller keeps id < VOCAB.
        unsafe { base.add(self.embed.off as usize + id as usize * EMBED_ROW) }
    }

    /// Write the answers' rows (within their chunks), positions and candidate
    /// counts, and stage their candidates' embedding rows.
    fn stage_answers(&self, chunks: &[Chunk], answers: &[Answer]) {
        let b = &self.b;
        let mut rows = vec![0u32; answers.len()];
        for ch in chunks {
            for j in ch.a0..ch.a0 + ch.n {
                rows[j] = (answers[j].row - ch.pos0) as u32;
            }
        }
        let pos: Vec<u32> = answers.iter().map(|a| a.row as u32).collect();
        let ncand: Vec<u32> = answers.iter().map(|a| a.candidates.len() as u32).collect();
        write(&b.ans_rows, 0, &rows);
        write(&b.ans_pos, 0, &pos);
        write(&b.ncand, 0, &ncand);
        let dst = b.cand_stage.contents() as *mut u8;
        for (i, a) in answers.iter().enumerate() {
            for (c, &id) in a.candidates.iter().enumerate() {
                // SAFETY: candidates are checked < VOCAB and ≤ MAX_CANDIDATES
                // per answer, and answers ≤ MAX_ANSWERS.
                unsafe {
                    std::ptr::copy_nonoverlapping(self.embed_row(id), dst.add((i * MAX_CANDIDATES + c) * EMBED_ROW), EMBED_ROW)
                };
            }
        }
    }

    /// Encode a whole run with `f`, commit every command buffer it makes and
    /// wait for the last. With the ANE off that is one buffer, exactly as
    /// before; with it on, `f` splits the run at the hand-off points.
    fn submit_run(&self, f: impl FnOnce(&mut Run)) -> Result<()> {
        metal::objc::rc::autoreleasepool(|| {
            let queue = self.dev.command_queue();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let mut run = Run {
                queue,
                cmd,
                enc,
                event: self.ane.as_ref().map(|a| &*a.event),
                committed: Vec::new(),
                failed: None,
            };
            f(&mut run);
            run.enc.end_encoding();
            run.cmd.commit();
            run.committed.push(run.cmd);
            // Watchdog: a Core ML predict that never returns would leave the
            // GPU waiting on the event forever, and this thread with it. Wait
            // for the ANE's jobs with a deadline first; on a timeout, release
            // the GPU by moving the event to the last value any buffer waits
            // for (the run's result is discarded), then report the hang.
            if let Some(ane) = &self.ane {
                if let Err(e) = ane.session.finish(ANE_JOB_TIMEOUT_S) {
                    ane.event.set_signaled_value(ane.used.get());
                    run.fail(e.context(format!("the Neural Engine did not finish within {ANE_JOB_TIMEOUT_S} s")));
                }
            }
            run.cmd.wait_until_completed();
            let mut bad = None;
            for cmd in &run.committed {
                crate::gpu_time::record(cmd);
                if cmd.status() != MTLCommandBufferStatus::Completed && bad.is_none() {
                    bad = Some(cmd.status());
                }
            }
            // The ANE's jobs have all signalled (the last buffer waited on the
            // last of them), but their errors are collected here.
            if let Some(ane) = &self.ane {
                ane.session.take_error()?;
                let (want, got) = (ane.used.get(), ane.event.signaled_value());
                ensure!(got >= want, "the ANE event reached {got}, expected {want}: a hand-off was lost");
            }
            if let Some(e) = run.failed {
                return Err(e);
            }
            match bad {
                None => Ok(()),
                Some(status) => bail!("the GPU command buffer ended with status {status:?}"),
            }
        })
    }

    /// The chunk's embedding rows → `x`.
    fn encode_embed(&self, enc: &ComputeCommandEncoderRef, ch: &Chunk) {
        let b = &self.b;
        let table = (&b.embed_stage, (ch.stage_row * EMBED_ROW) as u64);
        self.k.embed.embed(enc, (&b.iota, 0), table, (&b.x, 0), ch.rows);
    }

    /// Put the snapshot's K/V rows back into rows `0..prefix` of the caches.
    fn restore_prefix(&mut self) {
        let s = self.snapshot.as_ref().expect("a snapshot");
        for (cache, rows) in self.b.kc.iter().zip(&s.k).chain(self.b.vc.iter().zip(&s.v)) {
            write(cache, 0, rows);
        }
        self.prefix_clean = true;
    }

    /// The source of chunk 0's (conv, recurrent) state for linear layer `i`.
    fn source(&self, source: Source, i: usize) -> (&Buffer, &Buffer) {
        match source {
            Source::Zero => (&self.b.zero_conv, &self.b.zero_state),
            Source::Snapshot => {
                let s = self.snapshot.as_ref().expect("a snapshot");
                (&s.conv[i], &s.state[i])
            }
            Source::State(e) => (&self.states[e].conv[i], &self.states[e].state[i]),
        }
    }

    /// An entry for the state that `ids` (through the state marker) ends
    /// with: `slot`'s buffers (an evicted entry), or new ones, with room
    /// for its K/V rows after the prefix. Not in the cache until its
    /// capture has run.
    fn state_entry(&mut self, slot: Option<StateEntry>, ids: &[u32], prefix: usize) -> StateEntry {
        let (id, rows) = (self.next_entry_id, ids.len() - prefix);
        self.next_entry_id += 1;
        let dev = self.dev;
        let split = self.ane.is_some();
        let kv = |rows: usize| {
            (0..self.b.kc.len())
                .map(|_| if split { tracked(dev, rows * KV_DIM * F16) } else { shared(dev, rows * KV_DIM * F16) })
                .collect::<Vec<_>>()
        };
        // Room in whole 128-row blocks, so an evicted entry's rows often fit the next state.
        let room = rows.next_multiple_of(128);
        match slot {
            Some(mut e) => {
                if e.kv_rows < rows {
                    (e.k, e.v, e.kv_rows) = (kv(room), kv(room), room);
                }
                StateEntry { id, ids: ids.to_vec(), ..e }
            }
            None => StateEntry {
                id,
                ids: ids.to_vec(),
                state: self.b.state.iter().map(|s| self.buffer(s.length() as usize)).collect(),
                conv: self.b.conv.iter().map(|c| self.buffer(c.length() as usize)).collect(),
                k: kv(room),
                v: kv(room),
                kv_rows: room,
            },
        }
    }

    /// Copy an entry's K/V rows, positions `prefix..entry.ids.len()`, from
    /// the caches into the entry (`to_entry`) or back.
    fn encode_state_rows(&self, enc: &ComputeCommandEncoderRef, entry: &StateEntry, prefix: usize, to_entry: bool) {
        let (k, b) = (&self.k, &self.b);
        let identity = (&b.iota, 0);
        let rows = entry.ids.len() - prefix;
        let at = (prefix * KV_DIM * F16) as u64;
        for (cache, rows_of) in b.kc.iter().zip(&entry.k).chain(b.vc.iter().zip(&entry.v)) {
            if to_entry {
                k.head.gather_rows_f16(enc, (cache, at), identity, (rows_of, 0), rows, KV_DIM);
            } else {
                k.head.gather_rows_f16(enc, (rows_of, 0), identity, (cache, at), rows, KV_DIM);
            }
        }
    }

    /// One chunk of a request (or of the snapshot build): embedding, the 32
    /// layers, and the head for the chunk's answers.
    /// With `capture`, the chunk's [`Chunk::capture`] row ends the request's
    /// state, and the linear layers write their state there into `capture`.
    fn encode_chunk(&self, run: &mut Run, ch: &Chunk, source: Source, capture: Option<&StateEntry>) {
        self.encode_embed(run.enc(), ch);
        for l in 0..N_LAYERS - 1 {
            self.encode_layer(run, l, ch, source, capture);
        }
        if self.prune {
            self.encode_last_layer_pruned(run, ch);
        } else {
            self.encode_layer(run, N_LAYERS - 1, ch, source, capture);
            if ch.n > 0 {
                self.encode_head(run.enc(), ch);
            }
        }
    }

    /// The last layer (full attention), pruned to the answer rows.
    /// Every row still needs its K and V, which the answer rows
    /// of this and later chunks attend to; nothing else of a non-answer row
    /// is ever read. So: norm and the k|v part of the input projection for
    /// all rows, into the cache; then the q|gate part, attention, the output
    /// projection, the MLP and the head for the chunk's answer rows only.
    fn encode_last_layer_pruned(&self, run: &mut Run, ch: &Chunk) {
        let enc = run.enc();
        let (k, b, w) = (&self.k, &self.b, &self.layers[N_LAYERS - 1]);
        let Mixer::Attention { a, q_norm, k_norm } = &w.mixer else {
            unreachable!("the last layer is full attention")
        };
        let (a, rows) = (*a, ch.rows);
        // Weight rows ATT_K.. of w_in are k|v: a byte offset into the tensor.
        let w_kv = (&w.w_in.buf, w.w_in.off + (ATT_K * HIDDEN * F16) as u64);
        k.norm.rmsnorm(enc, (&b.x, 0), w.in_norm.arg(), (&b.h, 0), (&b.iota, 0), rows, HIDDEN, HIDDEN, HIDDEN, true);
        k.gemm.gemm(enc, (&b.h, 0), w_kv, (&b.p, 0), rows, 2 * KV_DIM, HIDDEN, HIDDEN, HIDDEN, 2 * KV_DIM,
                    Epilogue::StoreF32);
        k.rope.kv_norm_rope(enc, (&b.p, 0), k_norm.arg(), (&b.inv_freq, 0), (&b.kc[a], 0), (&b.vc[a], 0), rows,
                            2 * KV_DIM, 0, KV_DIM, ch.pos0);
        if ch.n == 0 {
            return;
        }
        // The chunk's answers: rows `ans_rows` of this chunk, at absolute
        // positions `ans_pos`, into slots a0.. of the answer buffers.
        let (n, a0) = (ch.n, ch.a0);
        let rowmap = (&b.ans_rows, (a0 * U32) as u64);
        let qpos = (&b.ans_pos, (a0 * U32) as u64);
        let hs = (&b.hs, (a0 * HIDDEN * F16) as u64);
        let ps = (&b.ps, (a0 * ATT_K * F32) as u64);
        let xs = (&b.xs, (a0 * HIDDEN * F32) as u64);
        k.head.gather_rows_f16(enc, (&b.h, 0), rowmap, hs, n, HIDDEN);
        // Weight rows 0..ATT_K are q|gate; `ps` rows have stride ATT_K.
        k.gemm.gemm(enc, hs, w.w_in.arg(), ps, n, ATT_K, HIDDEN, HIDDEN, HIDDEN, ATT_K, Epilogue::StoreF32);
        k.rope.q_norm_rope(enc, ps, q_norm.arg(), (&b.inv_freq, 0), qpos, (&b.qo, 0), n, ATT_K);
        // The kernel reads the gate at column ATT_GATE of each `ps` row.
        k.attn.attention(enc, (&b.qo, 0), (&b.kc[a], 0), (&b.vc[a], 0), ps, qpos, (&b.g, 0), n, ATT_K);
        k.head.gather_rows_f32(enc, (&b.x, 0), rowmap, xs, n, HIDDEN);
        k.gemm.gemm(enc, (&b.g, 0), w.w_out.arg(), xs, n, HIDDEN, VALUE_DIM, VALUE_DIM, VALUE_DIM, HIDDEN,
                    Epilogue::AddF32);
        self.encode_mlp(run, N_LAYERS - 1, w, xs, hs, n);
        self.encode_final_head(run.enc(), ch);
    }

    /// Layer `l` over every row of the chunk: `x` += mixer(norm(x)), then the MLP.
    /// With `capture` (see [`Self::encode_chunk`]), a linear layer also
    /// writes its conv and recurrent state after row `ch.capture` into it.
    fn encode_layer(&self, run: &mut Run, l: usize, ch: &Chunk, source: Source,
                    capture: Option<&StateEntry>) {
        let enc = run.enc();
        let (k, b, w) = (&self.k, &self.b, &self.layers[l]);
        let rows = ch.rows;
        k.norm.rmsnorm(enc, (&b.x, 0), w.in_norm.arg(), (&b.h, 0), (&b.iota, 0), rows, HIDDEN, HIDDEN, HIDDEN, true);
        match &w.mixer {
            Mixer::Linear { i, conv_w, a_log, dt_bias, gnorm } => {
                let i = *i;
                k.gemm.gemm(enc, (&b.h, 0), w.w_in.arg(), (&b.p, 0), rows, LIN_IN_N, HIDDEN, HIDDEN, HIDDEN, LIN_IN_N,
                            Epilogue::StoreF32);
                let (conv_in, state_in) = if ch.first { self.source(source, i) } else { (&b.conv[i], &b.state[i]) };
                k.conv.conv_silu_l2(enc, (&b.p, 0), (conv_in, 0), conv_w.arg(), (&b.q, 0), (&b.k, 0), (&b.v, 0), rows,
                                    LIN_IN_N);
                let capture = capture.zip(ch.capture);
                if let Some((entry, t)) = capture {
                    // Before the in-place update below, which may overwrite `conv_in`.
                    k.conv.conv_state_update(enc, (&b.p, 0), (conv_in, 0), (&entry.conv[i], 0), t, LIN_IN_N);
                }
                if ch.keep_state {
                    k.conv.conv_state_update(enc, (&b.p, 0), (conv_in, 0), (&b.conv[i], 0), rows, LIN_IN_N);
                }
                // The recurrence over rows `from..to`, from `state_in` to `state_out`.
                let delta = |from: usize, to: usize, state_in: &Buffer, state_out: &Buffer, write: bool| {
                    let row = |width: usize, col: usize| (&b.p, ((from * width + col) * F32) as u64);
                    let at = |buf, width: usize| (buf, (from * width * F32) as u64);
                    k.delta.gated_delta(enc, at(&b.q, KEY_DIM), at(&b.k, KEY_DIM), at(&b.v, VALUE_DIM),
                                        row(LIN_IN_N, LIN_A), row(LIN_IN_N, LIN_B), a_log.arg(), dt_bias.arg(),
                                        (state_in, 0), (state_out, 0), at(&b.o, VALUE_DIM), to - from, LIN_IN_N,
                                        write);
                };
                match capture {
                    // Split at the end of the state: the state there goes to the entry,
                    // and the rest of the chunk continues from it (exactly: fp32 through memory).
                    Some((entry, t)) => {
                        delta(0, t, state_in, &entry.state[i], true);
                        if t < rows {
                            delta(t, rows, &entry.state[i], &b.state[i], ch.keep_state);
                        } else if ch.keep_state {
                            k.head.gather_rows_f32(enc, (&entry.state[i], 0), (&b.iota, 0), (&b.state[i], 0), NV, DV * DK);
                        }
                    }
                    None => delta(0, rows, state_in, &b.state[i], ch.keep_state),
                }
                k.gnorm.gated_rmsnorm_silu(enc, (&b.o, 0), (&b.p, (LIN_Z * F32) as u64), gnorm.arg(), (&b.g, 0), rows,
                                           LIN_IN_N);
            }
            Mixer::Attention { a, q_norm, k_norm } => {
                let a = *a;
                let qpos = (&b.iota, (ch.pos0 * U32) as u64);
                k.gemm.gemm(enc, (&b.h, 0), w.w_in.arg(), (&b.p, 0), rows, ATT_IN_N, HIDDEN, HIDDEN, HIDDEN, ATT_IN_N,
                            Epilogue::StoreF32);
                k.rope.q_norm_rope(enc, (&b.p, 0), q_norm.arg(), (&b.inv_freq, 0), qpos, (&b.qo, 0), rows, ATT_IN_N);
                k.rope.kv_norm_rope(enc, (&b.p, 0), k_norm.arg(), (&b.inv_freq, 0), (&b.kc[a], 0), (&b.vc[a], 0), rows,
                                    ATT_IN_N, ATT_K, ATT_V, ch.pos0);
                // The kernel reads the gate at column ATT_GATE of each row of `p`.
                k.attn.attention(enc, (&b.qo, 0), (&b.kc[a], 0), (&b.vc[a], 0), (&b.p, 0), qpos, (&b.g, 0), rows,
                                 ATT_IN_N);
            }
        }
        k.gemm.gemm(enc, (&b.g, 0), w.w_out.arg(), (&b.x, 0), rows, HIDDEN, VALUE_DIM, VALUE_DIM, VALUE_DIM, HIDDEN,
                    Epilogue::AddF32);
        self.encode_mlp(run, l, w, (&b.x, 0), (&b.h, 0), rows);
    }

    /// `x` += MLP(norm(x)) over `rows` rows of `x` (f32) with `h` (f16) as the
    /// normed rows.
    ///
    /// On a split layer the ANE takes intermediate dims
    /// `0..n_inter` and the GPU the rest: this signals once the norm has
    /// written `h`, queues the layer's ANE job, encodes the GPU's slice into
    /// the next command buffer, and folds `partial / out_scale` into `x` at
    /// the head of the buffer after that (which waits for the ANE). The
    /// split's `h` is always `b.h`: layer `N_LAYERS - 1`, the only one whose
    /// MLP runs on gathered answer rows, is never split.
    fn encode_mlp(&self, run: &mut Run, l: usize, w: &Layer, x: Arg, h: Arg, rows: usize) {
        let (k, b) = (&self.k, &self.b);
        k.norm.rmsnorm(run.enc(), x, w.post_norm.arg(), h, (&b.iota, 0), rows, HIDDEN, HIDDEN, HIDDEN, true);
        let Some(ane) = self.ane.as_ref().filter(|a| l < a.cfg.layers) else {
            let enc = run.enc();
            k.gemm.gemm(enc, h, w.w_gu.arg(), (&b.mlp, 0), rows, MLP_IN_N, HIDDEN, HIDDEN, HIDDEN, INTER,
                        Epilogue::SwigluF16);
            k.gemm.gemm(enc, (&b.mlp, 0), w.w_down.arg(), x, rows, HIDDEN, INTER, INTER, INTER, HIDDEN,
                        Epilogue::AddF32);
            return;
        };
        if run.failed.is_some() {
            return;
        }
        if !(std::ptr::eq(h.0, &b.h) && h.1 == 0) {
            run.fail(anyhow::anyhow!(
                "layer {l}: a split layer's MLP must read the whole-chunk `h` the ANE session is bound to"
            ));
            return;
        }
        let n_gpu = ane.n_gpu();
        let chunks = rows.div_ceil(ane.cfg.chunk_rows);
        let (ready, done) = ane.next_pair();
        // The ANE reads `h` between these two values; nothing writes `h` again
        // until the buffer that waits for `done`.
        // Registered before the buffer is committed, so the job is waiting on
        // the event before the GPU can possibly signal it.
        if let Err(e) = ane.session.enqueue(l, chunks, ready, done) {
            run.fail(e.context(format!("layer {l}: queueing the ANE job")));
            return;
        }
        run.split(Some(ready), None);
        if ane.sync {
            // The norm's command buffer must have completed before the ANE
            // reads `h`; waiting here proves the hand-off, at the cost of the
            // overlap this split exists for.
            run.wait_committed();
        }
        let enc = run.enc();
        k.gemm.gemm(enc, h, w.w_gu.arg(), (&b.mlp, 0), rows, 2 * n_gpu, HIDDEN, HIDDEN, HIDDEN, n_gpu,
                    Epilogue::SwigluF16);
        k.gemm.gemm(enc, (&b.mlp, 0), w.w_down.arg(), x, rows, HIDDEN, n_gpu, n_gpu, n_gpu, HIDDEN, Epilogue::AddF32);
        run.split(None, Some(done));
        k.ane.add_partial(run.enc(), x, (&ane.partial, 0), rows, HIDDEN, HIDDEN, HIDDEN, 1.0 / ane.cfg.out_scale);
    }

    /// Gather the chunk's answer rows of `x` into `xs`, then the final norm
    /// and candidate logits.
    fn encode_head(&self, enc: &ComputeCommandEncoderRef, ch: &Chunk) {
        let (k, b) = (&self.k, &self.b);
        let xs = (&b.xs, (ch.a0 * HIDDEN * F32) as u64);
        k.head.gather_rows_f32(enc, (&b.x, 0), (&b.ans_rows, (ch.a0 * U32) as u64), xs, ch.n, HIDDEN);
        self.encode_final_head(enc, ch);
    }

    /// Final norm + logits of the chunk's answers, from `xs`.
    fn encode_final_head(&self, enc: &ComputeCommandEncoderRef, ch: &Chunk) {
        let (k, b) = (&self.k, &self.b);
        let a0 = ch.a0;
        k.head.final_head(enc, (&b.xs, (a0 * HIDDEN * F32) as u64), self.final_norm.arg(), (&b.cand_stage, 0),
                          (&b.cand, (a0 * MAX_CANDIDATES * U32) as u64), (&b.ncand, (a0 * U32) as u64),
                          (&b.logits, (a0 * MAX_CANDIDATES * F32) as u64), ch.n);
    }
}

impl Buffers {
    /// `split` (the ANE is on) allocates every buffer hazard-TRACKED instead
    /// of untracked: a split run spans several command buffers, so the single
    /// serial encoder is no longer what orders every dependent access (see the
    /// module docs), and Metal has to insert the barriers itself.
    fn new(dev: &DecideDevice, cap: usize, split: bool) -> Self {
        let buf = |bytes: usize| if split { tracked(dev, bytes) } else { shared(dev, bytes) };
        let n_attn = (0..N_LAYERS).filter(|&l| is_full_attention(l)).count();
        let n_lin = N_LAYERS - n_attn;
        let cache = MAX_TOKENS * KV_DIM * F16;
        let state = NV * DV * DK * F32;
        let conv = 3 * CONV_DIM * F32;
        let iota = buf(MAX_TOKENS * U32);
        write(&iota, 0, &(0..MAX_TOKENS as u32).collect::<Vec<_>>());
        let freqs = buf(ROT_DIM / 2 * F32);
        write(&freqs, 0, &inv_freq());
        let cand = buf(MAX_ANSWERS * MAX_CANDIDATES * U32);
        write(&cand, 0, &(0..(MAX_ANSWERS * MAX_CANDIDATES) as u32).collect::<Vec<_>>());
        Self {
            iota,
            inv_freq: freqs,
            embed_stage: buf(MAX_TOKENS * EMBED_ROW),
            x: buf(cap * HIDDEN * F32),
            h: buf(cap * HIDDEN * F16),
            p: buf(cap * LIN_IN_N * F32),
            q: buf(cap * KEY_DIM * F32),
            k: buf(cap * KEY_DIM * F32),
            v: buf(cap * VALUE_DIM * F32),
            o: buf(cap * VALUE_DIM * F32),
            g: buf(cap * VALUE_DIM * F16),
            mlp: buf(cap * INTER * F16),
            qo: buf(cap * Q_DIM * F16),
            kc: (0..n_attn).map(|_| buf(cache)).collect(),
            vc: (0..n_attn).map(|_| buf(cache)).collect(),
            state: (0..n_lin).map(|_| buf(state)).collect(),
            conv: (0..n_lin).map(|_| buf(conv)).collect(),
            zero_state: zeroed(dev, state, split),
            zero_conv: zeroed(dev, conv, split),
            ans_rows: buf(MAX_ANSWERS * U32),
            ans_pos: buf(MAX_ANSWERS * U32),
            xs: buf(MAX_ANSWERS * HIDDEN * F32),
            hs: buf(MAX_ANSWERS * HIDDEN * F16),
            ps: buf(MAX_ANSWERS * ATT_K * F32),
            cand,
            ncand: buf(MAX_ANSWERS * U32),
            cand_stage: buf(MAX_ANSWERS * MAX_CANDIDATES * EMBED_ROW),
            logits: buf(MAX_ANSWERS * MAX_CANDIDATES * F32),
        }
    }
}

/// The manifest `reference/ane_convert.py` writes next to the models.
const ANE_MANIFEST: &str = "ane-models.json";
const ANE_MANIFEST_FORMAT: &str = "decide-ane-models-v1";

/// The models next to the weights must have been built for this weight
/// set's split: `ane-models.json` has to agree with the index's `"ane"` object
/// on every parameter and list every split layer. With `DECIDE_ANE_VERIFY=1`
/// each `.mlmodelc` is also hashed (sha256 over its files in sorted relative
/// path order, as the converter records it) — slow, for provenance checks.
fn check_ane_manifest(weights_dir: &Path, cfg: &AneConfig) -> Result<()> {
    let path = weights_dir.join(ANE_MANIFEST);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("{}: the ANE models' manifest (run `decide ane-convert`)", path.display()))?;
    let m: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let at = |what: &str| format!("{}: {what}", path.display());
    ensure!(m["format"].as_str() == Some(ANE_MANIFEST_FORMAT), "{}", at(&format!("format {} is not {ANE_MANIFEST_FORMAT}", m["format"])));
    let same_u = |key: &str, want: usize| -> Result<()> {
        let got = m[key].as_u64().with_context(|| at(&format!("missing {key}")))?;
        ensure!(got as usize == want, "{}", at(&format!("{key} {got} but the weight set was split with {want}")));
        Ok(())
    };
    same_u("n_inter", cfg.n_inter)?;
    same_u("chunk_rows", cfg.chunk_rows)?;
    same_u("layers", cfg.layers)?;
    let scale = m["out_scale"].as_f64().with_context(|| at("missing out_scale"))?;
    ensure!(scale as f32 == cfg.out_scale, "{}", at(&format!("out_scale {scale} but the weight set expects {}", cfg.out_scale)));
    let models = m["models"].as_object().with_context(|| at("missing models"))?;
    let verify = matches!(std::env::var("DECIDE_ANE_VERIFY").as_deref(), Ok("1"));
    for l in 0..cfg.layers {
        let entry = models.get(&format!("L{l}")).with_context(|| at(&format!("no model recorded for layer {l}")))?;
        let rel = entry["path"].as_str().with_context(|| at(&format!("L{l}: missing path")))?;
        let dir = weights_dir.join(rel);
        ensure!(dir.is_dir(), "{}: the manifest's model for layer {l} is missing", dir.display());
        if verify {
            let want = entry["sha256"].as_str().with_context(|| at(&format!("L{l}: missing sha256")))?;
            let got = sha256_dir(&dir)?;
            ensure!(got == want, "{}: sha256 {got} but the manifest records {want}", dir.display());
        }
    }
    Ok(())
}

/// sha256 over the files under `dir`, each file's bytes in sorted relative
/// path order (what `reference/ane_convert.py` computes).
fn sha256_dir(dir: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, std::path::PathBuf)>) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else {
                out.push((path.strip_prefix(root).unwrap().to_string_lossy().into_owned(), path));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files)?;
    files.sort();
    let mut h = Sha256::new();
    for (_, path) in files {
        h.update(std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Look up `name` and check its dtype and shape.
fn tensor(w: &Weights, name: &str, dtype: DType, shape: &[usize]) -> Result<W> {
    let t = w.get(name)?;
    ensure!(
        t.dtype == dtype && t.shape == shape,
        "{name} is {:?} {:?}; expected {dtype:?} {shape:?}",
        t.dtype,
        t.shape
    );
    Ok(W { buf: t.buf.clone(), off: t.offset })
}

fn resolve_layer(w: &Weights, l: usize) -> Result<Layer> {
    // A split layer's MLP weights hold only the GPU's share (the ANE's
    // `n_inter` intermediate dims were cut out by `decide ane-convert`).
    let n_gpu = match w.ane() {
        Some(cfg) if l < cfg.layers => INTER - cfg.n_inter,
        _ => INTER,
    };
    let f16 = |n: &str, shape: &[usize]| tensor(w, &format!("L{l}.{n}"), DType::F16, shape);
    let f32 = |n: &str, shape: &[usize]| tensor(w, &format!("L{l}.{n}"), DType::F32, shape);
    let (n_in, mixer) = if is_full_attention(l) {
        let a = (0..l).filter(|&j| is_full_attention(j)).count();
        (ATT_IN_N, Mixer::Attention { a, q_norm: f32("q_norm", &[HEAD_DIM])?, k_norm: f32("k_norm", &[HEAD_DIM])? })
    } else {
        let i = (0..l).filter(|&j| !is_full_attention(j)).count();
        (LIN_IN_N, Mixer::Linear {
            i,
            conv_w: f32("conv_w", &[CONV_DIM, CONV_K])?,
            a_log: f32("a_log", &[NV])?,
            dt_bias: f32("dt_bias", &[NV])?,
            gnorm: f32("gnorm", &[DV])?,
        })
    };
    Ok(Layer {
        in_norm: f32("in_norm", &[HIDDEN])?,
        post_norm: f32("post_norm", &[HIDDEN])?,
        w_in: f16("w_in", &[n_in, HIDDEN])?,
        w_out: f16("w_out", &[HIDDEN, VALUE_DIM])?,
        w_gu: f16("w_gu", &[2 * n_gpu, HIDDEN])?,
        w_down: f16("w_down", &[HIDDEN, n_gpu])?,
        mixer,
    })
}

fn check_ids(ids: &[u32]) -> Result<()> {
    ensure!(!ids.is_empty(), "no input tokens");
    ensure!(ids.len() <= MAX_TOKENS, "{} input tokens; decide takes at most {MAX_TOKENS}", ids.len());
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= VOCAB) {
        bail!("token id {bad} is outside the {VOCAB}-token vocabulary");
    }
    Ok(())
}

/// Where a prompt's state ends: one past the first `marker` in
/// `ids[prefix..limit]` (`limit`, the first answer row, bounds the search:
/// the state must end at or before it). `None` if there is none, e.g. when
/// the marker tokenized differently in this prompt. Correctness never rests
/// on this: a state entry is used only for ids that start with exactly the
/// ids it was captured from; the marker only picks where to capture.
pub fn state_boundary(ids: &[u32], prefix: usize, marker: &[u32], limit: usize) -> Option<usize> {
    let hay = ids.get(prefix..limit.min(ids.len()))?;
    hay.windows(marker.len()).position(|w| w == marker).map(|i| prefix + i + marker.len())
}

/// Validate the answers against an input of `len` tokens and sort them by row.
fn sorted_answers<C: AsRef<[u32]>>(len: usize, answers: &[(usize, C)]) -> Result<Vec<Answer<'_>>> {
    ensure!(
        (1..=MAX_ANSWERS).contains(&answers.len()),
        "{} answers; a request has 1 to {MAX_ANSWERS}",
        answers.len()
    );
    let mut out = Vec::with_capacity(answers.len());
    for (index, (row, candidates)) in answers.iter().enumerate() {
        let (row, candidates) = (*row, candidates.as_ref());
        ensure!(row < len, "answer {index}: logit row {row} is past the {len}-token input");
        ensure!(
            (1..=MAX_CANDIDATES).contains(&candidates.len()),
            "answer {index}: {} candidates; a field has 1 to {MAX_CANDIDATES}",
            candidates.len()
        );
        if let Some(&bad) = candidates.iter().find(|&&c| c as usize >= VOCAB) {
            bail!("answer {index}: candidate {bad} is outside the {VOCAB}-token vocabulary");
        }
        out.push(Answer { index, row, candidates });
    }
    out.sort_by_key(|a| (a.row, a.index));
    Ok(out)
}

/// A shared buffer Metal hazard-tracks: for memory the ANE also touches, where
/// the serial encoder is not what orders the accesses.
fn tracked(dev: &DecideDevice, bytes: usize) -> Buffer {
    dev.device().new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

/// A shared, untracked buffer (see the module docs on ordering).
fn shared(dev: &DecideDevice, bytes: usize) -> Buffer {
    dev.device().new_buffer(
        bytes.max(16) as u64,
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
    )
}

fn zeroed(dev: &DecideDevice, bytes: usize, tracked_buf: bool) -> Buffer {
    let buf = if tracked_buf { tracked(dev, bytes) } else { shared(dev, bytes) };
    // SAFETY: the buffer is `bytes` long (or more) and not yet in use.
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes) };
    buf
}

/// Write `data` at element offset `at` of a shared buffer the GPU is not using.
fn write<T: Copy>(buf: &Buffer, at: usize, data: &[T]) {
    let size = std::mem::size_of::<T>();
    assert!((at + data.len()) * size <= buf.length() as usize, "write past the buffer end");
    // SAFETY: bounds checked above; the GPU is idle between requests.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (buf.contents() as *mut T).add(at), data.len()) };
}

/// The first `n` elements of a shared buffer the GPU is done with.
fn read_vec<T: Copy + Default>(buf: &Buffer, n: usize) -> Vec<T> {
    let mut out = vec![T::default(); n];
    read_into(buf, &mut out);
    out
}

fn read_into<T: Copy>(buf: &Buffer, out: &mut [T]) {
    assert!(std::mem::size_of_val(out) <= buf.length() as usize, "read past the buffer end");
    // SAFETY: bounds checked above; the command buffer has completed.
    unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const T, out.as_mut_ptr(), out.len()) };
}
