//! The public engine: a request's JSON in, InternLM's response out.
//!
//! [`Engine::open`] loads the tokenizer and the weights, builds every GPU
//! pipeline (`Forward::new` also builds every fast-GEMM pipeline the shape
//! table can pick), runs the constant prompt prefix into the prefix snapshot,
//! then runs one throwaway request through the production path and checks
//! that it built no pipeline. After that no pipeline is compiled inside a
//! request, and the layer weights (prefaulted and pinned at load) and every
//! buffer have been used by the GPU once.
//!
//! The embedding table is the exception. Its blob is mapped lazily and only
//! the CPU reads it, when it stages a request's embedding and candidate rows,
//! so a request with token ids that no earlier request used can take page
//! faults on those rows (from the page cache, or from disk after a cold
//! start) inside its forward time. A repeated request does not.
//!
//! [`Engine::decide`] is InternLM's `DecisionEngine.predict`: prepare the
//! request (validate, compile, render, tokenize), one forward pass from the
//! prefix snapshot to the last answer row, then the head (softmax,
//! calibration, the Jev envelope). As in the reference, `timing.inference_ms`
//! is the forward pass's wall time and `usage.input_tokens` the full prompt
//! length, although the forward pass stops at the last answer row.

use crate::config::MAX_TOKENS;
use crate::forward::{Forward, Route};
use crate::gpu_time;
use crate::head::{self, FieldHead, HeadError};
use anyhow::{anyhow, ensure, Context};
use decide_prompt::tokenize::{PrepareError, Prepared, Tokenizer};
use serde_json::Value;
use std::path::Path;
use std::time::{Duration, Instant};

/// The tokenizer inside the weights directory (`decide convert` copies it there).
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The throwaway request [`Engine::open`] runs: short, and through every
/// stage a real request takes (the snapshot route, the pruned last layer
/// with its answer rows, the head).
const WARM_UP_REQUEST: &str = r#"{"state": "The engine has just loaded its weights.", "questions": {"ready": {"type": "noul", "instructions": "Is the engine ready?"}, "speed": {"type": "score", "instructions": "Rate the load time.", "criteria": ["Slow", "Fast"]}}}"#;

/// Why [`Engine::decide`] refused or failed a request. `Display` is the bare
/// message.
#[derive(Debug)]
pub enum EngineError {
    /// The request was refused before the forward pass. Where the reference
    /// refuses it too, the message is the reference's (`validate_request`,
    /// `compile_row` and `encode` raise `ValueError`s); otherwise it says why
    /// decide refuses it (text-only, not valid JSON).
    Request(String),
    /// The head refused the logits where the reference's post-processing
    /// raises, with the reference's message.
    Head(HeadError),
    /// The forward pass failed: a GPU or internal error.
    Forward(anyhow::Error),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Request(msg) => f.write_str(msg),
            EngineError::Head(e) => write!(f, "{e}"),
            EngineError::Forward(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<PrepareError> for EngineError {
    fn from(e: PrepareError) -> Self {
        EngineError::Request(e.to_string())
    }
}

/// Where one request's time went, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Parse, validate, compile, render and tokenize.
    pub tokenize_ms: f64,
    /// The forward pass, token ids ready → logits read back: the response's
    /// `timing.inference_ms` (before rounding).
    pub forward_ms: f64,
    /// The GPU's own time for the forward pass's command buffers
    /// (`GPUEndTime − GPUStartTime`), when Metal reports it.
    pub gpu_ms: Option<f64>,
    /// Softmax, calibration and the response value.
    pub head_ms: f64,
    /// The whole call: request text in → response value out.
    pub total_ms: f64,
}

/// Where [`Engine::open`]'s time went, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoadTimings {
    /// The tokenizer, and the prefix ids.
    pub tokenizer_ms: f64,
    /// `Forward::new`: the weights mapped, prefaulted and pinned; buffers;
    /// every pipeline built.
    pub weights_ms: f64,
    /// The prefix snapshot.
    pub snapshot_ms: f64,
    /// The throwaway request.
    pub warm_up_ms: f64,
    pub total_ms: f64,
}

/// One request, with everything computed on the way to its response.
#[derive(Debug, Clone)]
pub struct Decision {
    /// InternLM's response.
    pub response: Value,
    /// The prompt: ids, logit rows, candidate ids, the compiled fields.
    pub prepared: Prepared,
    /// Each field's candidate logits, in option order.
    pub logits: Vec<Vec<f32>>,
    /// Each field's calibrated probabilities and decision.
    pub heads: Vec<FieldHead>,
    /// How the forward pass ran.
    pub route: Route,
    /// Positions the forward pass computed: from the end of the prefix (from
    /// 0 on the full route, from the end of the state on a follow-up) to the
    /// last answer row.
    pub rows: usize,
    pub timings: Timings,
}

pub struct Engine {
    tokenizer: Tokenizer,
    forward: Forward,
    load: LoadTimings,
    last: Option<Timings>,
    /// The directory the weights (and, with the ANE on, the models) came from.
    weights_dir: std::path::PathBuf,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

impl Engine {
    /// Load `weights_dir` (from `decide convert`: blobs, index and
    /// `tokenizer.json`), warm every GPU pipeline up and build the prefix
    /// snapshot. Holds about 7 GB of wired memory from here on.
    pub fn open(weights_dir: &Path) -> anyhow::Result<Self> {
        let started = Instant::now();
        let path = weights_dir.join(TOKENIZER_FILE);
        let tokenizer = Tokenizer::from_file(&path).with_context(|| format!("loading {}", path.display()))?;
        let prefix = tokenizer.prefix_ids()?;
        let tokenizer_done = Instant::now();
        let mut forward = Forward::new(weights_dir)?;
        let weights_done = Instant::now();
        forward.build_snapshot(&prefix)?;
        let snapshot_done = Instant::now();
        let pipelines = forward.gemm_pipelines();
        let mut engine = Engine {
            tokenizer,
            forward,
            load: LoadTimings::default(),
            last: None,
            weights_dir: weights_dir.to_path_buf(),
        };
        engine.decide(WARM_UP_REQUEST).map_err(|e| anyhow!("the warm-up request failed: {e}"))?;
        ensure!(
            engine.forward.last_route() == Some(Route::Snapshot),
            "the warm-up request missed the prefix snapshot"
        );
        ensure!(
            engine.forward.gemm_pipelines() == pipelines,
            "the warm-up request built {} GEMM pipelines that the GEMM warm-up missed",
            engine.forward.gemm_pipelines() - pipelines
        );
        // The warm-up's state is nobody's; its buffers stay for the first capture.
        engine.forward.forget_states();
        let done = Instant::now();
        engine.last = None;
        engine.load = LoadTimings {
            tokenizer_ms: ms(tokenizer_done - started),
            weights_ms: ms(weights_done - tokenizer_done),
            snapshot_ms: ms(snapshot_done - weights_done),
            warm_up_ms: ms(done - snapshot_done),
            total_ms: ms(done - started),
        };
        Ok(engine)
    }

    /// The weights directory this engine was opened on (provenance for
    /// benchmark reports: `weights` vs `weights/ane`).
    pub fn weights_dir(&self) -> &Path {
        &self.weights_dir
    }

    /// Where [`Engine::open`]'s time went.
    pub fn load_timings(&self) -> LoadTimings {
        self.load
    }

    /// The timings of the last request that reached a response.
    pub fn last_timings(&self) -> Option<Timings> {
        self.last
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The forward pass: its kernels, prefix length and last route.
    pub fn forward(&self) -> &Forward {
        &self.forward
    }

    /// The forward pass, to change its settings (e.g. the state cache).
    pub fn forward_mut(&mut self) -> &mut Forward {
        &mut self.forward
    }

    /// Validate, compile, render and tokenize a request, refusing prompts
    /// longer than the reference's `max_length` (8192 tokens).
    pub fn prepare(&self, request_json: &str) -> Result<Prepared, EngineError> {
        Ok(self.tokenizer.prepare(request_json, MAX_TOKENS)?)
    }

    /// Each field's candidate logits, in option order: one forward pass over
    /// the prompt's ids with the answers at `prepared.positions` and
    /// candidates `prepared.candidates`.
    pub fn logits(&mut self, prepared: &Prepared) -> anyhow::Result<Vec<Vec<f32>>> {
        let answers: Vec<(usize, &[u32])> =
            prepared.positions.iter().copied().zip(prepared.candidates.iter().map(Vec::as_slice)).collect();
        self.forward.logits(&prepared.ids, &answers)
    }

    /// InternLM's `DecisionEngine.predict`: the request's JSON text in, the
    /// response out.
    pub fn decide(&mut self, request_json: &str) -> Result<Value, EngineError> {
        Ok(self.decide_detailed(request_json)?.response)
    }

    /// [`Engine::decide`], also returning the prompt, the logits, the
    /// calibrated probabilities, the route and the timings.
    pub fn decide_detailed(&mut self, request_json: &str) -> Result<Decision, EngineError> {
        let started = Instant::now();
        let prepared = self.prepare(request_json)?;
        let prepared_at = Instant::now();
        gpu_time::take(); // start this request's GPU total from zero
        let logits = self.logits(&prepared).map_err(EngineError::Forward)?;
        let forward_done = Instant::now();
        let gpu_ms = gpu_time::take();
        let forward_ms = ms(forward_done - prepared_at);
        let heads = head::fields(&prepared, &logits).map_err(EngineError::Head)?;
        let response =
            head::respond_from(&prepared, &heads, prepared.ids.len(), forward_ms).map_err(EngineError::Head)?;
        let done = Instant::now();
        let route = self.forward.last_route().expect("the forward pass sets its route");
        let end = prepared.positions.iter().max().expect("a prepared request has fields") + 1;
        let rows = end - self.forward.last_start().expect("the forward pass sets its start");
        let timings = Timings {
            tokenize_ms: ms(prepared_at - started),
            forward_ms,
            gpu_ms,
            head_ms: ms(done - forward_done),
            total_ms: ms(done - started),
        };
        self.last = Some(timings);
        Ok(Decision { response, prepared, logits, heads, route, rows, timings })
    }
}
