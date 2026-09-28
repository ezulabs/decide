//! `decide bench`: warm single-request latency, per request.
//!
//! One engine, loaded once (its load time reported apart). Per request:
//! `warmup` untimed runs (at least one, which records the request's tokens,
//! route and decisions), then `iters` timed ones, each through
//! [`Engine::decide`] exactly as `serve` runs it. Three times per run,
//! whose exact spans are [`SPANS`] (printed in the header and saved in the
//! report, so that the baseline scripts can time the same spans):
//! **forward** (the number compared against the baselines), **gpu** (inside
//! forward) and **e2e** (JSON text in → JSON text out).
//!
//! It measures the steady state of a repeated request: see
//! [`STEADY_STATE`].
//!
//! **Cold and follow-up runs.** The engine caches the state at the end of a
//! prompt's state section, so asking about a state seen before runs only
//! the question tokens. Every run is cold by default: the
//! state cache is emptied (outside the timed region) before each run, so
//! each one pays for its state, as a new request does, and saves it.
//! `--follow-ups` also times each request as a follow-up: its state cached
//! by an untimed run first, as when a second question set about the same
//! state follows the first.
//!
//! p50/p95 by linear interpolation between the closest ranks (numpy's
//! default). The machine state is recorded before and after: power source,
//! low-power mode, load average per core (flagged above 1), free memory,
//! swap and the busiest processes. Timing runs must hold the GPU lock (see
//! the plan's resource rules).

use anyhow::{ensure, Context, Result};
use decide_engine::engine::{Engine, Timings};
use decide_engine::forward::Route;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// What each timed span includes, verbatim in the header and in the
/// report's `protocol.spans`.
pub const SPANS: [(&str, &str); 4] = [
    (
        "forward",
        "token ids in memory -> logits on the CPU: the wall time of Engine::logits (Forward::logits): route choice \
         (prefix snapshot / state-cache lookup), staging of the span's embedding rows and the answers' candidate rows \
         (CPU memcpy from the mmapped bf16 table), any K/V row restore, encoding, commit, wait for completion, and \
         readback of the logits. Excludes tokenization and the head. Equals the response's timing.inference_ms \
         before rounding.",
    ),
    (
        "gpu",
        "GPUEndTime - GPUStartTime of the forward pass's command buffers, summed (one per request with the ANE off; \
         two per split layer with it on): GPU execution only, without the CPU staging, encoding and readback. Lies \
         inside forward. With the ANE on it includes the time a buffer spends stalled at its head waiting for the \
         ANE's event, so it is not pure GPU compute there.",
    ),
    (
        "e2e",
        "request JSON text in memory -> response JSON string: parse, validate, compile, render and tokenize, then \
         forward, then the head (softmax, calibration, response value) and serde_json::to_string of the response. \
         Excludes reading the request file and writing stdout.",
    ),
    ("compared", "forward p50 is the number compared against the MLX and PyTorch baselines."),
];

/// What the runs are the steady state of, verbatim in the header and in
/// the report's `protocol.steady_state`.
pub const STEADY_STATE: &str = "the steady state of a repeated request: each request runs warm-up + timed times \
     in a row, so the embedding pages of its token ids are resident after its first run. A request with token ids \
     no earlier request used can take page faults on those embedding rows inside forward (the embedding blob is \
     mapped lazily and read by the CPU during staging); this is not measured here. Cold runs empty the state cache \
     only.";

/// Every request file: a directory's `*.json` (by name) or the file itself.
fn request_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("reading {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"))
                .collect();
            files.sort();
            ensure!(!files.is_empty(), "no *.json requests in {}", path.display());
            out.extend(files);
        } else {
            ensure!(path.is_file(), "no request file {}", path.display());
            out.push(path.clone());
        }
    }
    Ok(out)
}

/// `<parent dir>/<stem>`, e.g. `core/card`.
pub(crate) fn request_name(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match path.parent().and_then(Path::file_name) {
        Some(dir) => format!("{}/{stem}", dir.to_string_lossy()),
        None => stem,
    }
}

/// The `p`-quantile (0..=1) of sorted samples, interpolating linearly
/// between the closest ranks.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = p * (sorted.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}

/// A sample's summary.
#[derive(Clone, Copy)]
pub(crate) struct Stats {
    pub p50: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl Stats {
    pub fn of(samples: &[f64]) -> Stats {
        let mut v = samples.to_vec();
        v.sort_by(f64::total_cmp);
        Stats {
            p50: quantile(&v, 0.5),
            p95: quantile(&v, 0.95),
            min: v.first().copied().unwrap_or(f64::NAN),
            max: v.last().copied().unwrap_or(f64::NAN),
            mean: v.iter().sum::<f64>() / v.len().max(1) as f64,
        }
    }

    pub fn json(&self, samples: &[f64]) -> Value {
        let r = |x: f64| (x * 1e3).round() / 1e3;
        json!({
            "p50": r(self.p50), "p95": r(self.p95), "min": r(self.min), "max": r(self.max), "mean": r(self.mean),
            "samples": samples.iter().map(|&x| r(x)).collect::<Vec<_>>(),
        })
    }
}

/// One request's results.
pub(crate) struct Measured {
    pub name: String,
    /// Timed as a follow-up (its state cached), not cold.
    pub follow_up: bool,
    pub path: PathBuf,
    pub tokens: usize,
    pub rows: usize,
    pub route: Route,
    /// {field: decision}.
    pub decisions: Value,
    /// The first (untimed) run, as a sample of what every run computes:
    /// its prompt ids, candidate logits and response.
    pub ids: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
    pub response: Value,
    pub forward: Vec<f64>,
    pub gpu: Vec<f64>,
    pub e2e: Vec<f64>,
    pub tokenize: Vec<f64>,
    pub head: Vec<f64>,
    /// Per timed run, with the ANE on: predicts it ran and the milliseconds
    /// inside them. A run that shows 0 predicts with the ANE on, or predict
    /// times far above the ANE's, is Core ML falling back to the CPU.
    pub ane_predicts: Vec<u64>,
    pub ane_ms: Vec<f64>,
}

impl Measured {
    /// `{"predicts": …, "predict_ms": …}` over the timed runs, or null with the ANE off.
    pub fn ane_stats_json(&self) -> Value {
        if self.ane_predicts.is_empty() {
            return Value::Null;
        }
        let predicts: Vec<f64> = self.ane_predicts.iter().map(|&n| n as f64).collect();
        json!({
            "predicts": Stats::of(&predicts).json(&predicts),
            "predict_ms": Stats::of(&self.ane_ms).json(&self.ane_ms),
        })
    }
}

/// `{n_inter, chunk_rows, out_scale, layers, on: true}` or null: the engine's ANE split.
pub(crate) fn ane_json(engine: &Engine) -> Value {
    match engine.forward().ane() {
        Some(a) => json!({ "on": true, "n_inter": a.n_inter, "chunk_rows": a.chunk_rows, "out_scale": a.out_scale, "layers": a.layers }),
        None => Value::Null,
    }
}

pub(crate) fn route_name(route: Route) -> &'static str {
    match route {
        Route::Snapshot => "snapshot",
        Route::Full => "full",
        Route::StateCache => "cache",
    }
}

fn kind_name(follow_up: bool) -> &'static str {
    if follow_up { "follow-up" } else { "cold" }
}

fn measure(engine: &mut Engine, path: &Path, warmup: usize, iters: usize, follow_up: bool) -> Result<Measured> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let primer = follow_up.then_some(text.as_str());
    measure_text(engine, path, &text, warmup, iters, primer)
}

/// Time `text` (the request at `path`): cold (`primer` None: the state cache
/// is emptied before every run, outside the timed region) or as a follow-up
/// (`primer` Some: that request runs once, untimed, to cache its state, and
/// every run of `text` finds it).
pub(crate) fn measure_text(engine: &mut Engine, path: &Path, text: &str, warmup: usize, iters: usize,
                           primer: Option<&str>) -> Result<Measured> {
    let name = request_name(path);
    let follow_up = primer.is_some();
    let reset = |engine: &mut Engine| {
        if !follow_up {
            engine.forward_mut().forget_states();
        }
    };
    if let Some(primer) = primer {
        engine.decide(primer).map_err(|e| anyhow::anyhow!("{name}: the request that caches the state: {e}"))?;
    }
    // The first warm-up run also records what the request is.
    reset(engine);
    let first = engine.decide_detailed(text).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    let decisions: serde_json::Map<String, Value> = first.response["answers"]
        .as_object()
        .map(|a| a.iter().map(|(field, answer)| (field.clone(), answer["decision"].clone())).collect())
        .unwrap_or_default();
    let mut m = Measured {
        name: name.clone(),
        follow_up,
        path: path.to_path_buf(),
        tokens: first.prepared.ids.len(),
        rows: first.rows,
        route: first.route,
        decisions: Value::Object(decisions),
        ids: first.prepared.ids.clone(),
        logits: first.logits.clone(),
        response: first.response.clone(),
        forward: Vec::with_capacity(iters),
        gpu: Vec::with_capacity(iters),
        e2e: Vec::with_capacity(iters),
        tokenize: Vec::with_capacity(iters),
        head: Vec::with_capacity(iters),
        ane_predicts: Vec::new(),
        ane_ms: Vec::new(),
    };
    for _ in 1..warmup {
        reset(engine);
        engine.decide(text).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    }
    for _ in 0..iters {
        reset(engine);
        let ane_before = engine.forward().ane_stats();
        let started = Instant::now();
        let response = engine.decide(text).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let out = serde_json::to_string(&response)?;
        let e2e = started.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(out);
        if let (Some((n0, s0)), Some((n1, s1))) = (ane_before, engine.forward().ane_stats()) {
            m.ane_predicts.push(n1 - n0);
            m.ane_ms.push((s1 - s0) * 1e3);
        }
        let t: Timings = engine.last_timings().context("no timings after a request")?;
        ensure!(engine.forward().last_route() == Some(m.route), "{name}: the route changed between runs");
        m.e2e.push(e2e);
        m.forward.push(t.forward_ms);
        if let Some(gpu) = t.gpu_ms {
            m.gpu.push(gpu);
        }
        m.tokenize.push(t.tokenize_ms);
        m.head.push(t.head_ms);
    }
    Ok(m)
}

/// Standard output of a command, or `None` if it did not run.
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}

/// The 1, 5 and 15 minute load averages.
fn load_average() -> Option<[f64; 3]> {
    let mut load = [0.0f64; 3];
    // SAFETY: `load` holds the 3 doubles asked for.
    (unsafe { getloadavg(load.as_mut_ptr(), 3) } == 3).then_some(load)
}

/// What the machine is doing, as far as the timings are concerned.
pub(crate) fn machine_state() -> Value {
    let power = command_output("pmset", &["-g", "batt"])
        .and_then(|s| s.split('\'').nth(1).map(str::to_string))
        .unwrap_or_else(|| "unknown".into());
    let low_power = command_output("pmset", &["-g"]).and_then(|s| {
        s.lines()
            .find(|l| l.trim_start().starts_with("lowpowermode"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u32>().ok())
    });
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let load = load_average();
    let per_core = load.map(|l| l[0] / cores as f64);
    let memory_free = command_output("memory_pressure", &[]).and_then(|s| {
        let line = s.lines().rev().find(|l| l.contains("free percentage"))?.to_string();
        line.rsplit(':').next()?.trim().trim_end_matches('%').parse::<u32>().ok()
    });
    // `total = 5120.00M  used = 4185.75M  free = 934.25M  (encrypted)`
    let swap_used_mb = command_output("sysctl", &["-n", "vm.swapusage"]).and_then(|s| {
        let used = s.split("used = ").nth(1)?.split_whitespace().next()?;
        used.trim_end_matches('M').parse::<f64>().ok().map(f64::round)
    });
    let top: Vec<Value> = command_output("ps", &["-Ao", "pcpu=,comm=", "-r"])
        .map(|s| {
            s.lines()
                .take(5)
                .filter_map(|l| {
                    let (cpu, comm) = l.trim().split_once(' ')?;
                    let comm = comm.trim();
                    let name = comm.rsplit('/').next().unwrap_or(comm);
                    Some(json!({ "cpu_percent": cpu.parse::<f64>().ok()?, "process": name }))
                })
                .collect()
        })
        .unwrap_or_default();
    json!({
        "power_source": power,
        "low_power_mode": low_power,
        "load_average": load,
        "cores": cores,
        "load_per_core": per_core.map(|x| (x * 100.0).round() / 100.0),
        "load_flag": per_core.is_some_and(|x| x > 1.0),
        "memory_free_percent": memory_free,
        "swap_used_mb": swap_used_mb,
        "top_processes": top,
    })
}

pub(crate) fn describe_machine(m: &Value) -> String {
    let load = m["load_average"]
        .as_array()
        .map(|l| l.iter().map(|x| format!("{:.2}", x.as_f64().unwrap_or(f64::NAN))).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "unknown".into());
    let top = m["top_processes"]
        .as_array()
        .map(|t| {
            t.iter()
                .map(|p| format!("{} {:.0}%", p["process"].as_str().unwrap_or("?"), p["cpu_percent"].as_f64().unwrap_or(0.0)))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!(
        "{}, low-power mode {}, load {load} on {} cores = {:.2}/core{}, memory free {}%, swap used {} MB\n  busiest: {top}",
        m["power_source"].as_str().unwrap_or("unknown"),
        match m["low_power_mode"].as_u64() {
            Some(0) => "off".to_string(),
            Some(_) => "ON".to_string(),
            None => "unknown".to_string(),
        },
        m["cores"],
        m["load_per_core"].as_f64().unwrap_or(f64::NAN),
        if m["load_flag"].as_bool() == Some(true) { " [FLAG: above 1/core]" } else { "" },
        m["memory_free_percent"].as_u64().map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
        m["swap_used_mb"].as_f64().map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
    )
}

/// Run the benchmark and print (and optionally save) the report.
pub fn bench(weights: &Path, requests: &[PathBuf], warmup: usize, iters: usize, follow_ups: bool,
             json_out: Option<&Path>) -> Result<()> {
    ensure!(iters >= 1, "--iters must be at least 1");
    let files = request_files(requests)?;
    let before = machine_state();
    let date = command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).map(|s| s.trim().to_string());
    let mut engine = crate::open(weights)?;
    let pipelines = engine.forward().gemm_pipelines();
    println!("decide bench: {}", crate::describe(&engine));
    println!("machine before: {}", describe_machine(&before));
    println!("{warmup} warm-up + {iters} timed runs per request; p50/p95 by linear interpolation; times in ms");
    println!("spans:");
    for (name, text) in SPANS {
        println!("  {name}: {text}");
    }
    println!("steady state: {STEADY_STATE}");
    println!("cold runs start with an empty state cache{}",
             if follow_ups { "; follow-up runs with the request's state cached" } else { "" });
    let mut results = Vec::new();
    for path in &files {
        results.push(measure(&mut engine, path, warmup, iters, false)?);
        if follow_ups {
            results.push(measure(&mut engine, path, warmup, iters, true)?);
        }
    }
    let fwd = engine.forward();
    let load = engine.load_timings();
    let pipelines_built = fwd.gemm_pipelines() - pipelines;
    let ane = ane_json(&engine);
    let weights_dir = engine.weights_dir().display().to_string();
    let engine_info = json!({
        "weights": weights_dir,
        "ane": ane,
        "DECIDE_ANE": std::env::var("DECIDE_ANE").ok(),
        "gemm": format!("{:?}", fwd.gemm()).to_lowercase(),
        "delta": format!("{:?}", fwd.delta()).to_lowercase(),
        "DECIDE_GEMM": std::env::var("DECIDE_GEMM").ok(),
        "DECIDE_DELTA": std::env::var("DECIDE_DELTA").ok(),
        "prefix_tokens": fwd.prefix_len(),
        "chunk_rows": fwd.chunk(),
        "last_layer_pruning": fwd.pruning(),
        "state_cache_entries": fwd.state_cache_capacity(),
        "gemm_pipelines_built_during_runs": pipelines_built,
    });
    // Unloaded first, so that the state after is the machine's, not ours.
    drop(engine);
    let after = machine_state();
    println!();
    println!(
        "{:<24} {:<9} {:>6} {:>5}  {:<8}  {:>17}  {:>17}  {:>17}  {:>8}  {:>8}",
        "request", "run", "tokens", "rows", "route", "forward p50/p95", "GPU p50/p95", "e2e p50/p95", "tokenize", "head"
    );
    for m in &results {
        let (f, g, e) = (Stats::of(&m.forward), Stats::of(&m.gpu), Stats::of(&m.e2e));
        println!(
            "{:<24} {:<9} {:>6} {:>5}  {:<8}  {:>8.1}/{:<8.1}  {:>8.1}/{:<8.1}  {:>8.1}/{:<8.1}  {:>8.2}  {:>8.2}",
            m.name,
            kind_name(m.follow_up),
            m.tokens,
            m.rows,
            route_name(m.route),
            f.p50,
            f.p95,
            g.p50,
            g.p95,
            e.p50,
            e.p95,
            Stats::of(&m.tokenize).p50,
            Stats::of(&m.head).p50,
        );
    }
    if results.iter().any(|m| !m.ane_predicts.is_empty()) {
        println!();
        println!("Neural Engine per run (a run with 0 predicts, or predict ms far above the ANE's, is Core ML on the CPU):");
        for m in results.iter().filter(|m| !m.ane_predicts.is_empty()) {
            let predicts: Vec<f64> = m.ane_predicts.iter().map(|&n| n as f64).collect();
            println!(
                "  {:<24} {:<9} predicts p50 {:>5.0}  predict ms p50/p95 {:>7.1}/{:<7.1}",
                m.name, kind_name(m.follow_up), Stats::of(&predicts).p50, Stats::of(&m.ane_ms).p50, Stats::of(&m.ane_ms).p95
            );
        }
    }
    println!();
    println!("machine after (engine unloaded): {}", describe_machine(&after));
    if before["load_flag"].as_bool() == Some(true) || after["load_flag"].as_bool() == Some(true) {
        println!("FLAG: load above 1 per core; these timings are not protocol numbers");
    }
    if before["power_source"] != "AC Power" || before["low_power_mode"].as_u64() != Some(0) {
        println!("FLAG: not on AC power with low-power mode off; these timings are not protocol numbers");
    }
    if pipelines_built > 0 {
        println!("FLAG: {pipelines_built} GEMM pipelines were built during the runs (the warm-up missed them)");
    }

    if let Some(out) = json_out {
        let report = json!({
            "date": date,
            "engine": engine_info,
            "load_ms": {
                "total": load.total_ms, "tokenizer": load.tokenizer_ms, "weights_and_pipelines": load.weights_ms,
                "prefix_snapshot": load.snapshot_ms, "warm_up_request": load.warm_up_ms,
            },
            "protocol": {
                "warmup": warmup,
                "iters": iters,
                "percentiles": "linear interpolation between closest ranks",
                "spans": SPANS.iter().map(|(name, text)| (name.to_string(), Value::from(*text)))
                    .collect::<serde_json::Map<String, Value>>(),
                "steady_state": STEADY_STATE,
            },
            "machine": { "before": before, "after": after },
            "weights": weights_dir,
            "ane": ane,
            "requests": results.iter().map(|m| json!({
                "name": m.name,
                "run": kind_name(m.follow_up),
                "path": m.path.display().to_string(),
                "tokens": m.tokens,
                "rows_computed": m.rows,
                "route": route_name(m.route),
                "decisions": m.decisions,
                "forward_ms": Stats::of(&m.forward).json(&m.forward),
                "gpu_ms": if m.gpu.is_empty() { Value::Null } else { Stats::of(&m.gpu).json(&m.gpu) },
                "e2e_ms": Stats::of(&m.e2e).json(&m.e2e),
                "tokenize_ms": Stats::of(&m.tokenize).json(&m.tokenize),
                "head_ms": Stats::of(&m.head).json(&m.head),
                "ane_stats": m.ane_stats_json(),
            })).collect::<Vec<_>>(),
        });
        std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")
            .with_context(|| format!("writing {}", out.display()))?;
        println!("report: {}", out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::quantile;

    #[test]
    fn quantiles_interpolate_like_numpy() {
        let v: Vec<f64> = (1..=20).map(f64::from).collect();
        assert_eq!(quantile(&v, 0.5), 10.5);
        assert!((quantile(&v, 0.95) - 19.05).abs() < 1e-12);
        assert_eq!(quantile(&[3.0], 0.95), 3.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0], 0.5), 2.0);
    }
}
