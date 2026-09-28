//! `decide race`: one request through decide and each competitor, side by
//! side: load time, forward p50/p95, the answers, and parity
//! with the fp32 gold next to every time, then decide's speedup over each.
//!
//! **One engine at a time.** Each competitor is one of the baseline
//! scripts (`reference/bench_mlx.py --impl best|port` (`port` was `shipped`),
//! `reference/bench_torch_mps.py`) in its own process, in `--race` mode: one
//! request in, one JSON object out, whose `engine` and `label` name it in the
//! output. They run one after another, each process
//! exiting (and so freeing its model) before the next starts, and decide runs
//! last, in this process, after every competitor has exited. That keeps the
//! peak at one model: decide's weights are never resident beside a
//! competitor's, and nothing depends on a library releasing memory on
//! request. Before each engine loads, the race waits (up to two minutes) for
//! [`MIN_FREE_PERCENT`] of memory to be free.
//!
//! **Locks.** The race takes `/tmp/decide-model.lock`, then
//! `/tmp/decide-gpu.lock` (the plan's resource rules; `flock`, which
//! `lockf(1)` also uses) and holds both for the whole sequence, so do not
//! wrap it in `lockf`. `--no-lock` skips them for a caller that holds them.
//!
//! **Honesty.** Every engine's parity with the gold is printed next to its
//! time; a failed competitor is printed with its error; and each engine
//! states its own timing spans, which are checked against decide's (the
//! endpoints of `forward` and `e2e`), with a loud warning where they differ.
//! decide is timed cold (the state cache emptied before every run, as for
//! a new request), which is the route the competitors' shortcuts match.
//!
//! `--follow-ups <second.json>` adds the state-cache demo: decide on the
//! second request (the same state, other questions) cold, and as a
//! follow-up after the first request has cached the state.

use crate::bench::{self, Measured, Stats};
use crate::check::{MARGIN, MAX_DP};
use anyhow::{bail, Context, Result};
use decide_engine::head::top2_margin;
use serde_json::{json, Map, Value};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The plan's model lock (one engine's weights loaded at a time) and GPU
/// lock (one timing process at a time).
pub const MODEL_LOCK: &str = "/tmp/decide-model.lock";
pub const GPU_LOCK: &str = "/tmp/decide-gpu.lock";
/// Free memory (per `memory_pressure`) wanted before an engine loads.
pub const MIN_FREE_PERCENT: u64 = 35;
/// How long to wait for it before loading anyway (with a warning).
const MEMORY_WAIT: Duration = Duration::from_secs(120);

/// A competitor: a baseline script in `--race` mode. Its id and
/// display name in the output are the ones its own JSON gives (`engine`,
/// `label`); `name` only selects it.
struct Competitor {
    name: &'static str,
    /// Other names `--competitors` accepts for it.
    aliases: &'static [&'static str],
    script: &'static str,
    /// `--impl` values, in order of preference: the first one the script's
    /// source offers is passed (`[]`: no `--impl`).
    impls: &'static [&'static str],
}

const COMPETITORS: [Competitor; 3] = [
    Competitor { name: "mlx-best", aliases: &[], script: "bench_mlx.py", impls: &["best"] },
    Competitor { name: "mlx-port", aliases: &["mlx-shipped"], script: "bench_mlx.py", impls: &["port", "shipped"] },
    Competitor { name: "torch-mps", aliases: &[], script: "bench_torch_mps.py", impls: &[] },
];

/// The competitor names `--competitors` accepts.
pub fn competitor_names() -> Vec<&'static str> {
    COMPETITORS.iter().flat_map(|c| std::iter::once(c.name).chain(c.aliases.iter().copied())).collect()
}

fn competitor(name: &str) -> Option<&'static Competitor> {
    COMPETITORS.iter().find(|c| c.name == name || c.aliases.contains(&name))
}

/// The `--impl` value to pass: the first of `c.impls` that the script's
/// source offers as a quoted choice (the scripts rename implementations).
fn impl_arg(c: &Competitor, script: &Path) -> Option<&'static str> {
    let source = std::fs::read_to_string(script).unwrap_or_default();
    c.impls.iter().copied().find(|i| source.contains(&format!("\"{i}\""))).or(c.impls.first().copied())
}

pub struct Options<'a> {
    pub weights: &'a Path,
    pub request: &'a Path,
    pub competitors: &'a [String],
    pub warmup: usize,
    pub iters: usize,
    pub follow_up: Option<&'a Path>,
    pub json: Option<&'a Path>,
    pub python: &'a Path,
    pub scripts: &'a Path,
    pub gold: &'a Path,
    pub lock: bool,
    pub timeout: Duration,
}

/// One engine's line of the race.
struct Row {
    engine: String,
    /// The engine's display name, if it gives one.
    label: Option<String>,
    /// A note printed under the row.
    note: Option<String>,
    error: Option<String>,
    load_s: Option<f64>,
    forward_p50: Option<f64>,
    forward_p95: Option<f64>,
    e2e_p50: Option<f64>,
    rows: Option<u64>,
    route: Option<String>,
    /// {field: decision}, in field order.
    answers: Map<String, Value>,
    gold: Option<String>,
    max_dp: Option<f64>,
    max_dlogit: Option<f64>,
    /// (field, ours, gold's, gold top-2 margin) for every decision that differs.
    mismatches: Vec<(String, String, String, Option<f64>)>,
    /// The engine's own `forward` and `e2e` span texts.
    spans: Map<String, Value>,
    /// Wall time of the engine's whole turn (process start to exit, or load to unload).
    turn_s: f64,
    /// The engine's own output (a competitor's race object; decide's summary).
    raw: Value,
}

impl Row {
    fn failed(engine: &str, error: String, turn_s: f64, raw: Value) -> Row {
        Row {
            engine: engine.to_string(),
            label: None,
            note: None,
            error: Some(error),
            load_s: None,
            forward_p50: None,
            forward_p95: None,
            e2e_p50: None,
            rows: None,
            route: None,
            answers: Map::new(),
            gold: None,
            max_dp: None,
            max_dlogit: None,
            mismatches: Vec::new(),
            spans: Map::new(),
            turn_s,
            raw,
        }
    }

    /// Parity with the gold under the plan's criteria: `None` without a
    /// gold, else whether max |Δp| ≤ 0.01 and no decision differs where the
    /// gold's top-2 margin is ≥ 0.05.
    fn parity(&self) -> Option<bool> {
        self.gold.as_ref()?;
        let dp_ok = self.max_dp.is_some_and(|d| d <= MAX_DP);
        let flips = self.mismatches.iter().any(|m| m.3.is_none_or(|g| g.is_nan() || g >= MARGIN));
        Some(dp_ok && !flips)
    }

    fn parity_text(&self) -> String {
        match self.parity() {
            None => "no gold".to_string(),
            Some(ok) => {
                let fields = self.answers.len();
                format!(
                    "{}: max |Δp| {}, decisions {}/{fields} match",
                    if ok { "ok" } else { "FAILS" },
                    self.max_dp.map_or("n/a".to_string(), |d| format!("{d:.5}")),
                    fields.saturating_sub(self.mismatches.len())
                )
            }
        }
    }
}

extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}
const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;

/// Take an exclusive `flock` on `path` (created if missing), waiting for it
/// if another process holds it. Held until the file is dropped.
fn take_lock(path: &str) -> Result<File> {
    let file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(path)
        .with_context(|| format!("opening {path}"))?;
    // SAFETY: a valid descriptor; flock has no memory effects.
    if unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
        println!("waiting for {path} (another process holds it; decide race takes the locks itself, so it must \
                  not run inside `lockf`)");
        // SAFETY: as above.
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            bail!("locking {path}: {}", std::io::Error::last_os_error());
        }
    }
    Ok(file)
}

/// Wait up to [`MEMORY_WAIT`] for [`MIN_FREE_PERCENT`] free memory before
/// `engine` loads; returns the free percentage it saw last.
fn wait_for_memory(engine: &str) -> Option<u64> {
    let started = Instant::now();
    loop {
        let free = bench::machine_state()["memory_free_percent"].as_u64();
        match free {
            Some(f) if f < MIN_FREE_PERCENT && started.elapsed() < MEMORY_WAIT => {
                println!("  {engine}: memory free {f}% < {MIN_FREE_PERCENT}%, waiting…");
                std::thread::sleep(Duration::from_secs(10));
            }
            Some(f) if f < MIN_FREE_PERCENT => {
                println!("  WARNING: {engine} loads with only {f}% memory free");
                return free;
            }
            _ => return free,
        }
    }
}

/// Run a competitor's script in race mode; its row, failed or not.
fn run_competitor(c: &Competitor, request: &Path, o: &Options) -> Row {
    let started = Instant::now();
    let script = o.scripts.join(c.script);
    let mut cmd = Command::new(o.python);
    cmd.arg(&script);
    if let Some(imp) = impl_arg(c, &script) {
        cmd.args(["--impl", imp]);
    }
    cmd.args(["--race", "--warmup", &o.warmup.to_string(), "--iters", &o.iters.to_string(), "--requests"])
        .arg(request)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            let msg = format!("could not start {} {}: {e} (see --python, --scripts)", o.python.display(), script.display());
            return Row::failed(c.name, msg, started.elapsed().as_secs_f64(), Value::Null);
        }
    };
    // Read both pipes while the child runs, so that neither fills up.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_string(&mut text);
            }
            text
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() > o.timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("timed out after {} s and was killed", o.timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => break Err(format!("waiting for it: {e}")),
        }
    };
    let (out, err) = (stdout.join().unwrap_or_default(), stderr.join().unwrap_or_default());
    let turn_s = started.elapsed().as_secs_f64();
    let tail: Vec<&str> = err.lines().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    let parsed: Option<Value> = out.lines().rev().find(|l| !l.trim().is_empty()).and_then(|l| serde_json::from_str(l).ok());
    let raw = parsed.clone().unwrap_or(Value::Null);
    let error = match (&status, &parsed) {
        (Err(e), _) => Some(e.clone()),
        (_, Some(v)) if !v["error"].is_null() => Some(v["error"].as_str().unwrap_or("failed").to_string()),
        (Ok(s), _) if !s.success() => Some(format!("exited with {s}; stderr: {}", tail.join(" | "))),
        (_, None) => Some(format!("printed no race JSON object; stderr: {}", tail.join(" | "))),
        _ => None,
    };
    if let Some(error) = error {
        return Row::failed(c.name, error, turn_s, raw);
    }
    let v = parsed.expect("checked above");
    let f = |k: &str| v[k].as_f64();
    Row {
        engine: v["engine"].as_str().unwrap_or(c.name).to_string(),
        label: v["label"].as_str().map(str::to_string),
        note: None,
        error: None,
        load_s: f("load_s"),
        forward_p50: f("forward_p50_ms").or(f("p50_ms")),
        forward_p95: f("forward_p95_ms").or(f("p95_ms")),
        e2e_p50: f("e2e_p50_ms"),
        rows: v["computed_tokens"].as_u64(),
        route: v["route"].as_str().map(str::to_string),
        answers: v["answers"].as_object().cloned().unwrap_or_default(),
        gold: v["gold"].as_str().map(str::to_string),
        max_dp: f("max_dp_vs_gold"),
        max_dlogit: f("max_dlogit_vs_gold"),
        mismatches: v["decision_mismatches"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|m| {
                let s = |k: &str| m[k].as_str().map_or_else(|| m[k].to_string(), str::to_string);
                (s("field"), s("ours"), s("gold"), m["gold_margin"].as_f64())
            })
            .collect(),
        spans: v["spans"].as_object().cloned().unwrap_or_default(),
        turn_s,
        raw: v,
    }
}

/// The fp32 gold whose prompt is `ids` (`<gold>/<set>/<name>.json`), if any.
fn find_gold(dir: &Path, ids: &[u32]) -> Result<Option<(String, Value)>> {
    let mut sets: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    sets.sort();
    for set in sets {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&set)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        for path in files {
            let g: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
                .with_context(|| format!("parsing {}", path.display()))?;
            let same = g["input_ids"].as_array().is_some_and(|a| {
                a.len() == ids.len() && a.iter().zip(ids).all(|(x, &y)| x.as_u64() == Some(u64::from(y)))
            });
            if same {
                let name = format!(
                    "{}/{}",
                    set.file_name().unwrap_or_default().to_string_lossy(),
                    path.file_stem().unwrap_or_default().to_string_lossy()
                );
                return Ok(Some((name, g)));
            }
        }
    }
    Ok(None)
}

/// decide's measured run as a row, with its parity against the gold.
/// `cache_entries` is the state cache's capacity.
fn decide_row(m: &Measured, load_s: f64, turn_s: f64, gold: Option<&(String, Value)>, cache_entries: usize,
              ane: &Value, weights: &str) -> Row {
    let (f, e) = (Stats::of(&m.forward), Stats::of(&m.e2e));
    let answers = m.decisions.as_object().cloned().unwrap_or_default();
    let mut row = Row {
        engine: "decide".into(),
        label: Some("decide (Rust + Metal: fp16 weights, prefix snapshot, tail truncation, last layer pruned to the \
                     answer rows)".into()),
        note: Some(format!(
            "timed cold: its state cache ({cache_entries} entries) emptied before every run, as for a new state"
        )),
        error: None,
        load_s: Some(load_s),
        forward_p50: Some(f.p50),
        forward_p95: Some(f.p95),
        e2e_p50: Some(e.p50),
        rows: Some(m.rows as u64),
        route: Some(bench::route_name(m.route).to_string()),
        answers,
        gold: None,
        max_dp: None,
        max_dlogit: None,
        mismatches: Vec::new(),
        spans: bench::SPANS
            .iter()
            .filter(|(k, _)| matches!(*k, "forward" | "e2e"))
            .map(|(k, v)| (k.to_string(), Value::from(*v)))
            .collect(),
        turn_s,
        raw: Value::Null,
    };
    if let Some((name, g)) = gold {
        let (mut dp, mut dlogit) = (0.0f64, 0.0f64);
        let worst = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) };
        let fields: Vec<&str> = g["fields"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        for (i, field) in fields.iter().enumerate() {
            let (ours, want) = (&m.response["answers"][field], &g["response"]["answers"][field]);
            match (ours["probabilities"].as_object(), want["probabilities"].as_object()) {
                (Some(p), Some(q)) if p.len() == q.len() => {
                    for (option, x) in p {
                        let d = match (x.as_f64(), q.get(option).and_then(Value::as_f64)) {
                            (Some(x), Some(y)) => (x - y).abs(),
                            _ => f64::NAN,
                        };
                        dp = worst(dp, d);
                    }
                }
                _ => dp = f64::NAN,
            }
            let gold_logits: Vec<f32> =
                g["logits"][field].as_array().into_iter().flatten().filter_map(Value::as_f64).map(|x| x as f32).collect();
            match m.logits.get(i) {
                Some(l) if l.len() == gold_logits.len() => {
                    for (a, b) in l.iter().zip(&gold_logits) {
                        dlogit = worst(dlogit, f64::from((a - b).abs()));
                    }
                }
                _ => dlogit = f64::NAN,
            }
            if ours["decision"] != want["decision"] {
                let margin = top2_margin(&gold_logits);
                let s = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_string);
                row.mismatches.push((
                    field.to_string(),
                    s(&ours["decision"]),
                    s(&want["decision"]),
                    margin.is_finite().then_some(margin),
                ));
            }
        }
        row.gold = Some(name.clone());
        row.max_dp = Some(dp);
        row.max_dlogit = Some(dlogit);
    }
    row.raw = json!({
        "load_s": load_s,
        "forward_ms": f.json(&m.forward),
        "gpu_ms": if m.gpu.is_empty() { Value::Null } else { Stats::of(&m.gpu).json(&m.gpu) },
        "e2e_ms": e.json(&m.e2e),
        "tokens": m.tokens,
        "rows_computed": m.rows,
        "route": bench::route_name(m.route),
        "state_cache_entries": cache_entries,
        "timed": "cold: the state cache emptied before every run",
        // Provenance: the weight set and the Neural Engine split decide ran
        // with, and what the ANE did per run (0 predicts with the split on
        // would mean Core ML fell back to the CPU).
        "weights": weights,
        "ane": ane,
        "DECIDE_ANE": std::env::var("DECIDE_ANE").ok(),
        "ane_stats": m.ane_stats_json(),
    });
    row
}

/// A span's endpoints, "from -> to" before its first ": ", normalised:
/// parenthesised asides, the word "candidate", case and spacing dropped.
fn endpoints(span: &str) -> Option<(String, String)> {
    let head = span.split(": ").next()?;
    let (from, to) = head.split_once("->")?;
    let norm = |s: &str| {
        let (mut out, mut depth) = (String::new(), 0usize);
        for c in s.chars() {
            match c {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                c if depth == 0 => out.push(c),
                _ => {}
            }
        }
        out.to_lowercase().replace("candidate ", "").split_whitespace().collect::<Vec<_>>().join(" ")
    };
    Some((norm(from), norm(to)))
}

/// Warnings for every competitor span whose endpoints differ from decide's.
fn span_warnings(decide: &Row, rows: &[Row]) -> Vec<String> {
    let mut out = Vec::new();
    for r in rows.iter().filter(|r| r.error.is_none() && r.engine != "decide") {
        for span in ["forward", "e2e"] {
            let ours = decide.spans.get(span).and_then(Value::as_str).and_then(endpoints);
            let theirs = r.spans.get(span).and_then(Value::as_str);
            match (ours, theirs.and_then(endpoints)) {
                (Some(a), Some(b)) if a == b => {}
                (Some(a), Some(b)) => out.push(format!(
                    "{}'s {span} span is not decide's: decide times \"{} -> {}\", {} times \"{} -> {}\". \
                     Its {span} times are not strictly comparable (full texts in the report).",
                    r.engine, a.0, a.1, r.engine, b.0, b.1
                )),
                (_, None) => out.push(format!("{} states no {span} span: its {span} times cannot be compared", r.engine)),
                (None, _) => out.push(format!("decide's {span} span cannot be parsed")),
            }
        }
    }
    out
}

fn ms(x: Option<f64>) -> String {
    x.map_or("-".into(), |x| format!("{x:.1}"))
}

/// Run the race and print it; `Ok(false)` if decide itself failed.
pub fn race(o: &Options) -> Result<bool> {
    for name in o.competitors {
        if competitor(name).is_none() {
            bail!("unknown competitor {name:?}; choose from {}", competitor_names().join(", "));
        }
    }
    anyhow::ensure!(o.iters >= 1, "--iters must be at least 1");
    let text = std::fs::read_to_string(o.request).with_context(|| format!("reading {}", o.request.display()))?;
    let request_abs = std::fs::canonicalize(o.request)?;
    let second = match o.follow_up {
        Some(p) => Some((p, std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)),
        None => None,
    };
    let _locks = if o.lock { Some((take_lock(MODEL_LOCK)?, take_lock(GPU_LOCK)?)) } else { None };
    let before = bench::machine_state();
    let date = Command::new("date").args(["-u", "+%Y-%m-%dT%H:%M:%SZ"]).output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let order: Vec<String> = o.competitors.iter().cloned().chain(["decide".to_string()]).collect();

    println!("decide race: {}", bench::request_name(o.request));
    println!("machine: {}", bench::describe_machine(&before));
    if let Some(per_core) = before["load_per_core"].as_f64().filter(|&x| x > 1.0) {
        println!("WARNING: load {per_core:.2} per core (above 1): the times below are indicative only");
    }
    println!(
        "{} warm-up + {} timed runs per engine; p50/p95 by linear interpolation; times in ms; forward p50 is the \
         compared number",
        o.warmup.max(1),
        o.iters
    );
    println!(
        "order: {}; one engine loaded at a time: each competitor is a process that exits before the next starts, \
         and decide loads last, in this process{}",
        order.join(", "),
        if o.lock { "; holding the model and GPU locks throughout" } else { "" }
    );
    println!();

    let mut rows = Vec::new();
    let mut memory = Map::new();
    for name in o.competitors {
        let c = competitor(name).expect("checked above");
        memory.insert(c.name.into(), wait_for_memory(c.name).into());
        print!("  {name:<12} running…");
        std::io::stdout().flush().ok();
        let row = run_competitor(c, &request_abs, o);
        match &row.error {
            None => println!(" done in {:.0} s", row.turn_s),
            Some(e) => println!(" FAILED after {:.0} s: {e}", row.turn_s),
        }
        rows.push(row);
    }

    // decide, last: cold on the request, then the follow-up demo.
    memory.insert("decide".into(), wait_for_memory("decide").into());
    print!("  {:<12} running…", "decide");
    std::io::stdout().flush().ok();
    let started = Instant::now();
    let mut follow = None;
    let mut provenance = (Value::Null, String::new());
    let decide = (|| -> Result<Row> {
        let mut engine = crate::open(o.weights)?;
        let load_s = engine.load_timings().total_ms / 1e3;
        let cache_entries = engine.forward().state_cache_capacity();
        provenance = (bench::ane_json(&engine), engine.weights_dir().display().to_string());
        let m = bench::measure_text(&mut engine, o.request, &text, o.warmup, o.iters, None)?;
        if let Some((path, second)) = &second {
            let cold = bench::measure_text(&mut engine, path, second, o.warmup, o.iters, None)?;
            let fu = bench::measure_text(&mut engine, path, second, o.warmup, o.iters, Some(&text))?;
            follow = Some((cold, fu));
        }
        drop(engine);
        let gold = find_gold(o.gold, &m.ids)?;
        Ok(decide_row(&m, load_s, started.elapsed().as_secs_f64(), gold.as_ref(), cache_entries, &provenance.0, &provenance.1))
    })();
    let decide = decide.unwrap_or_else(|e| Row::failed("decide", format!("{e:#}"), started.elapsed().as_secs_f64(), Value::Null));
    match &decide.error {
        None => println!(" done in {:.0} s", decide.turn_s),
        Some(e) => println!(" FAILED: {e}"),
    }
    let after = bench::machine_state();
    let decide_ok = decide.error.is_none();
    rows.insert(0, decide);
    let decide = &rows[0];
    let warnings = span_warnings(decide, &rows);

    // The table.
    println!();
    println!(
        "{:<12} {:>7} {:>11} {:>11} {:>9} {:>6}  {:<8}  parity vs fp32 gold",
        "engine", "load s", "forward p50", "forward p95", "e2e p50", "rows", "route"
    );
    for r in &rows {
        match &r.error {
            Some(e) => println!("{:<12} FAILED: {e}", r.engine),
            None => println!(
                "{:<12} {:>7} {:>11} {:>11} {:>9} {:>6}  {:<8}  {}",
                r.engine,
                r.load_s.map_or("-".into(), |s| format!("{s:.1}")),
                ms(r.forward_p50),
                ms(r.forward_p95),
                ms(r.e2e_p50),
                r.rows.map_or("-".into(), |n| n.to_string()),
                r.route.as_deref().unwrap_or("-"),
                r.parity_text()
            ),
        }
        if let Some(note) = &r.note {
            println!("{:<12}   {note}", "");
        }
        for (field, ours, want, margin) in &r.mismatches {
            println!(
                "{:<12}   decision differs at {field}: {ours}, gold {want} (gold top-2 margin {})",
                "",
                margin.map_or("n/a".into(), |m| format!("{m:.3}"))
            );
        }
    }
    println!("load s: each engine's own start-up (decide: tokenizer, weights, pipelines, prefix snapshot, warm-up \
              request; the scripts: after their imports)");

    // The answers, side by side.
    let ok: Vec<&Row> = rows.iter().filter(|r| r.error.is_none()).collect();
    let fields: Vec<String> = ok.iter().find(|r| !r.answers.is_empty()).map(|r| r.answers.keys().cloned().collect()).unwrap_or_default();
    // The gold's own decisions, for the last column: the gold any engine matched.
    let gold_name = rows.iter().find_map(|r| r.gold.clone());
    let gold_answers: Option<Map<String, Value>> = gold_name.as_ref().and_then(|name| {
        let g: Value = serde_json::from_str(&std::fs::read_to_string(o.gold.join(format!("{name}.json"))).ok()?).ok()?;
        let answers = g["response"]["answers"].as_object()?;
        Some(answers.iter().map(|(f, a)| (f.clone(), a["decision"].clone())).collect())
    });
    if !fields.is_empty() {
        println!();
        let fw = fields.iter().map(String::len).max().unwrap_or(5).max(7);
        let cell = |v: Option<&Value>| v.map_or("-".to_string(), |v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
        let mut cols: Vec<(String, Map<String, Value>)> = ok.iter().map(|r| (r.engine.clone(), r.answers.clone())).collect();
        if let (Some(name), Some(answers)) = (&gold_name, &gold_answers) {
            cols.push((format!("gold ({name})"), answers.clone()));
        }
        let widths: Vec<usize> = cols
            .iter()
            .map(|(name, a)| fields.iter().map(|f| cell(a.get(f)).len() + 1).chain([name.len()]).max().unwrap_or(8))
            .collect();
        let mut header = format!("{:<fw$}", "answers");
        for ((name, _), w) in cols.iter().zip(&widths) {
            header.push_str(&format!("  {name:<w$}"));
        }
        println!("{}", header.trim_end());
        let mut starred = false;
        for f in &fields {
            let mut line = format!("{f:<fw$}");
            for (i, ((_, a), w)) in cols.iter().zip(&widths).enumerate() {
                // `*`: an engine's decision that is not the gold's.
                let differs = i < ok.len() && gold_answers.as_ref().is_some_and(|g| g.get(f) != a.get(f));
                starred |= differs;
                let text = format!("{}{}", cell(a.get(f)), if differs { "*" } else { "" });
                line.push_str(&format!("  {text:<w$}"));
            }
            println!("{}", line.trim_end());
        }
        if starred {
            println!("(* = not the gold's decision)");
        }
    }

    // Speedups.
    println!();
    if let (true, Some(ours)) = (decide_ok, rows[0].forward_p50) {
        println!("decide's speedup over each competitor (forward p50, competitor / decide):");
        for r in rows.iter().skip(1) {
            match (&r.error, r.forward_p50) {
                (None, Some(theirs)) => {
                    let x = theirs / ours;
                    let verdict = if x >= 1.0 { format!("{x:.2}x faster") } else { format!("{x:.2}x (slower)") };
                    let spans = if warnings.iter().any(|w| w.starts_with(&format!("{}'s", r.engine))) {
                        "  [spans differ: see WARNING]"
                    } else {
                        ""
                    };
                    println!(
                        "  vs {:<12} {verdict:<14} ({theirs:.1} vs {ours:.1} ms)   its parity {}{spans}",
                        r.engine,
                        r.parity_text()
                    );
                }
                _ => println!("  vs {:<12} not measured: {}", r.engine, r.error.as_deref().unwrap_or("no forward time")),
            }
        }
        println!("  (decide's own parity: {})", rows[0].parity_text());
    } else {
        println!("no speedups: decide failed");
    }
    if warnings.is_empty() && rows.iter().skip(1).any(|r| r.error.is_none()) {
        println!("spans: every competitor's forward and e2e spans have decide's endpoints (token ids in memory -> \
                  logits on the CPU; request JSON text -> response JSON string)");
    }
    for w in &warnings {
        println!("WARNING: {w}");
    }

    // Every engine's own words for what it timed.
    println!();
    println!("engines and their timing spans, as each states them:");
    for r in rows.iter().filter(|r| r.error.is_none()) {
        println!("  {}: {}", r.engine, r.label.as_deref().unwrap_or("(no label given)"));
        for span in ["forward", "e2e"] {
            let text = r.spans.get(span).and_then(Value::as_str).unwrap_or("(not stated)");
            println!("    {span}: {text}");
        }
    }

    // The follow-up demo.
    let mut follow_json = Value::Null;
    if let (Some((cold, fu)), Some((path, _))) = (&follow, &second) {
        let (c, f) = (Stats::of(&cold.forward), Stats::of(&fu.forward));
        println!();
        println!("follow-up (decide; {} asks other questions about the same state):", bench::request_name(path));
        println!("  {:<10} {:>6}  {:>21}  {:<8}  answers", "run", "rows", "forward p50/p95", "route");
        for (label, m, s) in [("cold", cold, c), ("follow-up", fu, f)] {
            let answers: Vec<String> = m.decisions.as_object().into_iter().flatten()
                .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("?"))).collect();
            println!("  {label:<10} {:>6}  {:>9.1} / {:<9.1}  {:<8}  {}", m.rows, s.p50, s.p95,
                     bench::route_name(m.route), answers.join(" "));
        }
        let x = c.p50 / f.p50;
        println!(
            "  the follow-up is {x:.2}x faster: the first request cached the state, so it computes {} rows instead of {}",
            fu.rows, cold.rows
        );
        if fu.route != decide_engine::forward::Route::StateCache {
            println!("WARNING: the follow-up did not find a cached state (route {}): the two requests do not share \
                      a state", bench::route_name(fu.route));
        }
        if cold.decisions != fu.decisions {
            println!("WARNING: the follow-up's decisions differ from the cold run's");
        }
        follow_json = json!({
            "request": path.display().to_string(),
            "cold": {"rows": cold.rows, "route": bench::route_name(cold.route), "decisions": cold.decisions,
                     "forward_ms": c.json(&cold.forward), "e2e_ms": Stats::of(&cold.e2e).json(&cold.e2e)},
            "follow_up": {"rows": fu.rows, "route": bench::route_name(fu.route), "decisions": fu.decisions,
                          "forward_ms": f.json(&fu.forward), "e2e_ms": Stats::of(&fu.e2e).json(&fu.e2e)},
            "speedup_forward_p50": x,
        });
    }
    println!();
    println!("machine after: {}", bench::describe_machine(&after));

    if let Some(out) = o.json {
        let engines: Vec<Value> = rows
            .iter()
            .map(|r| {
                json!({
                    "engine": r.engine, "label": r.label, "note": r.note, "error": r.error, "load_s": r.load_s,
                    "forward_p50_ms": r.forward_p50, "forward_p95_ms": r.forward_p95, "e2e_p50_ms": r.e2e_p50,
                    "rows_computed": r.rows, "route": r.route, "answers": r.answers,
                    "gold": r.gold, "max_dp_vs_gold": r.max_dp, "max_dlogit_vs_gold": r.max_dlogit,
                    "decision_mismatches": r.mismatches.iter().map(|m| json!({
                        "field": m.0, "ours": m.1, "gold": m.2, "gold_margin": m.3})).collect::<Vec<_>>(),
                    "parity": r.parity(), "spans": r.spans, "turn_s": r.turn_s, "raw": r.raw,
                })
            })
            .collect();
        let speedups: Map<String, Value> = rows
            .iter()
            .skip(1)
            .filter_map(|r| Some((r.engine.clone(), Value::from(r.forward_p50? / rows[0].forward_p50?))))
            .collect();
        let report = json!({
            "date": date,
            "request": o.request.display().to_string(),
            "order": order,
            "protocol": {
                "warmup": o.warmup.max(1), "iters": o.iters,
                "percentiles": "linear interpolation between closest ranks",
                "compared": "forward p50",
                "decide_runs": "cold: the state cache emptied before every run (outside the timed region)",
                "decide_spans": bench::SPANS.iter().map(|(k, v)| (k.to_string(), Value::from(*v))).collect::<Map<_, _>>(),
                "memory_free_percent_before_each_load": memory,
                "locks": if o.lock { json!([MODEL_LOCK, GPU_LOCK]) } else { Value::Null },
            },
            "machine": {"before": before, "after": after},
            "weights": provenance.1,
            "ane": provenance.0,
            "engines": engines,
            "speedup_forward_p50": speedups,
            "span_warnings": warnings,
            "follow_up": follow_json,
        });
        std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n").with_context(|| format!("writing {}", out.display()))?;
        println!("report: {}", out.display());
    }
    Ok(decide_ok)
}

#[cfg(test)]
mod tests {
    use super::endpoints;

    #[test]
    fn span_endpoints_compare_what_is_timed() {
        let decide = endpoints(crate::bench::SPANS[0].1).unwrap();
        assert_eq!(decide, ("token ids in memory".to_string(), "logits on the cpu".to_string()));
        let mlx = "token ids in memory (a Python list) -> candidate logits on the CPU (Python floats): mx.array …";
        assert_eq!(endpoints(mlx).unwrap(), decide);
        let torch = "token ids in memory (HFBackend.encode has returned the tokenizer's CPU tensors) -> the model \
                     output on the device after torch.mps.synchronize(): the ids' copy to MPS";
        assert_ne!(endpoints(torch).unwrap(), decide);
        assert_eq!(endpoints(crate::bench::SPANS[2].1).unwrap().1, "response json string");
        assert!(endpoints("no arrow here").is_none());
    }
}
