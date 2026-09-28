//! `decide check`: every gold request through the engine, end to end (the
//! request's JSON in, the response out), against InternLM's fp32 gold.
//!
//! The plan's end-to-end criteria: identical decisions wherever the gold's
//! top-2 logit margin is at least [`MARGIN`], and calibrated probabilities
//! within [`MAX_DP`]. Beyond them, each request's prompt (ids, logit rows,
//! fields, candidate ids) must be the gold's exactly, and its response must
//! have the gold's keys in the gold's order, the same strings and ints and
//! exactly the gold's `calibration` (`backend` and `timing` aside, other
//! floats compared through the probabilities).
// `!(x <= limit)` rather than `x > limit`, so that a NaN fails.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use anyhow::{ensure, Context, Result};
use decide_engine::engine::{Decision, Engine};
use decide_engine::forward::Route;
use decide_engine::head::top2_margin;
use decide_prompt::pyjson::{classify, PyNum};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Decisions must agree wherever the gold's top-2 logit margin is at least this.
pub const MARGIN: f64 = 0.05;
/// Calibrated probabilities must agree within this, absolute.
pub const MAX_DP: f64 = 0.01;

/// A gold file and the request it was made from.
struct Case {
    /// `<set>/<stem>`, e.g. `core/card`.
    name: String,
    gold: PathBuf,
    request: PathBuf,
}

/// `<gold>/<set>/<name>.json` with its request `<requests>/<set>/<name>.json`
/// (the layout `reference/gold.py` writes): the `core` set first, then the
/// others, each by name.
fn cases(gold: &Path, requests: &Path) -> Result<Vec<Case>> {
    let mut sets: Vec<String> = std::fs::read_dir(gold)
        .with_context(|| format!("reading {}", gold.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    sets.sort_by_key(|s| (s != "core", s.clone()));
    let mut out = Vec::new();
    for set in sets {
        let mut files: Vec<PathBuf> = std::fs::read_dir(gold.join(&set))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        for path in files {
            let file = path.file_name().expect("a file name");
            let stem = path.file_stem().expect("a stem").to_string_lossy();
            out.push(Case { name: format!("{set}/{stem}"), request: requests.join(&set).join(file), gold: path });
        }
    }
    ensure!(!out.is_empty(), "no gold files in {}/<set>/", gold.display());
    Ok(out)
}

/// One request's line of the table.
struct Row {
    tokens: usize,
    route: Route,
    dlogit: f64,
    dp: f64,
    agree: usize,
    fields: usize,
    /// The smallest gold top-2 margin over the request's fields.
    margin: f64,
    forward_ms: f64,
}

/// The whole run's tallies.
#[derive(Default)]
struct Tally {
    requests: usize,
    fields: usize,
    agree: usize,
    /// Decision flips where the gold margin is below [`MARGIN`] (allowed).
    small_flips: Vec<String>,
    prompts_equal: usize,
    shapes_equal: usize,
    snapshot: usize,
    max_dlogit: (f64, String),
    max_dp: (f64, String),
    sum_dlogit: f64,
    logits: usize,
    forward_ms: Vec<f64>,
    failures: Vec<String>,
}

/// The larger of two errors; NaN if either is.
fn worst(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Keep the largest error and where it is; the first NaN sticks.
fn note_worst(slot: &mut (f64, String), value: f64, at: &str) {
    if !slot.0.is_nan() && (value.is_nan() || value > slot.0) {
        *slot = (value, at.to_string());
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

/// Whether the prompt is the gold's: ids, logit rows, fields, candidates.
fn same_prompt(d: &Decision, gold: &Value) -> bool {
    let p = &d.prepared;
    let fields: Vec<String> = gold["fields"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from).collect();
    let positions: Vec<usize> = u32s(&gold["positions"]).into_iter().map(|x| x as usize).collect();
    p.ids == u32s(&gold["input_ids"])
        && p.positions == positions
        && p.compiled.fields == fields
        && p.candidates.len() == fields.len()
        && fields.iter().zip(&p.candidates).all(|(f, c)| *c == u32s(&gold["candidate_ids"][f]))
}

/// Same keys in the same order, equal strings, ints and bools, at every
/// level; floats, `decision`, `choice`, `backend` and `timing` are left to
/// the caller. Appends each difference to `out`.
fn same_shape(ours: &Value, gold: &Value, path: &str, out: &mut Vec<String>) {
    match (ours, gold) {
        (Value::Object(a), Value::Object(b)) => {
            let (ka, kb): (Vec<&String>, Vec<&String>) = (a.keys().collect(), b.keys().collect());
            if ka != kb {
                out.push(format!("{path}: keys {ka:?}, gold {kb:?}"));
                return;
            }
            for (key, value) in a {
                if !matches!(key.as_str(), "decision" | "choice" | "backend" | "timing") {
                    same_shape(value, &b[key], &format!("{path}.{key}"), out);
                }
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                same_shape(x, y, &format!("{path}[{i}]"), out);
            }
        }
        (Value::Number(x), Value::Number(y)) => match (classify(x), classify(y)) {
            (PyNum::Float(_), PyNum::Float(_)) => {}
            (PyNum::Int(a), PyNum::Int(b)) if a == b => {}
            _ => out.push(format!("{path}: {x}, gold {y}")),
        },
        (a, b) if a == b => {}
        (a, b) => out.push(format!("{path}: {a}, gold {b}")),
    }
}

/// Compare one request with its gold; returns its table row.
fn compare(name: &str, d: &Decision, gold: &Value, t: &mut Tally) -> Row {
    let fields: Vec<&str> = gold["fields"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if same_prompt(d, gold) {
        t.prompts_equal += 1;
    } else {
        t.failures.push(format!("{name}: the prompt (ids, logit rows, fields or candidates) differs from the gold's"));
    }
    let mut shape = Vec::new();
    same_shape(&d.response, &gold["response"], name, &mut shape);
    // Constant per model: the method and the temperature's exact repr.
    if d.response["calibration"] != gold["response"]["calibration"] {
        shape.push(format!(
            "{name}.calibration: {}, gold {}",
            d.response["calibration"], gold["response"]["calibration"]
        ));
    }
    if shape.is_empty() {
        t.shapes_equal += 1;
    } else {
        t.failures.extend(shape.into_iter().map(|s| format!("response differs: {s}")));
    }
    let mut row = Row {
        tokens: d.prepared.ids.len(),
        route: d.route,
        dlogit: 0.0,
        dp: 0.0,
        agree: 0,
        fields: fields.len(),
        margin: f64::INFINITY,
        forward_ms: d.timings.forward_ms,
    };
    for (i, field) in fields.iter().enumerate() {
        let at = format!("{name}:{field}");
        let gold_logits = f64s(&gold["logits"][field]);
        let ours = d.logits.get(i).map(|v| v.iter().map(|&x| f64::from(x)).collect()).unwrap_or_else(Vec::new);
        if ours.len() != gold_logits.len() {
            t.failures.push(format!("{at}: {} logits, gold {}", ours.len(), gold_logits.len()));
            row.dlogit = f64::NAN;
            continue;
        }
        let gold_f32: Vec<f32> = gold_logits.iter().map(|&x| x as f32).collect();
        let margin = top2_margin(&gold_f32);
        row.margin = row.margin.min(margin);
        let mut dlogit = 0.0f64;
        for (a, b) in ours.iter().zip(&gold_logits) {
            let d = (a - b).abs();
            dlogit = worst(dlogit, d);
            t.sum_dlogit += d;
            t.logits += 1;
        }
        let answer = &d.response["answers"][field];
        let want = &gold["response"]["answers"][field];
        let mut dp = 0.0f64;
        match (answer["probabilities"].as_object(), want["probabilities"].as_object()) {
            (Some(p), Some(q)) if p.len() == q.len() => {
                for (key, pv) in p {
                    let delta = match (pv.as_f64(), q.get(key).and_then(Value::as_f64)) {
                        (Some(x), Some(y)) => (x - y).abs(),
                        _ => f64::NAN,
                    };
                    dp = worst(dp, delta);
                }
            }
            _ => dp = f64::NAN,
        }
        if !(dp <= MAX_DP) {
            t.failures.push(format!("{at}: |Δp| = {dp:.5}, above {MAX_DP}"));
        }
        if dlogit.is_nan() {
            t.failures.push(format!("{at}: NaN logit difference"));
        }
        row.dlogit = worst(row.dlogit, dlogit);
        row.dp = worst(row.dp, dp);
        note_worst(&mut t.max_dlogit, dlogit, &at);
        note_worst(&mut t.max_dp, dp, &at);
        let (ours_decision, gold_decision) = (&answer["decision"], &want["decision"]);
        if ours_decision == gold_decision && answer.get("choice") == want.get("choice") {
            row.agree += 1;
        } else {
            let flip = format!("{at}: {ours_decision}, gold {gold_decision} (gold top-2 margin {margin:.4})");
            if margin >= MARGIN || margin.is_nan() {
                t.failures.push(format!("decision differs at {flip}"));
            } else {
                t.small_flips.push(flip);
            }
        }
    }
    t.fields += fields.len();
    t.agree += row.agree;
    t.forward_ms.push(row.forward_ms);
    if row.route != Route::Full {
        t.snapshot += 1;
    }
    row
}

fn route_name(route: Route) -> &'static str {
    match route {
        Route::Snapshot => "snapshot",
        Route::Full => "full",
        Route::StateCache => "cache",
    }
}

/// Run every gold request under `gold` and print the table and a summary;
/// returns whether parity holds.
pub fn check(engine: &mut Engine, gold: &Path, requests: &Path) -> Result<bool> {
    let cases = cases(gold, requests)?;
    println!("{} gold requests from {}, requests from {}", cases.len(), gold.display(), requests.display());
    println!(
        "criteria: decisions agree wherever the gold top-2 logit margin is >= {MARGIN}; calibrated |Δp| <= {MAX_DP}; \
         prompts and response envelopes identical"
    );
    println!();
    println!(
        "{:<18} {:>6}  {:<8}  {:>11}  {:>9}  {:>9}  {:>11}  {:>10}",
        "request", "tokens", "route", "max|Δlogit|", "max|Δp|", "decisions", "gold margin", "forward ms"
    );
    let mut t = Tally::default();
    for case in &cases {
        let gold: Value = serde_json::from_str(
            &std::fs::read_to_string(&case.gold).with_context(|| format!("reading {}", case.gold.display()))?,
        )
        .with_context(|| format!("parsing {}", case.gold.display()))?;
        let request =
            std::fs::read_to_string(&case.request).with_context(|| format!("reading {}", case.request.display()))?;
        t.requests += 1;
        let d = match engine.decide_detailed(&request) {
            Ok(d) => d,
            Err(e) => {
                println!("{:<18} error: {e}", case.name);
                t.failures.push(format!("{}: the engine refused or failed the request: {e}", case.name));
                continue;
            }
        };
        let row = compare(&case.name, &d, &gold, &mut t);
        println!(
            "{:<18} {:>6}  {:<8}  {:>11.4}  {:>9.5}  {:>4}/{:<4}  {:>11.3}  {:>10.1}",
            case.name,
            row.tokens,
            route_name(row.route),
            row.dlogit,
            row.dp,
            row.agree,
            row.fields,
            row.margin,
            row.forward_ms
        );
    }
    let pass = t.failures.is_empty();
    println!();
    if !pass {
        println!("failures:");
        for f in &t.failures {
            println!("  {f}");
        }
    }
    let flips = if t.small_flips.is_empty() { "none".to_string() } else { t.small_flips.join("; ") };
    let mean_ms = t.forward_ms.iter().sum::<f64>() / t.forward_ms.len().max(1) as f64;
    let max_ms = t.forward_ms.iter().copied().fold(0.0, f64::max);
    println!("summary: {} requests, {} fields: {}", t.requests, t.fields, if pass { "PASS" } else { "FAIL" });
    println!("  prompts identical to the gold's (ids, logit rows, fields, candidates): {}/{}", t.prompts_equal, t.requests);
    println!(
        "  responses with the gold's keys, key order, strings, ints and calibration (backend, timing, other floats \
         aside): {}/{}",
        t.shapes_equal, t.requests
    );
    println!("  decisions agree: {}/{}; flips below the {MARGIN} gold margin (allowed): {flips}", t.agree, t.fields);
    println!(
        "  max |Δlogit| {:.4} at {}; mean |Δlogit| {:.4} over {} logits",
        t.max_dlogit.0,
        t.max_dlogit.1,
        t.sum_dlogit / t.logits.max(1) as f64,
        t.logits
    );
    println!("  max |Δp| {:.5} at {} (tolerance {MAX_DP})", t.max_dp.0, t.max_dp.1);
    println!("  route: from the prefix snapshot (or a cached state) {}/{}; forward mean {mean_ms:.1} ms, max {max_ms:.1} ms",
             t.snapshot, t.requests);
    Ok(pass)
}
