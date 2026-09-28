//! `decide`: the command-line front end.

mod bench;
mod check;
mod race;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use decide_engine::engine::Engine;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "decide", version, about = "Intern-Decision-4B requests on Metal")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert the bf16 safetensors checkpoint into decide's weight blobs.
    Convert {
        /// Hugging Face model directory (config.json, safetensors shards, tokenizer.json).
        #[arg(long)]
        model: PathBuf,
        /// Output directory: blobs, decide-index.json, tokenizer.json and config.json.
        #[arg(long)]
        out: PathBuf,
    },
    /// Build the GPU+ANE weight set from converted weights:
    /// `<out>/decide-index.json` + blobs holding the GPU's MLP slices, then
    /// the per-layer Core ML models via `reference/ane_convert.py`.
    AneConvert {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
        /// Output directory (the engine opens it with `DECIDE_ANE=1`).
        #[arg(long, default_value = "weights/ane")]
        out: PathBuf,
        /// SwiGLU intermediate dims on the ANE (multiple of 128).
        #[arg(long, default_value_t = 4096)]
        n_inter: usize,
        /// Rows per ANE predict (the models' fixed shape).
        #[arg(long, default_value_t = 128)]
        chunk: usize,
        /// The models' w_up pre-scale (the engine adds partial / scale).
        #[arg(long, default_value_t = 64.0)]
        out_scale: f32,
        /// Layers 0..layers are split; the rest stay on the GPU.
        #[arg(long, default_value_t = 31)]
        layers: usize,
        /// A Python with coremltools (not reference/.venv) for the models;
        /// `none` writes the weight set only.
        #[arg(long, default_value = ".venv-ane/bin/python")]
        python: PathBuf,
    },
    /// Run every gold request end to end and compare with InternLM's fp32
    /// gold outputs; exits 1 if parity fails.
    Check {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
        /// Gold outputs: `<gold>/<set>/<name>.json`.
        #[arg(long, default_value = "tests/fixtures/gold/fp32")]
        gold: PathBuf,
        /// The requests they were made from: `<requests>/<set>/<name>.json`.
        #[arg(long, default_value = "tests/fixtures/requests")]
        requests: PathBuf,
    },
    /// Decide one request: print InternLM's response as JSON (Python's
    /// `json.dumps(response, ensure_ascii=False, indent=2)`).
    Run {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
        /// The request's JSON file, or `-` for stdin.
        request: PathBuf,
    },
    /// Stay loaded and decide one request per stdin line, writing one JSON
    /// line per request to stdout: the response, or `{"error": "<message>"}`.
    /// Blank lines are skipped; lines over 4 MiB are refused.
    Serve {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
    },
    /// Warm single-request latency: p50/p95 of the forward pass and of end
    /// to end (JSON in → JSON out) per request, with the machine state and
    /// the load time. Hold the GPU timing lock while it runs.
    Bench {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
        /// Request files, or directories of `*.json` requests.
        #[arg(long, required = true, num_args = 1..)]
        requests: Vec<PathBuf>,
        /// Untimed runs per request before timing (at least one runs).
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Timed runs per request.
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Also write the full report (every sample) as JSON here.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Also time each request as a follow-up: its state already in the
        /// state cache. (Without it every run is cold: a new state.)
        #[arg(long)]
        follow_ups: bool,
    },
    /// Race one request through decide and each competitor (the baseline
    /// scripts in `reference/`), one engine loaded at a time: load time, forward p50/p95,
    /// answers and parity with the fp32 gold side by side, then decide's
    /// speedups. Takes the model and GPU locks itself (don't wrap it in lockf).
    Race {
        /// The converted weights (from `decide convert`).
        #[arg(long, default_value = "weights")]
        weights: PathBuf,
        /// The request's JSON file.
        request: PathBuf,
        /// Competitors, run in this order before decide: mlx-best, mlx-port
        /// (alias mlx-shipped), torch-mps; `none` for decide alone.
        #[arg(long, value_delimiter = ',', default_value = "mlx-best,mlx-port,torch-mps")]
        competitors: Vec<String>,
        /// Untimed runs per engine before timing (at least one runs).
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Timed runs per engine.
        #[arg(long, default_value_t = 10)]
        iters: usize,
        /// A second request about the same state: decide on it cold and as a
        /// follow-up (the state cached by the first request).
        #[arg(long)]
        follow_ups: Option<PathBuf>,
        /// Also write everything as JSON here.
        #[arg(long)]
        json: Option<PathBuf>,
        /// The Python that runs the baseline scripts.
        #[arg(long, default_value = "reference/.venv/bin/python")]
        python: PathBuf,
        /// Where the baseline scripts are.
        #[arg(long, default_value = "reference")]
        scripts: PathBuf,
        /// The fp32 gold outputs, matched by token ids.
        #[arg(long, default_value = "tests/fixtures/gold/fp32")]
        gold: PathBuf,
        /// Don't take the model and GPU locks (the caller holds them).
        #[arg(long)]
        no_lock: bool,
        /// Kill a competitor that runs longer than this many seconds.
        #[arg(long, default_value_t = 1800)]
        timeout: u64,
    },
}

fn main() -> ExitCode {
    match run_command(Cli::parse().command) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_command(command: Command) -> Result<ExitCode> {
    match command {
        Command::Convert { model, out } => convert(&model, &out).map(|()| ExitCode::SUCCESS),
        Command::AneConvert { weights, out, n_inter, chunk, out_scale, layers, python } => {
            ane_convert(&weights, &out, n_inter, chunk, out_scale, layers, &python).map(|()| ExitCode::SUCCESS)
        }
        Command::Check { weights, gold, requests } => {
            let mut engine = open(&weights)?;
            println!("decide check: {}", describe(&engine));
            let pass = check::check(&mut engine, &gold, &requests)?;
            Ok(if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
        Command::Run { weights, request } => run(&weights, &request).map(|()| ExitCode::SUCCESS),
        Command::Serve { weights } => serve(&weights).map(|()| ExitCode::SUCCESS),
        Command::Bench { weights, requests, warmup, iters, json, follow_ups } => {
            bench::bench(&weights, &requests, warmup, iters, follow_ups, json.as_deref()).map(|()| ExitCode::SUCCESS)
        }
        Command::Race {
            weights, request, competitors, warmup, iters, follow_ups, json, python, scripts, gold, no_lock, timeout,
        } => {
            let competitors: Vec<String> = competitors.into_iter().filter(|c| !c.is_empty() && c != "none").collect();
            let options = race::Options {
                weights: &weights,
                request: &request,
                competitors: &competitors,
                warmup,
                iters,
                follow_up: follow_ups.as_deref(),
                json: json.as_deref(),
                python: &python,
                scripts: &scripts,
                gold: &gold,
                lock: !no_lock,
                timeout: std::time::Duration::from_secs(timeout),
            };
            Ok(if race::race(&options)? { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
    }
}

/// `decide run`: the response on stdout; a refused request is an error
/// (`error: <message>` on stderr, exit 1).
fn run(weights: &Path, request: &Path) -> Result<()> {
    let text = if request == Path::new("-") {
        std::io::read_to_string(std::io::stdin()).context("reading the request from stdin")?
    } else {
        std::fs::read_to_string(request).with_context(|| format!("reading {}", request.display()))?
    };
    let mut engine = open(weights)?;
    let response = engine.decide(&text)?;
    println!("{}", decide_prompt::pyjson::dumps(&response, 2));
    Ok(())
}

/// Longest request line `serve` reads: 4 MiB, far above any request within
/// the 8192-token limit.
const MAX_LINE: usize = 4 << 20;

/// A line over the cap is drained up to its newline, but only this many
/// times the cap: past that the input is not a line-oriented client (a
/// stream with no newlines at all, e.g. `/dev/zero`), and draining it would
/// spin forever, so reading stops with an error.
const DRAIN_BUDGET: usize = 16;

/// Read one line of `input` into `line`, without its `\n`, holding at most
/// `cap` bytes: a longer line is drained up to its newline and reported as
/// `Some(false)`, with `line` empty. `None` at the end of the input. An
/// over-long line that runs past [`DRAIN_BUDGET`] × `cap` bytes without a
/// newline is an error (`InvalidData`).
fn read_line_capped(input: &mut impl std::io::BufRead, line: &mut Vec<u8>, cap: usize) -> std::io::Result<Option<bool>> {
    line.clear();
    let (mut read_any, mut fits, mut drained) = (false, true, 0usize);
    loop {
        let buf = input.fill_buf()?;
        if buf.is_empty() {
            return Ok(read_any.then_some(fits));
        }
        read_any = true;
        let newline = buf.iter().position(|&b| b == b'\n');
        let part = &buf[..newline.unwrap_or(buf.len())];
        if fits && line.len() + part.len() <= cap {
            line.extend_from_slice(part);
        } else {
            if fits {
                // Count what the line already held: the budget is over the whole line.
                drained = line.len();
            }
            fits = false;
            line.clear();
            drained = drained.saturating_add(part.len());
            if newline.is_none() && drained > DRAIN_BUDGET.saturating_mul(cap) {
                let used = buf.len();
                input.consume(used);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("a line ran past {} bytes without a newline; the input is not one request per line", DRAIN_BUDGET * cap),
                ));
            }
        }
        let used = newline.map_or(buf.len(), |i| i + 1);
        input.consume(used);
        if newline.is_some() {
            return Ok(Some(fits));
        }
    }
}

/// `decide serve`: one request per stdin line, one JSON line per request on
/// stdout, until stdin closes. The engine stays loaded and warm between
/// requests. A bad line (not JSON, refused, not UTF-8, over [`MAX_LINE`])
/// gets `{"error": "<message>"}` and the next line is served as usual; a
/// blank line is skipped, with no reply.
fn serve(weights: &Path) -> Result<()> {
    use std::io::Write;
    let mut engine = open(weights)?;
    eprintln!("decide serve: ready, one request per line ({})", describe(&engine));
    let mut input = std::io::stdin().lock();
    let mut out = std::io::stdout().lock();
    let mut line = Vec::new();
    // A read error ends serve with the reason (a stream that never sends a
    // newline is refused after DRAIN_BUDGET × MAX_LINE bytes, rather than
    // drained forever).
    while let Some(fits) = read_line_capped(&mut input, &mut line, MAX_LINE).context("reading stdin")? {
        let reply = if !fits {
            serde_json::json!({ "error": format!("the request line is longer than {MAX_LINE} bytes") }).to_string()
        } else {
            match std::str::from_utf8(&line) {
                Ok(text) if text.trim().is_empty() => continue,
                Ok(text) => match engine.decide(text.trim_end_matches('\r')) {
                    Ok(response) => serde_json::to_string(&response)?,
                    Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
                },
                Err(e) => serde_json::json!({ "error": format!("the request line is not UTF-8: {e}") }).to_string(),
            }
        };
        writeln!(out, "{reply}").context("writing stdout")?;
        out.flush().context("writing stdout")?;
    }
    Ok(())
}

/// Open the engine, or say which directory failed.
fn open(weights: &Path) -> Result<Engine> {
    Engine::open(weights).with_context(|| format!("opening the engine on {}", weights.display()))
}

/// The engine's kernels (and the variables that chose them) and load time.
fn describe(engine: &Engine) -> String {
    let fwd = engine.forward();
    let var = |name: &str| std::env::var(name).map(|v| format!("{name}={v}")).unwrap_or_else(|_| format!("{name} unset"));
    let load = engine.load_timings();
    // The Neural Engine split is provenance: which weight set, and whether the
    // MLP was split (a silent Core ML CPU fallback would show in ane_stats).
    let ane = match fwd.ane() {
        Some(a) => format!(
            "on (n_inter {}, chunks of {} rows, out_scale {}, layers 0..{})",
            a.n_inter, a.chunk_rows, a.out_scale, a.layers
        ),
        None => "off".to_string(),
    };
    format!(
        "weights {}; kernels GEMM {}, gated delta {} ({}, {}); Neural Engine {ane} ({}); prefix {} tokens, \
         chunks of {} rows, last-layer pruning {}; \
         loaded in {:.2} s (tokenizer {:.2} s, weights and pipelines {:.2} s, prefix snapshot {:.2} s, \
         warm-up request {:.2} s)",
        engine.weights_dir().display(),
        format!("{:?}", fwd.gemm()).to_lowercase(),
        format!("{:?}", fwd.delta()).to_lowercase(),
        var("DECIDE_GEMM"),
        var("DECIDE_DELTA"),
        var("DECIDE_ANE"),
        fwd.prefix_len().unwrap_or(0),
        fwd.chunk(),
        if fwd.pruning() { "on" } else { "off" },
        load.total_ms / 1e3,
        load.tokenizer_ms / 1e3,
        load.weights_ms / 1e3,
        load.snapshot_ms / 1e3,
        load.warm_up_ms / 1e3,
    )
}

/// Prints the conversion report, elapsed time and blob sizes as one JSON object.
fn convert(model: &Path, out: &Path) -> Result<()> {
    let start = Instant::now();
    let report = decide_engine::convert::convert(model, out)?;
    let elapsed_s = start.elapsed().as_secs_f64();

    let index_path = out.join(decide_engine::convert::INDEX_FILE);
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path).with_context(|| format!("reading {}", index_path.display()))?,
    )?;
    let mut blobs = serde_json::Map::new();
    let mut total_bytes = 0;
    for name in index["blobs"].as_array().context("index has no blob list")? {
        let name = name.as_str().context("blob name is not a string")?;
        let bytes = std::fs::metadata(out.join(name))?.len();
        total_bytes += bytes;
        blobs.insert(name.to_string(), bytes.into());
    }
    let summary = serde_json::json!({
        "report": report,
        "elapsed_s": (elapsed_s * 1000.0).round() / 1000.0,
        "blob_bytes": blobs,
        "total_blob_bytes": total_bytes,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    eprintln!("converted {} tensors in {elapsed_s:.1} s", report.tensors);
    Ok(())
}

/// `decide ane-convert`: the split weight set (Rust), then the models (Python).
fn ane_convert(weights: &Path, out: &Path, n_inter: usize, chunk: usize, out_scale: f32, layers: usize, python: &Path) -> Result<()> {
    let cfg = decide_engine::convert::AneConfig { n_inter, chunk_rows: chunk, out_scale, layers };
    let start = Instant::now();
    let report = decide_engine::convert::convert_ane(weights, out, &cfg)?;
    eprintln!("wrote the ANE weight set ({} tensors) to {} in {:.1} s", report.tensors, out.display(), start.elapsed().as_secs_f64());
    if python == Path::new("none") {
        return Ok(());
    }
    let script = Path::new("reference/ane_convert.py");
    let status = std::process::Command::new(python)
        .arg(script)
        .arg("--weights").arg(weights)
        .arg("--out").arg(out)
        .args(["--n-inter", &n_inter.to_string(), "--chunk", &chunk.to_string(), "--out-scale", &out_scale.to_string(),
               "--layers", &layers.to_string()])
        .status()
        .with_context(|| format!("running {} {}", python.display(), script.display()))?;
    anyhow::ensure!(status.success(), "{} failed ({status})", script.display());
    eprintln!("ANE weight set and models ready in {} ({:.1} s)", out.display(), start.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_line_capped;

    #[test]
    fn lines_are_capped_and_drained() {
        let mut input = std::io::BufReader::with_capacity(4, &b"ab\n0123456789\n\nxyz"[..]);
        let mut line = Vec::new();
        let mut got = Vec::new();
        while let Some(fits) = read_line_capped(&mut input, &mut line, 5).unwrap() {
            got.push((fits, String::from_utf8(line.clone()).unwrap()));
        }
        let want = [(true, "ab"), (false, ""), (true, ""), (true, "xyz")];
        assert_eq!(got, want.map(|(f, s)| (f, s.to_string())));
    }

    #[test]
    fn a_line_without_a_newline_is_refused_past_the_drain_budget() {
        // 5-byte cap, 16x budget = 80 bytes: a 200-byte "line" with no newline
        // must fail before it is consumed whole (fed /dev/zero, serve used to
        // spin forever here).
        let data = vec![b'z'; 200];
        let mut input = std::io::BufReader::with_capacity(7, &data[..]);
        let mut line = Vec::new();
        let err = read_line_capped(&mut input, &mut line, 5).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        assert!(err.to_string().contains("without a newline"), "{err}");
        // An over-long line that does end in a newline is still just skipped.
        let mut input = std::io::BufReader::with_capacity(7, &b"0123456789\nok\n"[..]);
        assert_eq!(read_line_capped(&mut input, &mut line, 5).unwrap(), Some(false));
        assert_eq!(read_line_capped(&mut input, &mut line, 5).unwrap(), Some(true));
        assert_eq!(line, b"ok");
    }
}
