"""Shared benchmark protocol: machine state, warmup, percentiles, output.

The protocol is `decide bench`'s (crates/decide-cli/src/bench.rs), so the numbers compare:
- per request, max(warmup, 1) untimed runs, then `iters` timed ones (defaults 3 and 20);
- p50/p95 by linear interpolation between the closest ranks (numpy's default quantile);
- the same spans: **forward** (token ids in memory -> candidate logits on the CPU; the number
  compared against decide) and **e2e** (request JSON text in memory -> response JSON string);
  each script states exactly what its spans hold (`spans`, printed and saved);
- load time reported apart; the machine state recorded before and after (the engine unloaded),
  parsed as decide bench does, with its FLAG lines for load above 1 per core, battery power or
  low-power mode, so that a handicapped run cannot pass silently.

Also shared by the baselines (bench_mlx.py, bench_torch_mps.py):
- `Pipeline`: InternLM's request path (inference.py's validate_request, compile_row, chat
  template and AutoTokenizer, then DecisionEngine.predict's post-processing), so every engine
  runs the gold's token ids and answers in inference.py's response format;
- the comparison against the fp32 gold (tests/fixtures/gold/fp32), matched by token ids.

Race protocol (--race), for `decide race`: one request (a file, or stdin), warm-up, N timed
runs, then ONE JSON object on stdout; everything else goes to stderr. Keys:
  engine, label, load_s, p50_ms, p95_ms (forward: the compared span), forward_p50_ms, forward_p95_ms,
  e2e_p50_ms, e2e_p95_ms, n, warmup, tokens, computed_tokens, route,
  answers {field: decision}, probs {field: {option: calibrated p}},
  gold, max_dp_vs_gold, max_dlogit_vs_gold, decisions_match_gold, decision_mismatches,
  spans, machine, machine_after, flags (decide bench's FLAG lines: load above 1 per core, not on
  AC power, low-power mode), error (null unless the engine failed; then exit status 1).
Non-finite numbers are written as null, so the output is strict JSON.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except OSError:
        return ""


def machine_state() -> dict:
    """decide bench's machine state (crates/decide-cli/src/bench.rs, parsed the same way), plus
    the plan's keys (ac_power, lowpowermode, machine, time)."""
    batt = _run("pmset", "-g", "batt")
    power = batt.split("'")[1] if batt.count("'") >= 2 else "unknown"
    low_power = next((l.split()[1] for l in _run("pmset", "-g").splitlines()
                      if l.strip().startswith("lowpowermode") and len(l.split()) > 1), None)
    low_power = int(low_power) if low_power is not None and low_power.isdigit() else None
    load = list(os.getloadavg())
    cores = os.cpu_count() or 1
    free = [l for l in _run("memory_pressure").splitlines() if "free percentage" in l]
    swap = _run("sysctl", "-n", "vm.swapusage").split("used = ")
    top = []
    for line in _run("ps", "-Ao", "pcpu=,comm=", "-r").splitlines()[:5]:
        cpu, _, comm = line.strip().partition(" ")
        try:
            top.append({"cpu_percent": float(cpu), "process": comm.strip().rsplit("/", 1)[-1]})
        except ValueError:
            pass
    return {"power_source": power, "ac_power": power == "AC Power", "low_power_mode": low_power,
            "lowpowermode": low_power != 0, "load_average": load, "cores": cores,
            "load_per_core": round(load[0] / cores, 2), "load_flag": load[0] / cores > 1.0,
            "memory_free_percent": int(free[-1].rsplit(":", 1)[1].strip().rstrip("%")) if free else None,
            "swap_used_mb": round(float(swap[1].split()[0].rstrip("M"))) if len(swap) > 1 else None,
            "top_processes": top, "machine": platform.machine(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def flags(before: dict, after: dict | None = None) -> list[str]:
    """decide bench's FLAG lines, so that a handicapped run cannot pass silently."""
    out = []
    if before.get("load_flag") or (after or {}).get("load_flag"):
        out.append("FLAG: load above 1 per core; these timings are not protocol numbers")
    if before.get("power_source") != "AC Power" or before.get("low_power_mode") != 0:
        out.append("FLAG: not on AC power with low-power mode off; these timings are not protocol numbers")
    return out


def describe_machine(m: dict) -> str:
    """decide bench's one-line machine description."""
    load = " ".join(f"{x:.2f}" for x in m["load_average"])
    lpm = {0: "off", None: "unknown"}.get(m["low_power_mode"], "ON")
    top = ", ".join(f"{p['process']} {p['cpu_percent']:.0f}%" for p in m["top_processes"])
    return (f"{m['power_source']}, low-power mode {lpm}, load {load} on {m['cores']} cores = {m['load_per_core']:.2f}/core"
            f"{' [FLAG: above 1/core]' if m['load_flag'] else ''}, memory free {m['memory_free_percent']}%, "
            f"swap used {m['swap_used_mb']} MB\n  busiest: {top}")


def quantile(sorted_samples: list[float], p: float) -> float:
    """numpy's default quantile (linear between the closest ranks), as decide bench computes it."""
    rank = p * (len(sorted_samples) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (rank - lo)


def summarize(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    r = lambda x: round(x, 3)  # noqa: E731
    return {"n": len(s), "p50": r(quantile(s, 0.5)), "p95": r(quantile(s, 0.95)), "min": r(s[0]), "max": r(s[-1]),
            "mean": r(sum(s) / len(s)), "samples": [r(x) for x in samples_ms]}


def write(name: str, payload: dict) -> Path:
    out = ROOT / "docs" / "measurements" / f"{time.strftime('%Y-%m-%d')}-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return out


E2E_SPAN = ("request JSON text in memory -> response JSON string: json.loads, then inference.py's validate_request, "
            "compile_row, chat template and tokenizer, then forward, then predict's post-processing (softmax, "
            "temperature scaling, the response dict) and json.dumps of the response. Excludes reading the request "
            "file and writing stdout.")
COMPARED = "forward p50 is the number compared against decide (decide bench's forward span)."
PERCENTILES = "linear interpolation between closest ranks (numpy's default quantile), as decide bench"
STEADY_STATE = ("the steady state of a repeated request: each request runs warm-up + timed runs in a row "
                "(as decide bench).")


# ---------------------------------------------------------------------------------------------
# InternLM's request path

MODEL = ROOT / "models" / "Intern-Decision-4B"
GOLD = ROOT / "tests" / "fixtures" / "gold" / "fp32"
CORE = ROOT / "tests" / "fixtures" / "requests" / "core"
MAX_LENGTH = 8192  # DecisionEngine's default max_length
MAX_DP = 0.01      # plan, Tolerances: calibrated probabilities within 0.01 of the gold
MARGIN = 0.05      # ... and identical decisions wherever the gold top-2 logit margin is >= 0.05


def inference():
    """InternLM's inference.py, imported from beside the weights, as gold.py does."""
    sys.dont_write_bytecode = True  # no __pycache__ in the model directory
    if str(MODEL) not in sys.path:
        sys.path.insert(0, str(MODEL))
    import inference as inf
    return inf


@dataclass
class Encoded:
    row: dict
    compiled: object  # inference.CompiledExample
    ids: list[int]
    positions: list[int]  # logit rows: one before each <decision> marker
    candidates: dict[str, list[int]]  # field -> candidate symbol ids


class Pipeline:
    """inference.py's text path around a model call, step for step: HFBackend.encode
    (validate_request, compile_row, the chat template, the tokenizer, the marker rows), the
    candidate ids as predict encodes them, and predict's post-processing (softmax over the
    candidates, temperature scaling, the response dict), via gold.response_from_logits."""

    def __init__(self, tok=None):
        self.inf = inference()
        import gold  # reference/gold.py: predict's post-processing, line for line

        self._respond = gold.response_from_logits
        if tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        if self.inf.DECISION_TOKEN not in tok.get_added_vocab():
            raise ValueError("Checkpoint must include its trained decision tokenizer")
        self.tok = tok
        self.marker = tok.convert_tokens_to_ids(self.inf.DECISION_TOKEN)

    def _text(self, compiled) -> str:
        return self.tok.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                            enable_thinking=False, add_vision_id=True)

    def encode(self, request: dict) -> Encoded:
        inf = self.inf
        row = inf.validate_request(request)
        if row.get("images"):
            raise ValueError("text-only baseline: requests with images are not supported")
        compiled = inf.compile_row(row)
        ids = self.tok(self._text(compiled), add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_LENGTH:
            raise ValueError(f"Example has {len(ids)} tokens, above {MAX_LENGTH}; truncation is forbidden")
        positions = [i - 1 for i, t in enumerate(ids) if t == self.marker]
        if len(positions) != len(compiled.fields) or any(p < 0 for p in positions):
            raise ValueError("Decision marker count or position mismatch")
        candidates = {}
        for field in compiled.fields:
            encoded = [self.tok.encode(s, add_special_tokens=False) for s in compiled.symbols[field]]
            if any(len(e) != 1 for e in encoded):
                raise ValueError("Candidate symbols must each encode to a single token")
            candidates[field] = [e[0] for e in encoded]
        return Encoded(row, compiled, ids, positions, candidates)

    def respond(self, enc: Encoded, logits: dict[str, list[float]], forward_ms: float, backend: str) -> dict:
        """The response DecisionEngine.predict would build from these candidate logits."""
        result = self._respond(enc.row, enc.compiled, logits, len(enc.ids), forward_ms)
        result["backend"] = backend
        return result

    def prefix_ids(self) -> list[int]:
        """The constant prompt head, everything through "## State\\n" (decide's prefix), tokenized
        alone. Every request whose ids start with these can resume from a snapshot taken there."""
        compiled = self.inf.compile_row(self.inf.validate_request({"state": None, "questions": {"q": {"type": "noul"}}}))
        text = self._text(compiled)
        return self.tok(text[: text.index("## State\n") + len("## State\n")], add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------------------------
# Fidelity against the fp32 gold

def gold_index() -> dict[tuple, dict]:
    """Every fp32 gold request (core, calib), keyed by its token ids."""
    index = {}
    for path in sorted(GOLD.glob("*/*.json")):
        g = json.loads(path.read_text(encoding="utf-8"))
        g["name"] = f"{path.parent.name}/{path.stem}"
        index[tuple(g["input_ids"])] = g
    return index


def top2_margin(logits: list[float]) -> float | None:
    s = sorted(logits, reverse=True)
    return s[0] - s[1] if len(s) > 1 else None


def compare(response: dict, logits: dict | None, gold: dict) -> dict:
    """max |Δp| over every calibrated probability, max |Δlogit| over the candidate logits, and
    per-field decision agreement, with the gold's top-2 logit margin for every disagreement
    (the plan's parity rule only binds at margins >= 0.05)."""
    gans = gold["response"]["answers"]
    max_dp, worst, agree, mismatches = 0.0, None, 0, []
    for field, ans in response["answers"].items():
        g = gans[field]
        for option, p in ans["probabilities"].items():
            d = abs(p - g["probabilities"][option])
            if d > max_dp or (math.isnan(d) and not math.isnan(max_dp)):  # a NaN sticks
                max_dp, worst = d, field
        if ans["decision"] == g["decision"]:
            agree += 1
        else:
            mismatches.append({"field": field, "ours": ans["decision"], "gold": g["decision"],
                               "gold_margin": top2_margin(gold["logits"][field])})
    max_dlogit = None
    if logits is not None:
        max_dlogit = max(abs(a - b) for f in gold["fields"] for a, b in zip(logits[f], gold["logits"][f]))
    return {"gold": gold["name"], "max_dp": max_dp, "worst_field": worst, "max_dlogit": max_dlogit,
            "decisions_agree": agree, "fields": len(response["answers"]), "mismatches": mismatches}


# ---------------------------------------------------------------------------------------------
# The protocol

@dataclass
class Sample:
    """One request through an engine."""
    response: dict
    forward_ms: float  # the engine's forward span
    e2e_ms: float      # request JSON text -> response JSON string
    logits: dict | None  # field -> candidate logits
    ids: list[int]
    computed: int      # tokens the model actually ran
    route: str         # "full" or "snapshot"


def read_requests(specs: list[str] | None, race: bool) -> list[tuple[str, str]]:
    """(name, request JSON text) from files and directories (*.json, sorted); --race takes exactly
    one request, from a file or from stdin (no path, or "-")."""
    if race:
        if not specs or specs == ["-"]:
            return [("stdin", sys.stdin.read())]
        if len(specs) != 1 or Path(specs[0]).is_dir():
            raise SystemExit("--race takes one request: a file, or stdin")
    out = []
    for spec in specs or [str(CORE)]:
        path = Path(spec)
        for f in sorted(path.glob("*.json")) if path.is_dir() else [path]:
            out.append((f"{f.parent.name}/{f.stem}", f.read_text(encoding="utf-8")))
    if not out:
        raise SystemExit(f"no requests in {specs}")
    return out


def add_protocol_args(ap) -> None:
    ap.add_argument("--requests", nargs="+", metavar="PATH",
                    help="request files and/or directories of *.json (default: tests/fixtures/requests/core); "
                         "with --race, one file, or stdin when omitted or '-'")
    ap.add_argument("--warmup", type=int, default=3, help="untimed runs per request (at least 1 runs; default 3)")
    ap.add_argument("--iters", type=int, default=20, help="timed runs per request (default 20)")
    ap.add_argument("--json", metavar="OUT", help="write the report here instead of docs/measurements/<date>-<engine>.json")
    ap.add_argument("--race", action="store_true", help="one request in, one JSON object out (the `decide race` protocol)")


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("mlx", "mlx_lm", "torch", "transformers"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", None) or getattr(getattr(m, "core", None), "__version__", None)
    return out


def _finite(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


class RaceOutput:
    """--race: stdout carries exactly one JSON object. Until then fd 1 points at stderr, so
    nothing else (library prints and C-level writes included) can reach the consumer."""

    def __init__(self):
        sys.stdout.flush()
        self.fd = os.dup(1)
        os.dup2(2, 1)

    def emit(self, obj: dict) -> None:
        sys.stdout.flush()
        os.write(self.fd, (json.dumps(_finite(obj), allow_nan=False) + "\n").encode())


def fail(engine: str, race_out: RaceOutput | None, exc: BaseException, machine: dict | None) -> int:
    msg = f"{type(exc).__name__}: {exc}"
    traceback.print_exc(file=sys.stderr)
    if race_out is not None:
        race_out.emit({"engine": engine, "error": msg, "machine": machine})
    else:
        print(f"{engine}: FAILED: {msg}", file=sys.stderr)
    return 1


def fmt(x, spec=".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def drive(engine: str, run, requests: list[tuple[str, str]], args, *, load_s: float, machine_before: dict,
          info: dict, label: str, spans: dict, race_out: RaceOutput | None = None, extras=None, unload=None) -> int:
    """Warm-up + timed runs of `run(request_text) -> Sample` for every request, then the fidelity
    check against the gold and the output (race object, or table + JSON payload). `extras()` adds
    engine facts; `unload()` frees the weights before the machine state after, as decide bench does."""
    if args.iters < 1 or args.warmup < 0:
        raise SystemExit("--iters must be >= 1 and --warmup >= 0")
    spans = {**spans, "e2e": E2E_SPAN, "compared": COMPARED}
    per = []
    for name, text in requests:
        for _ in range(max(args.warmup, 1)):
            run(text)
        per.append((name, [run(text) for _ in range(args.iters)]))
    extra = extras() if extras else {}
    if unload:
        unload()
    machine_after = machine_state()
    flag_lines = flags(machine_before, machine_after)
    golds = gold_index()
    rows = []
    for name, samples in per:
        last = samples[-1]
        gold = golds.get(tuple(last.ids))
        cmp = compare(last.response, last.logits, gold) if gold else None
        rows.append({
            "name": name, "tokens": len(last.ids), "computed": last.computed, "route": last.route,
            "forward_ms": summarize([s.forward_ms for s in samples]),
            "e2e_ms": summarize([s.e2e_ms for s in samples]),
            "identical_across_iters": all(s.logits == last.logits and s.response["answers"] == last.response["answers"]
                                          for s in samples),
            "answers": {f: a["decision"] for f, a in last.response["answers"].items()},
            "probabilities": {f: a["probabilities"] for f, a in last.response["answers"].items()},
            "logits": last.logits, "fidelity": cmp})

    if race_out is not None:
        r = rows[0]
        cmp = r["fidelity"] or {}
        race_out.emit({
            "engine": engine, "label": label, "load_s": load_s,
            "p50_ms": r["forward_ms"]["p50"], "p95_ms": r["forward_ms"]["p95"],
            "forward_p50_ms": r["forward_ms"]["p50"], "forward_p95_ms": r["forward_ms"]["p95"],
            "e2e_p50_ms": r["e2e_ms"]["p50"], "e2e_p95_ms": r["e2e_ms"]["p95"],
            "n": args.iters, "warmup": max(args.warmup, 1), "tokens": r["tokens"], "computed_tokens": r["computed"],
            "route": r["route"], "answers": r["answers"], "probs": r["probabilities"],
            "gold": cmp.get("gold"), "max_dp_vs_gold": cmp.get("max_dp"), "max_dlogit_vs_gold": cmp.get("max_dlogit"),
            "decisions_match_gold": (not cmp["mismatches"]) if cmp else None,
            "decision_mismatches": cmp.get("mismatches"), "spans": spans,
            "machine": machine_before, "machine_after": machine_after, "flags": flag_lines, "error": None, **extra})
        for line in flag_lines:
            print(f"{engine}: {line}", file=sys.stderr)
        print(f"{engine}: {r['name']} forward p50 {r['forward_ms']['p50']:.1f} ms, e2e p50 {r['e2e_ms']['p50']:.1f} ms, "
              f"load {load_s:.1f} s; max |Δp| vs gold {fmt(cmp.get('max_dp'))}", file=sys.stderr)
        return 0

    print(f"{engine}: {label}")
    print(f"load {load_s:.1f} s; {max(args.warmup, 1)} warm-up + {args.iters} timed runs per request; "
          f"p50/p95 by linear interpolation; times in ms")
    print(f"machine before: {describe_machine(machine_before)}")
    print("spans:")
    for key, text in spans.items():
        print(f"  {key}: {text}")
    print(f"{'request':16} {'tokens':>6} {'ran':>5} {'route':8} {'fwd p50':>8} {'fwd p95':>8} "
          f"{'e2e p50':>8} {'e2e p95':>8}  {'max|Δp|':>8} {'max|Δlogit|':>11} decisions")
    for r in rows:
        c = r["fidelity"]
        dec = f"{c['decisions_agree']}/{c['fields']}" if c else "no gold"
        print(f"{r['name']:16} {r['tokens']:6} {r['computed']:5} {r['route']:8} {r['forward_ms']['p50']:8.1f} "
              f"{r['forward_ms']['p95']:8.1f} {r['e2e_ms']['p50']:8.1f} {r['e2e_ms']['p95']:8.1f}  "
              f"{fmt(c and c['max_dp']):>8} {fmt(c and c['max_dlogit']):>11} {dec}")
        for m in (c or {}).get("mismatches", []):
            print(f"  decision differs at {r['name']}:{m['field']}: {m['ours']} vs gold {m['gold']} "
                  f"(gold top-2 logit margin {fmt(m['gold_margin'], '.3f')})")
    checked = [r for r in rows if r["fidelity"]]
    summary = None
    if checked:
        worst = max(checked, key=lambda r: r["fidelity"]["max_dp"])
        dlogits = [r["fidelity"]["max_dlogit"] for r in checked if r["fidelity"]["max_dlogit"] is not None]
        flips = [f"{r['name']}:{m['field']}" for r in checked for m in r["fidelity"]["mismatches"]
                 if m["gold_margin"] is None or m["gold_margin"] >= MARGIN]
        summary = {"requests": len(checked), "fields": sum(r["fidelity"]["fields"] for r in checked),
                   "decisions_agree": sum(r["fidelity"]["decisions_agree"] for r in checked),
                   "max_dp": worst["fidelity"]["max_dp"],
                   "worst": f"{worst['name']}:{worst['fidelity']['worst_field']}",
                   "max_dlogit": max(dlogits) if dlogits else None,
                   "flips_at_margin_ge_0.05": flips,
                   "parity": bool(worst["fidelity"]["max_dp"] <= MAX_DP and not flips)}
        print(f"fidelity vs fp32 gold: {summary['requests']} requests, {summary['fields']} fields; max |Δp| "
              f"{summary['max_dp']:.5f} ({summary['worst']}); max |Δlogit| {fmt(summary['max_dlogit'])}; decisions agree "
              f"{summary['decisions_agree']}/{summary['fields']}; flips at gold margin >= {MARGIN}: {flips or 'none'}; "
              f"within the plan's parity criteria: {'yes' if summary['parity'] else 'NO'}")
    missing = [r["name"] for r in rows if not r["fidelity"]]
    if missing:
        print(f"no gold for: {', '.join(missing)}")
    for key, value in extra.items():
        print(f"{key}: {json.dumps(value)}")
    print(f"machine after (engine unloaded): {describe_machine(machine_after)}")
    for line in flag_lines:
        print(line)
    payload = {"engine": engine, "label": label, **info, "versions": versions(), "load_s": load_s,
               "protocol": {"warmup": max(args.warmup, 1), "iters": args.iters, "percentiles": PERCENTILES,
                            "spans": spans, "steady_state": STEADY_STATE},
               "machine": {"before": machine_before, "after": machine_after}, "flags": flag_lines,
               "requests": rows, "fidelity": summary, **extra}
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_finite(payload), indent=1))
    else:
        out = write(engine, _finite(payload))
    print(f"wrote {out}")
    return 0
