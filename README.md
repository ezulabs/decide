# decide

Intern-Decision-4B on Apple Silicon: one Rust binary with hand-tuned Metal kernels and an optional Neural
Engine split.

[Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) (InternLM, Apache-2.0) answers typed
questions about a free-text state: which team should take this ticket, how urgent is it, is it a refund
request. For every option of every question it gives a calibrated probability. `decide` runs the model on a Mac
with no Python at runtime. Each request is one forward pass on the GPU. Optionally, part of every MLP runs on
the Neural Engine at the same time. The response is the same one InternLM's `inference.py` returns.

## Results

Forward time, p50 in milliseconds (lower is better). Measured on an M1 Pro (14-core GPU, 16 GB), macOS 15.8, AC
power. Each number is the median of 3 interleaved rounds.

| Engine | Card: 3 questions, 276 tokens | Ticket: 6 questions, 603 tokens | Long document: 3 questions, 2,307-token state |
| --- | --: | --: | --: |
| **decide + Neural Engine** | **313** | **681** | **3,462** |
| **decide, GPU only** | **342** | **922** | **5,260** |
| mlx-lm, best effort ¹ | 415 | 1,051 | 5,693 |
| mlx-lm, straightforward port ² | 894 | 1,738 | not run |
| InternLM's `inference.py`, PyTorch MPS fp16 ³ | 1,088 | 2,140 | not run |

decide's speedup, GPU only / with the Neural Engine. Each speedup is computed within one round, and the table
shows the median over the three rounds:

| Speedup over | Card | Ticket | Long document |
| --- | --: | --: | --: |
| mlx-lm, best effort ¹ | 1.22× / 1.32× | 1.12× / 1.56× | 1.14× / 1.62× |
| mlx-lm, straightforward port ² | 2.61× / 2.85× | 1.89× / 2.55× | not run |
| `inference.py`, PyTorch MPS fp16 ³ | 3.18× / 3.47× | 2.32× / 3.14× | not run |

1. mlx-lm 0.31.3 (MLX 0.32.2), given the same four work-saving tricks as decide (listed under
   [How it works](#how-it-works)), in fp16 with `A_log` kept in fp32
   ([`reference/bench_mlx.py`](reference/bench_mlx.py) `--impl best`). It computes exactly the rows decide
   computes, so the gap comes from kernels, dispatch and the Neural Engine, not from skipped work.
2. What you would write with mlx-lm: bf16, the whole prompt, the full-vocabulary head (`--impl port`).
3. The checkpoint's own `DecisionEngine` with `device="mps", dtype="float16"`. It runs in fp16 rather than bf16
   because bf16 is about 2× slower on MPS.

**What is measured.** *Forward time* runs from the prompt's token ids in memory to the answer rows' candidate
logits on the CPU. It excludes tokenization and the softmax and calibration step. Each engine runs in a warm
process that repeats the same request: 3 warm-up runs, then 20 timed runs (10 for the long document). Only one
engine is loaded at a time. decide's state cache is emptied before every timed run, so each run pays for its
whole state. Every time is printed beside that engine's agreement with the fp32 reference outputs. Every engine
got every decision right on the card and the ticket.

- **The Neural Engine's gain grows with the request:** 1.09× on the card, 1.35× on the ticket, 1.49× on the long
  document. On the card its p95 is worse than GPU only (390 ms against 343 ms).
- **It holds under sustained load.** On a 16-minute back-to-back soak on the ticket, GPU only ran at 922 ms p50
  and the Neural Engine split at 675 ms (1.37×). Neither changed between the first 5 minutes and the last, and
  the machine recorded no thermal warnings.
- **Startup** takes 4.6 s GPU only and 12–13 s with the Neural Engine. For comparison, mlx-lm takes 8–10 s and
  `inference.py` on MPS takes 38–44 s.

## Instant follow-ups

decide saves the model state at the end of a request's `## State` section, in a cache that holds 2 entries. A
later request about the same state, with different questions, starts from there and computes only its question
rows. In this example, the first request asks three triage questions about the long document, and the second
asks three new ones:

| Second request, 3 new questions | Rows computed | decide, GPU only | decide + Neural Engine |
| --- | --: | --: | --: |
| New state (cold) | 2,491 | 4,822 ms | 3,571 ms |
| Follow-up (state cached) | 184 | 390 ms | 451 ms |
| Speedup | | 12.4× | 7.9× |

Follow-ups give the same decisions as the cold path. Their probabilities differ by at most 5.3e-3. The gain
grows with the state: a short state gains much less (1.3–1.6× on the ticket's 143-token state). Each cache entry
takes about 53 MB plus 32 KB per state token.

## Correctness

The reference outputs come from InternLM's model code under transformers 5.14.1, run on the CPU in fp32, for 58
requests: the card, the ticket and 56 short calibration requests ([`tests/fixtures`](tests/fixtures)).
`decide check` runs all 58 through the same code path as `run` and `serve`:

| `decide check`, 58 requests | GPU only | + Neural Engine |
| --- | --: | --: |
| Prompts identical to the reference (token ids, logit rows, candidates) | 58/58 | 58/58 |
| Responses with the reference's keys, key order, strings and ints | 58/58 | 58/58 |
| Decisions agree | 65/65 | 65/65 |
| max \|Δlogit\| | 0.0081 | 0.0176 |
| max \|Δp\|, calibrated (tolerance 0.01) | 0.00089 | 0.00258 |

For comparison, InternLM's own bf16 path is further from the same fp32 reference outputs: max |Δp| 0.0161 on the
9 requests we ran it on, with no changed decisions. Part of the reason: the checkpoint stores `A_log` and the
gated-norm weights in fp32, and the bf16 path rounds them on load. decide keeps them in fp32.

**A tokenizer quirk, if you port this model.** The reference tokenizes with transformers 5.14.1's
`Qwen2Tokenizer`, which replaces the pre-tokenizer in `tokenizer.json` with the original Qwen2 regex. That regex
has no `\p{M}` (combining marks), so loading `tokenizer.json` directly gives different token ids for text with
combining marks, such as Hindi, Thai, Tamil, or emoji with variation selectors. decide uses the reference's
regex.

## Quickstart

You need:

- An Apple Silicon Mac. Only an M1 Pro with 16 GB on macOS 15.8 has been tested.
- Rust 1.85 or newer.
- Xcode, for its Metal toolchain and Swift compiler. The build runs `xcrun -sdk macosx metal` and `xcrun swiftc`
  even if you never use the Neural Engine.
- The Hugging Face CLI (`hf`).
- About 17 GB of disk: 8.5 GB for the checkpoint and 7.9 GB for the converted weights. The Neural Engine's
  weight set takes about 7 GB more.

```sh
cargo build --release
hf download internlm/Intern-Decision-4B --local-dir models/Intern-Decision-4B
target/release/decide convert --model models/Intern-Decision-4B --out weights   # about 45 s
target/release/decide check     # the 58 reference outputs ship in the repo; exits 1 if parity fails
```

`convert` rewrites the bf16 safetensors as fp16 blobs in fused layouts, copies the tokenizer and config, and
prints a JSON report. Every other command reads `--weights weights` by default. Run the commands from the
repository root.

**One request.** Pass a file, or `-` for stdin. The card is the model card's example: a customer charged twice,
and three questions (which team, how urgent, is it a refund request).

```sh
target/release/decide run tests/fixtures/requests/core/card.json
```

```jsonc
{"answers": {"team": {"type": "choice", "probabilities": {"billing": 0.987…, "delivery": 0.012…},
                      "confidence": 0.987…, "choice": "billing", "source": "local", "decision": "billing"}, …},
 "usage": {"input_tokens": 276, "output_tokens": 3, "decision_count": 3}, "timing": {"inference_ms": …},
 "calibration": {"method": "temperature-scaling", "temperature": 1.99241824},
 "model": "Intern-Decision-4B", "backend": "decide-metal"}
```

**A resident process.** `serve` loads the model once, then reads one request per line on stdin and writes one
JSON line per request on stdout. A bad line gets `{"error": "…"}`, and serving continues. There is no HTTP
server.

```sh
jq -c . tests/fixtures/requests/core/*.json | target/release/decide serve
```

**The Neural Engine (optional).** Building its weight set needs Python with coremltools. Running the split does
not.

```sh
uv venv --python 3.12 .venv-ane && uv pip install --python .venv-ane/bin/python coremltools numpy
target/release/decide ane-convert --weights weights --out weights/ane   # about 1 min
DECIDE_ANE=1 target/release/decide run --weights weights/ane tests/fixtures/requests/core/card.json
DECIDE_ANE=1 target/release/decide check --weights weights/ane
```

This writes a second weight set: smaller layer blobs, the embedding table hard-linked, and 31 Core ML models
(1.9 GB). decide refuses to open that set without `DECIDE_ANE=1`, and refuses a plain set with it.

**Reproduce the race.** `race` runs one request through each competitor and then decide, one engine loaded at a
time. It prints load time, forward p50/p95, the answers side by side, each engine's parity with the reference,
and decide's speedups. `bench` times decide alone; with `--follow-ups` it also times each request as a
follow-up.

```sh
uv venv --python 3.12 reference/.venv
uv pip install --python reference/.venv/bin/python \
    -r reference/requirements-internlm.txt -r reference/requirements-bench.txt
target/release/decide race tests/fixtures/requests/core/ticket6.json      # vs mlx-best, mlx-port, torch-mps
target/release/decide race tests/fixtures/requests/demo/incident_first.json --competitors mlx-best \
    --follow-ups tests/fixtures/requests/demo/incident_follow_up.json
target/release/decide bench --requests tests/fixtures/requests/core --follow-ups
```

## How it works

A request compiles to one chat prompt:

1. the system prompt;
2. the state;
3. a schema that maps every option to a one-token symbol (`A`–`Z`, `a`–`z`, `0`–`9`);
4. an assistant JSON skeleton with one `<decision>` marker per question.

The Rust prompt compiler reproduces InternLM's `compile_row` and chat template exactly, and is tested against
Python's output. One causal forward pass follows, with no decoding. For each question, decide takes the logits
just before its marker, applies softmax over that question's candidate symbols, and applies InternLM's fitted
temperature, T = 1.99241824.

**Doing less work.** The mlx-lm best-effort baseline uses all four of these as well.

- **Prefix snapshot.** The 107-token constant prefix runs once at startup, and every request starts from its
  saved state.
- **Tail truncation.** The model is causal, so nothing after the last marker is computed.
- **Last-layer pruning.** Layer 31 computes K/V for every row, but queries, attention, `o_proj` and the MLP
  only for the answer rows.
- **Candidate-only head.** The answer rows are scored against at most 62 symbol embeddings instead of the full
  248,320-token vocabulary.

**Doing it faster.** Each kernel in [`kernels/`](kernels) was measured against MLX's equivalent on the same day,
with the runs interleaved.

- **fp16 GEMM**, derived from MLX's steel GEMM. It handles the leftover rows inside the kernel, where MLX pads
  rows to a multiple of 64. It reads each weight once, and uses fused weight layouts (`qkv|z|b|a`,
  `q|gate|k|v`, `gate|up`) with SwiGLU and residual-add steps folded in. It was at or under MLX's `x @ W.T` in
  all 33 shape and row-count cases: 12–20% faster at 163 rows, and 4–10% faster at 493.
- **Gated-delta recurrence.** Each simdgroup keeps 16 state columns in registers, computes the gates itself and
  prefetches the next token. It takes 1.41 ms per layer at 493 tokens, against 3.0–3.6 ms for mlx-lm's kernel.
- **Attention** (head dim 256, causal GQA, fused sigmoid gate). It is at or slightly under MLX's SDPA at 163
  rows, 13–15% faster at 493 rows and 26% faster at 2,466.
- **Numerics.** Weights and GEMM inputs are fp16. Accumulation, the residual stream, the recurrence state and
  the small parameters are fp32.
- **GPU + Neural Engine.** In layers 0–30, the Neural Engine computes 4,096 of the MLP's 9,216 intermediate
  dimensions while the GPU computes the rest. It uses fixed-shape 128-row Core ML predictions. The GPU then
  waits on a shared event and adds the Neural Engine's part.

## Limits

- **Text only, one model, one request at a time.** The model is multimodal, but decide refuses `images`. Only
  the 4B checkpoint runs, because its shapes are compiled in. There is no batching.
- **One machine.** Every number comes from an M1 Pro (14-core GPU, 16 GB), where the kernel tiles were tuned.
  Other chips are untested. Peak memory was 7.5 GB GPU only and 5.6 GB with the Neural Engine.
- **8,192 tokens at most**, the reference's default `max_length`. Longer prompts are rejected, not truncated.
- **Warm, repeated requests.** The embedding table is mapped lazily, so a request with token ids that no earlier
  request used can take page faults inside its forward time. That case is not measured.
- **Stricter than the reference.** decide refuses some inputs that `inference.py` accepts:
  - `images`;
  - list or dict values in `instructions` or in option descriptions;
  - `NaN` or `Infinity` literals anywhere in the JSON;
  - JSON nested 128 or more levels deep;
  - non-ASCII decimal digits in score keys.
- **No input sanitising.** Special-token text in a request (for example, `<|im_start|>` in the state) is
  tokenized as the special token, as the reference does. Parity with the reference depends on this.
- **The Neural Engine split is off by default.** It needs its own weight set, Python with coremltools when you
  build that set, and about 8 s more startup. Its error is 2–3× the GPU-only path's, but still about 4× inside
  the 0.01 tolerance.
- **Power and heat.** All numbers were measured on AC power with low-power mode off. Expect slower, noisier
  numbers on battery or on a hot machine.
- **State cache.** Entry size is not capped yet: for a state near 8,192 tokens, the K/V alone is about 256 MB.

## License

- decide is [Apache-2.0](LICENSE), © 2026 Ezulabs. Its Metal device layer (`crates/decide-metal`) is derived
  from MicroLLM (Apache-2.0, Ezulabs).
- The fast GEMM and attention kernels in `kernels/` are derived from Apple MLX's steel kernels (MIT). See
  [`NOTICE`](NOTICE).
- **The model weights are not in this repository.** Download them from
  [internlm/Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B). They are Apache-2.0, and
  the model is fine-tuned from Qwen3.5-4B. The scripts in `reference/` call the checkpoint's own `inference.py`,
  which is not redistributed here.
- Not affiliated with InternLM or Apple.
