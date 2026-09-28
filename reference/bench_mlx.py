"""mlx-lm baselines for Intern-Decision-4B decision scoring.

--impl port     The obvious port (engine mlx-port; `shipped`, its old name, is accepted): what a
                user would write with mlx-lm. `mlx_lm.load` in the checkpoint's dtypes (bf16;
                A_log and the gated-norm weights stay fp32 as stored), one forward over the
                full prompt with the full-vocabulary head, then the logit rows before each
                <decision> marker and each field's candidate ids.
--impl best     The honest bar: mlx-lm with decide's algorithmic shortcuts.
                - precision (--dtypes, see DTYPES): by default the plan's rule, every bf16
                  parameter fp16 except A_log. The others keep decide's fp32 parameters: norm
                  weights (the exact 1 + w from the checkpoint; mlx-lm adds the 1 in bf16),
                  dt_bias, A_log (norms); + conv weights, so the conv output and recurrence
                  inputs are fp32 (exempt); + an fp32 residual stream (decide). Norm outputs are
                  cast back to fp16, so every GEMM stays fp16 x fp16. They cost 0.3-3% and were
                  not more faithful on card/ticket6, so plan stays the default (the fair bar);
                - a snapshot of the constant prompt prefix (107 tokens, recomputed at startup);
                  each request gets a fresh cache holding copies of the snapshot's state;
                - tail truncation: only rows 107..last answer row run;
                - last-layer row pruning, as decide: layer 31 (full attention) computes K/V for
                  every row, but q/gate, attention, o_proj and the MLP only for the answer rows
                  (single-token queries at their absolute positions, each masked to keys 0..=pos);
                - a candidate-only head: the post-norm answer rows against
                  embed_tokens.weight[candidate ids], in fp32.
                A request whose ids don't start with the prefix runs a full (truncated) prefill.

Both compile prompts with InternLM's own code (bench_common.Pipeline: validate_request,
compile_row, the chat template, AutoTokenizer), so the ids are the gold's, and both build the
response with DecisionEngine.predict's post-processing. Both pin the weights the way mlx-lm's
own generate path does (mx.set_wired_limit to the recommended working set).

Timing spans mirror decide bench's (bench_common; the exact text is in SPANS, printed and
saved): forward = token ids in memory -> candidate logits on the CPU, including route choice,
cache set-up, MLX's graph building and dispatch, mx.eval + mx.synchronize() and the readback;
e2e = request JSON text -> response JSON string. Run from the repo root; every run that loads
the model holds the model lock, and timing runs the GPU lock as well:

    lockf -k /tmp/decide-model.lock lockf -k /tmp/decide-gpu.lock \\
        reference/.venv/bin/python reference/bench_mlx.py --impl best --requests tests/fixtures/requests/core

--race reads one request (a file, or stdin) and prints one JSON object (see bench_common).
"""
import argparse
import gc
import json
import sys
import time
from functools import partial
from itertools import chain

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import bench_common as bc


def param_dtypes(model) -> dict:
    from mlx.utils import tree_flatten

    out: dict[str, int] = {}
    for _, p in tree_flatten(model.parameters()):
        key = str(p.dtype).removeprefix("mlx.core.")
        out[key] = out.get(key, 0) + 1
    return out


# mlx-best --dtypes decide: parameters kept fp32 (decide keeps them fp32), not cast to fp16.
KEEP_FP32 = ("norm.weight", "dt_bias", "conv1d.weight", "A_log")
# The zero-centred norms, whose (1 + w) mlx-lm's sanitize adds in bf16.
ZERO_CENTRED = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")


class F32Norm(nn.Module):
    """RMSNorm with an fp32 weight; the output is cast to `out` (fp16 where it feeds a GEMM, so
    the GEMMs stay fp16 as decide's do; None keeps fp32)."""

    def __init__(self, weight, eps, out):
        super().__init__()
        self.weight, self.eps, self.out = weight, eps, out

    def __call__(self, x):
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        return y if self.out is None else y.astype(self.out)


@partial(mx.compile, shapeless=True)
def _gated_f16(gate, x):
    return (nn.silu(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(mx.float16)


class GatedNormF16(nn.Module):
    """Qwen3NextRMSNormGated with its output cast to fp16 (out_proj's input stays fp16)."""

    def __init__(self, norm):
        super().__init__()
        self.weight, self.eps = norm.weight, norm.eps

    def __call__(self, h, gate):
        return _gated_f16(gate, mx.fast.rms_norm(h, self.weight, self.eps))


SPANS = {
    "port":"token ids in memory (a Python list) -> candidate logits on the CPU (Python floats): mx.array of the "
               "ids, the full forward over every token with the full-vocabulary head (graph building and dispatch on "
               "the CPU included), the gather of each answer row's candidate logits, mx.eval + mx.synchronize(), and "
               "the readback (.tolist()). Excludes tokenization and the head (softmax, calibration). Equals the "
               "response's timing.inference_ms before rounding.",
    "best": "token ids in memory (a Python list) -> candidate logits on the CPU (Python floats): route choice (prefix "
            "id check), a fresh cache holding the snapshot's state, mx.array of the span 107..last answer row, the "
            "forward over that span with the last layer pruned to the answer rows (graph building and dispatch on the "
            "CPU included), the candidate-only head (fp32), mx.eval + mx.synchronize(), and the readback. Excludes tokenization and the "
            "head (softmax, calibration). Equals the response's timing.inference_ms before rounding.",
}


DTYPES = {  # mlx-best's precision policies; every GEMM is fp16 x fp16 in all of them
    "plan": "fp16 except A_log",
    "norms": "fp16; exact fp32 norm weights, fp32 dt_bias and A_log",
    "exempt": "fp16; exact fp32 norm weights, fp32 conv, dt_bias and A_log",
    "decide": "fp16 GEMMs; exact fp32 norm weights, fp32 conv, dt_bias, A_log and residual stream",
}
LABELS = {
    "port": "mlx-lm obvious port (bf16, full prompt, full-vocabulary head)",
    "best": "mlx-lm best effort ({dtypes}; prefix snapshot, tail truncation, last layer pruned to the answer rows, "
            "candidate-only head)",
}


class Port:
    """The obvious port: what a user would write with mlx-lm."""
    name = "mlx-port"

    def __init__(self, pipe: bc.Pipeline):
        import mlx_lm

        self.model, _ = mlx_lm.load(str(bc.MODEL))

    def forward(self, enc: bc.Encoded):
        fields = enc.compiled.fields
        mx.synchronize()
        t0 = time.perf_counter()
        logits = self.model(mx.array(enc.ids)[None])  # [1, T, V]: every token, full vocabulary
        picked = [logits[0, p, mx.array(enc.candidates[f])].astype(mx.float32) for p, f in zip(enc.positions, fields)]
        mx.eval(picked)
        mx.synchronize()
        out = {f: x.tolist() for f, x in zip(fields, picked)}
        ms = (time.perf_counter() - t0) * 1e3
        return out, ms, len(enc.ids), "full"


class Best:
    name = "mlx-best"

    def __init__(self, pipe: bc.Pipeline, prune: bool = True, dtypes: str = "plan"):
        import mlx_lm
        from mlx.utils import tree_flatten, tree_map_with_path

        # Lazy load, so the bf16 weights are never all resident beside their fp16 copies.
        self.model, _ = mlx_lm.load(str(bc.MODEL), lazy=True)
        self.dtypes = dtypes
        keep = {"plan": ("A_log",), "norms": ("norm.weight", "dt_bias", "A_log")}.get(dtypes, KEEP_FP32)

        def cast(path, p):
            if path.endswith(keep):
                return p.astype(mx.float32)
            return p.astype(mx.float16) if p.dtype == mx.bfloat16 else p

        self.model.update(tree_map_with_path(cast, self.model.parameters()))
        self.inner = self.model.language_model.model  # the stack + final norm, no head
        if dtypes != "plan":
            self.decide_norms()
        params = [p for _, p in tree_flatten(self.model.parameters())]
        for i in range(0, len(params), 16):
            mx.eval(params[i:i + 16])
        self.embed = self.inner.embed_tokens.weight
        self.prune = prune
        assert not self.inner.layers[-1].is_linear, "row pruning needs a full-attention last layer"
        self.prefix = pipe.prefix_ids()
        cache = self.model.make_cache()
        self.hidden(mx.array(self.prefix)[None], cache, None)
        # KVCache.state slices the cache's 256-row buffers to the 107 used rows; ArraysCache.state
        # is its own [conv, recurrent] list. Keep compact private copies of both.
        self.snap = [[mx.contiguous(a) for a in c.state] for c in cache]
        mx.eval(self.snap)
        self.snap_host = [[np.array(a) for a in s] for s in self.snap]

    def decide_norms(self):
        """decide's norm numerics: every zero-centred norm gets the exact fp32 1 + w, from the
        checkpoint's raw w (mlx-lm's sanitize added the 1 in bf16), and outputs fp16 where it
        feeds a GEMM (the final norm, which feeds only the fp32 head, stays fp32); the gated norm
        outputs fp16 for out_proj. The residual stream is made fp32 in `hidden`."""
        raw = {}
        for f in sorted(bc.MODEL.glob("model-language-*.safetensors")):
            for k, v in mx.load(str(f)).items():
                if k.endswith(ZERO_CENTRED) or k == "model.language_model.norm.weight":
                    raw[k.removeprefix("model.language_model.")] = v
        norm = lambda key, old, out: F32Norm(raw.pop(key).astype(mx.float32) + 1.0, old.eps, out)  # noqa: E731
        f16 = mx.float16
        for i, layer in enumerate(self.inner.layers):
            layer.input_layernorm = norm(f"layers.{i}.input_layernorm.weight", layer.input_layernorm, f16)
            layer.post_attention_layernorm = norm(f"layers.{i}.post_attention_layernorm.weight",
                                                  layer.post_attention_layernorm, f16)
            if layer.is_linear:
                layer.linear_attn.norm = GatedNormF16(layer.linear_attn.norm)
            else:
                a = layer.self_attn
                a.q_norm = norm(f"layers.{i}.self_attn.q_norm.weight", a.q_norm, f16)
                a.k_norm = norm(f"layers.{i}.self_attn.k_norm.weight", a.k_norm, f16)
        self.inner.norm = norm("norm.weight", self.inner.norm, None)
        assert not raw, f"unused norm weights: {sorted(raw)}"

    def fresh_cache(self):
        """A new cache per request, holding new handles to the snapshot's arrays. KVCache's first
        update concatenates (a new array; its setter leaves no spare capacity to write into) and
        ArraysCache rebinds entries of its own new list, so a request never writes the snapshot."""
        cache = self.model.make_cache()
        for c, s in zip(cache, self.snap):
            c.state = [mx.array(a) for a in s]
        return cache

    def hidden(self, x, cache, rows):
        """Post-norm hidden states of the span `x` ([1, S] ids), continuing `cache` (None: from
        position 0). rows=None runs every layer on every row (Qwen3_5TextModel.__call__, spelt
        out); otherwise the last layer is pruned to `rows` (span-relative) and [1, K, H] returns."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        inner = self.inner
        h = inner.embed_tokens(x)
        if self.dtypes == "decide":
            h = h.astype(mx.float32)  # the residual stream is fp32, as decide's
        c = cache if cache is not None else [None] * len(inner.layers)
        fa_mask = create_attention_mask(h, c[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, c[inner.ssm_idx])
        n = len(inner.layers) if rows is None else len(inner.layers) - 1
        for layer, lc in zip(inner.layers[:n], c[:n]):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=lc)
        if rows is not None:
            h = self.last_layer_pruned(h, c[-1], rows)
        return inner.norm(h)

    def last_layer_pruned(self, h, cache, rows):
        """Layer 31 (DecoderLayer with Qwen3NextAttention) for the answer rows only: its input norm
        and K/V (k_norm, RoPE, cache update) on every row, as the causal attention needs them;
        q/gate, q_norm, RoPE at each row's absolute position, attention over keys 0..=pos, o_proj,
        the residual and the MLP on the K answer rows."""
        layer = self.inner.layers[-1]
        attn = layer.self_attn
        B, S, _ = h.shape
        offset = cache.offset if cache is not None else 0
        xn = layer.input_layernorm(h)
        keys = attn.k_norm(attn.k_proj(xn).reshape(B, S, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = attn.v_proj(xn).reshape(B, S, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        keys = attn.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        idx = mx.array(rows)
        K = len(rows)
        queries, gate = mx.split(attn.q_proj(xn[:, idx]).reshape(B, K, attn.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, K, -1)
        queries = attn.q_norm(queries)  # [1, K, heads, hd]
        pos = mx.array([offset + r for r in rows])
        rope = attn.rope
        # K single-token queries: the batch axis carries one RoPE offset per query.
        queries = mx.fast.rope(queries.transpose(1, 2, 0, 3), rope.dims, traditional=rope.traditional,
                               base=rope.base, scale=rope.scale, offset=pos).transpose(2, 1, 0, 3)  # [1, heads, K, hd]
        mask = mx.arange(keys.shape[2])[None] <= pos[:, None]  # [K, keys]: query i sees keys 0..=pos_i
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, K, -1)
        hs = h[:, idx] + attn.o_proj(out * mx.sigmoid(gate))
        return hs + layer.mlp(layer.post_attention_layernorm(hs))

    def snapshot_intact(self) -> bool:
        return all(np.array_equal(np.array(a), h) for s, hs in zip(self.snap, self.snap_host) for a, h in zip(s, hs))

    def forward(self, enc: bc.Encoded, route: str | None = None, prune: bool | None = None):
        ids, at = enc.ids, enc.positions
        prune = self.prune if prune is None else prune
        mx.synchronize()
        t0 = time.perf_counter()
        last = max(at)
        union = sorted(set(chain.from_iterable(enc.candidates.values())))
        n = len(self.prefix)
        if route != "full" and ids[:n] == self.prefix and min(at) >= n:
            cache, start, route = self.fresh_cache(), n, "snapshot"
        else:
            cache, start, route = None, 0, "full"
        rel = [p - start for p in at]
        if prune:
            rows = self.hidden(mx.array(ids[start:last + 1])[None], cache, rel)[0]  # [K, H] post-norm
        else:
            rows = self.hidden(mx.array(ids[start:last + 1])[None], cache, None)[0][mx.array(rel)]
        rows = rows.astype(mx.float32)
        logits = rows @ self.embed[mx.array(union)].astype(mx.float32).T  # [K, |union|]
        mx.eval(logits)
        mx.synchronize()
        r = logits.tolist()
        col = {t: j for j, t in enumerate(union)}
        out = {f: [r[i][col[t]] for t in enc.candidates[f]] for i, f in enumerate(enc.compiled.fields)}
        ms = (time.perf_counter() - t0) * 1e3
        return out, ms, last + 1 - start, route


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["port", "best", "shipped"], required=True,
                    help="port: the obvious mlx-lm port (shipped: its old name); best: decide's shortcuts")
    ap.add_argument("--self-check", action="store_true",
                    help="best: before timing, report max |Δlogit| of the snapshot path against a full "
                         "prefill, and of the pruned last layer against the full one")
    ap.add_argument("--dtypes", choices=list(DTYPES), default="plan",
                    help="best: the precision policy (see DTYPES); default plan, the fastest (measured: "
                         "norms +0.3/+1.2%%, exempt +2.5/+2.7%%, decide +2.4/+3.1%% forward p50 on card/ticket6), "
                         "every policy within the parity criteria on 8 gold requests")
    ap.add_argument("--no-prune", action="store_true",
                    help="best: run the last layer on every row (an A/B switch; not the honest bar)")
    bc.add_protocol_args(ap)
    args = ap.parse_args()
    if args.impl == "shipped":
        args.impl = "port"
    name = f"mlx-{args.impl}"
    race_out = bc.RaceOutput() if args.race else None
    machine = None
    try:
        requests = bc.read_requests(args.requests, args.race)
        t0 = time.perf_counter()
        bc.inference()  # InternLM's module (torch, transformers): process start-up, not load time
        import mlx_lm  # noqa: F401
        imports_s = time.perf_counter() - t0
        machine = bc.machine_state()
        t0 = time.perf_counter()
        pipe = bc.Pipeline()
        engine = (Best(pipe, prune=not args.no_prune, dtypes=args.dtypes) if args.impl == "best"
                  else Port(pipe))
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        load_s = time.perf_counter() - t0

        def run(text: str) -> bc.Sample:
            t = time.perf_counter()
            enc = pipe.encode(json.loads(text))
            logits, fwd_ms, computed, route = engine.forward(enc)
            response = pipe.respond(enc, logits, fwd_ms, engine.name)
            json.dumps(response, ensure_ascii=False)  # the response JSON string: the end of e2e
            e2e_ms = (time.perf_counter() - t) * 1e3
            return bc.Sample(response, fwd_ms, e2e_ms, logits, enc.ids, computed, route)

        info = {"impl": args.impl, "last_layer_pruned": args.impl == "best" and not args.no_prune,
                "dtypes": args.dtypes if args.impl == "best" else "checkpoint (bf16; A_log, gated norms fp32)",
                "imports_s": imports_s, "param_dtypes": param_dtypes(engine.model),
                "device": str(mx.default_device()), "wired_limit_bytes": mx.device_info()["max_recommended_working_set_size"]}
        self_check = {}
        if args.impl == "best":
            info["prefix_len"] = len(engine.prefix)
            if args.self_check:
                for req_name, text in requests:
                    enc = pipe.encode(json.loads(text))
                    snap, *_ = engine.forward(enc)
                    full, *_ = engine.forward(enc, route="full")
                    unpruned, *_ = engine.forward(enc, prune=not engine.prune)
                    d = lambda x, y: max(abs(a - b) for f in x for a, b in zip(x[f], y[f]))  # noqa: E731
                    self_check[req_name] = {"snapshot_vs_full_prefill": d(snap, full),
                                            "pruned_vs_unpruned": d(snap, unpruned)}
                    print(f"self-check {req_name}: max |Δlogit| snapshot vs full prefill {d(snap, full):.4f}, "
                          f"pruned vs unpruned last layer {d(snap, unpruned):.4f}", file=sys.stderr)
                # Which dtypes the GEMMs see (input x weight -> output), over one forward.
                census: dict[str, int] = {}
                linear_call = nn.Linear.__call__

                def counting(layer, x):
                    y = linear_call(layer, x)
                    key = f"{x.dtype} x {layer.weight.dtype} -> {y.dtype}".replace("mlx.core.", "")
                    census[key] = census.get(key, 0) + 1
                    return y

                nn.Linear.__call__ = counting
                try:
                    engine.forward(pipe.encode(json.loads(requests[0][1])))
                finally:
                    nn.Linear.__call__ = linear_call
                self_check["linear_dtypes"] = census
                print(f"self-check: Linear calls by dtype over one forward: {census}", file=sys.stderr)

        def extras() -> dict:
            out = {"peak_memory_gb": mx.get_peak_memory() / 1e9}
            if args.impl == "best":
                out["snapshot_intact"] = engine.snapshot_intact()
                if self_check:
                    out["self_check_max_dlogit"] = self_check
            return out

        def unload() -> None:
            engine.__dict__.clear()
            gc.collect()
            mx.clear_cache()

        return bc.drive(name, run, requests, args, load_s=load_s, machine_before=machine, info=info,
                        label=LABELS[args.impl].format(dtypes=DTYPES[args.dtypes]), spans={"forward": SPANS[args.impl]}, race_out=race_out, extras=extras, unload=unload)
    except Exception as exc:  # noqa: BLE001  reported, never worked around
        return bc.fail(name, race_out, exc, machine)


if __name__ == "__main__":
    sys.exit(main())
