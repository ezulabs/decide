"""Gold outputs for decide, from InternLM's own model code on the CPU.

--mode fp32 (the parity target): the released bf16 weights with every op in
    fp32. fp32 weights would need 18 GB, so Linear weights stay bf16 and are
    upcast per call; every other parameter is read as stored (A_log and the
    gated-norm weights are fp32 in the checkpoint) and converted to fp32 once,
    and the embedding output is upcast so the residual stream is fp32 throughout.
--mode bf16: InternLM's DecisionEngine exactly as shipped, on CPU in bf16.
    Not a target -- it measures how noisy the official path itself is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "Intern-Decision-4B"
REQUESTS = ROOT / "tests" / "fixtures" / "requests"
GOLD = ROOT / "tests" / "fixtures" / "gold"
HIDDEN = ROOT / "reference" / "out" / "hidden"
sys.path.insert(0, str(MODEL))
import inference as inf  # noqa: E402  InternLM's module, shipped beside the weights


def load_fp32_compute():
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa").eval()
    lm = model.model.language_model
    # The checkpoint stores A_log and the gated-norm weights in fp32, and dtype=bfloat16 rounds
    # them (that alone moves the final hidden state by ~7e-3 relative L2). So every non-Linear
    # parameter is re-read from the checkpoint as stored, then made fp32.
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}
    for module_name, module in lm.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        for name, p in module.named_parameters(recurse=False):
            key = ".".join(filter(None, ("model.language_model", module_name, name)))
            if weight_map[key] not in shards:
                shards[weight_map[key]] = safe_open(str(MODEL / weight_map[key]), framework="pt")
            p.data = shards[weight_map[key]].get_tensor(key).float()
        for name, b in list(module.named_buffers(recurse=False)):
            if b is not None and b.is_floating_point():
                setattr(module, name, b.float())

    def linear_fp32(self, x):
        return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())

    nn.Linear.forward = linear_fp32
    embed_forward = lm.embed_tokens.forward
    lm.embed_tokens.forward = lambda ids: embed_forward(ids).float()
    return tok, model


def response_from_logits(row, compiled, logits_by_field, length, inference_ms):
    """DecisionEngine.predict's post-processing, line for line, from given logits."""
    answers = {}
    for field in compiled.fields:
        question = row["questions"][field]
        options = inf._options(question)
        values = [value for value, _ in options]
        probabilities = torch.softmax(torch.tensor(logits_by_field[field], dtype=torch.float32), dim=-1).tolist()
        probs = dict(zip(values, probabilities))
        best = min(values, key=lambda value: (-probs[value], value))
        kind = question["type"]
        answer = {"type": kind, "probabilities": probs, "confidence": probs[best]}
        if kind == "noul":
            answer["noul"] = probs["yes"]
        elif kind == "score":
            answer["score"] = sum(float(value) * probs[value] for value in values)
            answer["legend"] = dict(options)
        else:
            answer["choice"] = best
        answers[field] = answer
    result = {"answers": answers, "usage": {"input_tokens": length, "output_tokens": len(answers)},
              "timing": {"inference_ms": round(inference_ms, 2)}}
    result = inf.scale_result(result, inf.DEFAULT_TEMPERATURE)
    for answer in result["answers"].values():
        answer["source"] = "local"
        answer["decision"] = inf.argmax(answer["probabilities"])
    result["usage"]["decision_count"] = len(result["answers"])
    result["model"] = inf.MODEL_NAME
    result["backend"] = "hf"
    return result


def encode(tok, request):
    row = inf.validate_request(request)
    compiled = inf.compile_row(row)
    text = tok.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False, add_vision_id=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    marker = tok.convert_tokens_to_ids(inf.DECISION_TOKEN)
    positions = (ids[0] == marker).nonzero().flatten() - 1
    assert len(positions) == len(compiled.fields) and bool((positions >= 0).all())
    cands = {f: [tok.encode(s, add_special_tokens=False)[0] for s in compiled.symbols[f]] for f in compiled.fields}
    return row, compiled, ids, positions.tolist(), cands


def gold_fp32(tok, model, name, request, dump_hidden, audit):
    row, compiled, ids, positions, cands = encode(tok, request)
    lm = model.model.language_model
    hooks, captured = [], []
    if dump_hidden:
        for i, layer in enumerate(lm.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, a, out, i=i: captured.append((i, (out[0] if isinstance(out, tuple) else out).detach()))))
    for mod_name, mod in lm.named_modules():
        if isinstance(mod, nn.Linear):
            def rec(m, args, out, mod_name=mod_name):
                a = audit.setdefault(mod_name, {"in": 0.0, "out": 0.0})
                a["in"] = max(a["in"], float(args[0].abs().max()))
                a["out"] = max(a["out"], float(out.abs().max()))
            hooks.append(mod.register_forward_hook(rec))
    t0 = time.perf_counter()
    with torch.inference_mode():
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        assert hidden.dtype == torch.float32, hidden.dtype
        emb = model.get_input_embeddings().weight
        logits = {f: (hidden[positions[i]] @ emb[cands[f]].float().T).tolist()
                  for i, f in enumerate(compiled.fields)}
    ms = (time.perf_counter() - t0) * 1000
    for h in hooks:
        h.remove()
    if dump_hidden:
        out = HIDDEN / name
        out.mkdir(parents=True, exist_ok=True)
        for i, t in captured:
            assert t.dtype == torch.float32
            t[0].contiguous().numpy().astype("<f4").tofile(out / f"layer_{i:02d}.f32")
        hidden.contiguous().numpy().astype("<f4").tofile(out / "final.f32")
        (out / "meta.json").write_text(json.dumps({"T": int(ids.shape[1]), "H": int(hidden.shape[1]),
                                                   "layers": len(captured), "input_ids": ids[0].tolist()}))
    return {"name": name, "mode": "fp32", "input_ids": ids[0].tolist(), "positions": positions,
            "fields": list(compiled.fields), "candidate_ids": cands, "logits": logits,
            "response": response_from_logits(row, compiled, logits, int(ids.shape[1]), ms), "seconds": ms / 1000}


def gold_bf16(engine, tok, name, request):
    stash = {}
    score = engine.backend.score

    def capture(row):
        compiled, logits, length, ms = score(row)
        stash["logits"] = logits.float().clone()
        return compiled, logits, length, ms

    engine.backend.score = capture
    t0 = time.perf_counter()
    response = engine.predict(request)
    seconds = time.perf_counter() - t0
    engine.backend.score = score
    row, compiled, ids, positions, cands = encode(tok, request)
    logits = {f: stash["logits"][i, cands[f]].tolist() for i, f in enumerate(compiled.fields)}
    return {"name": name, "mode": "bf16", "input_ids": ids[0].tolist(), "positions": positions,
            "fields": list(compiled.fields), "candidate_ids": cands, "logits": logits,
            "response": response, "seconds": seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--set", choices=["core", "calib", "all"], default="all")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sets = ["core", "calib"] if args.set == "all" else [args.set]
    if args.mode == "fp32":
        tok, model = load_fp32_compute()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        engine = inf.DecisionEngine(checkpoint=MODEL, device="cpu", dtype="bfloat16")
    audit: dict = {}
    for s in sets:
        out = GOLD / args.mode / s
        out.mkdir(parents=True, exist_ok=True)
        for path in sorted((REQUESTS / s).glob("*.json")):
            request = json.loads(path.read_text(encoding="utf-8"))
            if args.mode == "fp32":
                g = gold_fp32(tok, model, path.stem, request, dump_hidden=(s == "core"), audit=audit)
            else:
                g = gold_bf16(engine, tok, path.stem, request)
            (out / path.name).write_text(json.dumps(g, indent=1))
            dec = {f: a["decision"] for f, a in g["response"]["answers"].items()}
            print(f"{args.mode} {s}/{path.stem}: {len(g['input_ids'])} tok, {g['seconds']:.1f}s, {dec}", flush=True)
    if args.mode == "fp32":
        (GOLD / "fp32" / "activation_audit.json").write_text(json.dumps(audit, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
