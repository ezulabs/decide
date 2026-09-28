"""InternLM's own inference.py on PyTorch MPS in fp16: the official path.

The engine is `inf.DecisionEngine(checkpoint=MODEL, device="mps", dtype="float16")`, used as
shipped (MPS bf16 is about 2x slower; fp16 is the fair bar). inference.py synchronizes only for
CUDA, so its own `timing` field is ignored; the spans (decide bench's, see SPAN) are:
- forward: from HFBackend.encode returning (token ids in memory) to the candidate logits on
  the CPU: the ids' copy to the device, the forward with logits_to_keep = the answer rows,
  then one gather of the answer rows' candidate logits and its .float().cpu() readback (which
  waits for the GPU), as decide's forward ends with its logits' readback. The candidate ids
  come from bench_common.Pipeline.encode, run before the timed windows;
- e2e: request JSON text -> response JSON string around predict, with a
  torch.mps.synchronize() before and after predict. It includes that one extra readback
  (well under a millisecond) besides predict's own per-field softmax + .cpu().

No CPU fallback: PYTORCH_ENABLE_MPS_FALLBACK must be unset, and an op MPS lacks is reported
as an error, never worked around (a fallback would not be the official path).

Run from the repo root under the model lock, and the GPU lock too when timing:

    lockf -k /tmp/decide-model.lock lockf -k /tmp/decide-gpu.lock \\
        reference/.venv/bin/python reference/bench_torch_mps.py --requests tests/fixtures/requests/core

--race reads one request (a file, or stdin) and prints one JSON object (see bench_common).
"""
import argparse
import gc
import json
import os
import sys
import time
from itertools import chain

import bench_common as bc

LABEL = "InternLM inference.py on PyTorch MPS (fp16, full prompt; HF logits_to_keep = the answer rows)"
SPAN = ("token ids in memory (HFBackend.encode has returned the tokenizer's CPU tensors) -> candidate logits on "
        "the CPU: the ids' copy to MPS, the forward over the full prompt with logits_to_keep = the answer rows (full "
        "vocabulary on those rows), then one gather of the answer rows' candidate logits and its .float().cpu() "
        "readback, which waits for the GPU. The candidate ids come from bench_common.Pipeline.encode, run before the "
        "timed windows. Excludes tokenization and predict's head (its per-field softmax on MPS + .cpu(), "
        "calibration), which follow in e2e.")


class TimedForward:
    """Stands in for HFBackend.model; HFBackend.encode is wrapped to start the clock when the ids
    are ready. Ends the forward span once the answer rows' candidate logits (`cands`, a device
    index tensor set before each request) are on the CPU, and keeps them and the ids it ran for
    the fidelity check after the timed window."""

    def __init__(self, backend, torch):
        self.model, self.torch = backend.model, torch
        self.t_ids, self.ms, self.cands, self.picked, self.input_ids = None, None, None, None, None
        encode = backend.encode

        def timed_encode(row):
            result = encode(row)
            self.t_ids = time.perf_counter()
            return result

        backend.encode = timed_encode
        backend.model = self

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        picked = out.logits[0][:, self.cands].float().cpu()  # [K, |candidates|]; .cpu() waits for the GPU
        self.ms = (time.perf_counter() - self.t_ids) * 1e3
        self.picked, self.input_ids = picked, kwargs.get("input_ids")
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bc.add_protocol_args(ap)
    args = ap.parse_args()
    name = "torch-mps"
    race_out = bc.RaceOutput() if args.race else None
    machine = None
    try:
        requests = bc.read_requests(args.requests, args.race)
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK is set: a CPU fallback is not the official path")
        t0 = time.perf_counter()
        inf = bc.inference()
        import torch

        imports_s = time.perf_counter() - t0
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        machine = bc.machine_state()
        t0 = time.perf_counter()
        engine = inf.DecisionEngine(checkpoint=bc.MODEL, device="mps", dtype="float16")
        load_s = time.perf_counter() - t0
        dtypes: dict[str, int] = {}
        for p in engine.backend.model.parameters():
            key = str(p.dtype).removeprefix("torch.")
            dtypes[key] = dtypes.get(key, 0) + 1
        timed = TimedForward(engine.backend, torch)
        pipe = bc.Pipeline(tok=engine.tokenizer)  # the same tokenizer: the candidate ids for the readback

        def run(text: str) -> bc.Sample:
            # Before the timed windows: the candidate ids the forward span reads back.
            enc = pipe.encode(json.loads(text))
            union = sorted(set(chain.from_iterable(enc.candidates.values())))
            timed.cands = torch.tensor(union, device="mps")
            torch.mps.synchronize()
            t = time.perf_counter()
            response = engine.predict(json.loads(text))
            torch.mps.synchronize()
            json.dumps(response, ensure_ascii=False)  # the response JSON string: the end of e2e
            e2e_ms = (time.perf_counter() - t) * 1e3
            # Outside the timed window: the ids check, and the candidate logits for max |Δlogit|.
            if timed.input_ids[0].tolist() != enc.ids:
                raise AssertionError("inference.py ran different token ids than bench_common.Pipeline")
            col = {t: j for j, t in enumerate(union)}
            rows = timed.picked.tolist()
            logits = {f: [rows[i][col[t]] for t in enc.candidates[f]] for i, f in enumerate(enc.compiled.fields)}
            return bc.Sample(response, timed.ms, e2e_ms, logits, enc.ids, len(enc.ids), "full")

        info = {"imports_s": imports_s, "param_dtypes": dtypes, "device": "mps", "dtype": "float16",
                "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}

        def extras() -> dict:
            return {"mps_current_allocated_gb": torch.mps.current_allocated_memory() / 1e9,
                    "mps_driver_allocated_gb": torch.mps.driver_allocated_memory() / 1e9}

        def unload() -> None:
            timed.model = timed.picked = timed.input_ids = None
            engine.backend.model = None
            gc.collect()
            torch.mps.empty_cache()

        return bc.drive(name, run, requests, args, load_s=load_s, machine_before=machine, info=info,
                        label=LABEL, spans={"forward": SPAN}, race_out=race_out, extras=extras, unload=unload)
    except Exception as exc:  # noqa: BLE001  reported, never worked around
        return bc.fail(name, race_out, exc, machine)


if __name__ == "__main__":
    sys.exit(main())
