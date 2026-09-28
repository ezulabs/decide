"""Build decide's per-layer ANE models from a converted weight set.

For layers 0..layers-1, the ANE's slice of the MLP is the first `n_inter` SwiGLU
intermediate dims: partial = swiglu(h·Wg^T, h·Wu^T)·Wd[:, :n_inter]^T, fp16, fixed shape
[chunk, 2560] -> [chunk, 2560]. `L{i}.w_gu` is gate/up interleaved in 16-row blocks
(block b: rows [32b, 32b+16) gate, [32b+16, 32b+32) up, intermediate dims [16b, 16b+16)),
so intermediate dims [0, n_inter) are rows [0, 2·n_inter); they are de-interleaved here.
SwiGLU is written as g/(1+exp(-g))·u (the ANE's native silu is a low-precision LUT), and
w_up is pre-scaled by `out_scale` so the partial is out_scale× larger than the true value
(the ANE's absolute error floor); the engine adds partial/out_scale.

Output: <out>/L{i}.mlmodelc (compiled) and <out>/ane-models.json. Runs with coremltools
(a separate venv; not reference/.venv). Called by `decide ane-convert`, or directly:

  .venv-ane/bin/python reference/ane_convert.py --weights weights --out weights/ane
"""
import argparse, hashlib, json, os, shutil, sys, tempfile, time
import numpy as np
import logging
logging.getLogger("coremltools").setLevel(logging.ERROR)
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

HIDDEN, INTER = 2560, 9216


def load_tensor(weights_dir, index, name):
    t = index["tensors"][name]
    assert t["dtype"] == "f16", (name, t)
    n = int(np.prod(t["shape"]))
    mm = np.memmap(os.path.join(weights_dir, t["blob"]), dtype=np.float16, mode="r", offset=t["offset"], shape=(n,))
    return np.array(mm).reshape(t["shape"])


def build(w_gu, w_down, n_inter, chunk, out_scale):
    blocks = w_gu[: 2 * n_inter].reshape(n_inter // 16, 2, 16, HIDDEN)
    w_gate = np.ascontiguousarray(blocks[:, 0].reshape(n_inter, HIDDEN))
    w_up = np.ascontiguousarray(blocks[:, 1].reshape(n_inter, HIDDEN))
    w_up = (w_up.astype(np.float32) * out_scale).astype(np.float16)
    w_down_slice = np.ascontiguousarray(w_down[:, :n_inter])

    @mb.program(input_specs=[mb.TensorSpec(shape=(chunk, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS17)
    def prog(x):
        g = mb.linear(x=x, weight=w_gate)
        u = mb.linear(x=x, weight=w_up)
        den = mb.add(x=mb.exp(x=mb.mul(x=g, y=np.float16(-1.0))), y=np.float16(1.0))
        a = mb.mul(x=mb.real_div(x=g, y=den), y=u)
        return mb.linear(x=a, weight=w_down_slice, name="y")

    return ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=(chunk, HIDDEN), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="the converted weight set (decide-index.json + blobs)")
    ap.add_argument("--out", required=True, help="where L{i}.mlmodelc and ane-models.json go")
    ap.add_argument("--n-inter", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out-scale", type=float, default=64.0)
    ap.add_argument("--layers", type=int, default=31, help="models for layers 0..layers-1 (the last layer stays on the GPU)")
    ap.add_argument("--only", type=int, default=None, help="build one layer only (for tests)")
    args = ap.parse_args()
    assert args.n_inter % 128 == 0 and 0 < args.n_inter < INTER, "n_inter must be a multiple of 128 below 9216"

    index = json.load(open(os.path.join(args.weights, "decide-index.json")))
    assert index["format"] == "decide-weights-v1", index["format"]
    os.makedirs(args.out, exist_ok=True)
    layers = [args.only] if args.only is not None else list(range(args.layers))
    manifest_path = os.path.join(args.out, "ane-models.json")
    report = {"format": "decide-ane-models-v1", "n_inter": args.n_inter, "chunk_rows": args.chunk, "out_scale": args.out_scale,
              "act": "exp", "layers": args.layers, "models": {}, "coremltools": ct.__version__}
    if args.only is not None and os.path.exists(manifest_path):
        # Rebuilding one layer updates the manifest in place; a stale entry
        # would fail the engine's sha256 check, a missing one its layer check.
        old = json.load(open(manifest_path))
        same = all(old.get(k) == report[k] for k in ("format", "n_inter", "chunk_rows", "out_scale", "layers"))
        assert same, f"{manifest_path} was written for other parameters; rebuild every layer"
        report["models"] = old.get("models", {})
    t_all = time.perf_counter()
    for l in layers:
        t0 = time.perf_counter()
        w_gu = load_tensor(args.weights, index, f"L{l}.w_gu")
        w_down = load_tensor(args.weights, index, f"L{l}.w_down")
        model = build(w_gu, w_down, args.n_inter, args.chunk, args.out_scale)
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            pkg = os.path.join(tmp, f"L{l}.mlpackage")
            model.save(pkg)
            compiled = ct.models.utils.compile_model(pkg)  # a .mlmodelc in a temp location
            dst = os.path.join(args.out, f"L{l}.mlmodelc")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(str(compiled), dst)
        # sha256 over the files in sorted relative-path order (the engine's
        # DECIDE_ANE_VERIFY=1 recomputes exactly this).
        h = hashlib.sha256()
        paths = []
        for root, _, files in os.walk(dst):
            for f in files:
                full = os.path.join(root, f)
                paths.append((os.path.relpath(full, dst), full))
        for _, full in sorted(paths):
            h.update(open(full, "rb").read())
        report["models"][f"L{l}"] = {"path": f"L{l}.mlmodelc", "sha256": h.hexdigest(), "seconds": round(time.perf_counter() - t0, 2)}
        print(f"L{l}: {report['models'][f'L{l}']['seconds']} s", flush=True)
    report["total_seconds"] = round(time.perf_counter() - t_all, 1)
    json.dump(report, open(manifest_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "models"}))


if __name__ == "__main__":
    main()
