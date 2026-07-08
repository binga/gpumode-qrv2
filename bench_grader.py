"""Calibration harness: measure each leaderboard shape TWO ways on one B200.

Goal: prove whether the Modal<->Popcorn gap is methodology (median single-input
vs the real eval.py grader's mean-over-many-reps) or hardware.

For every benchmark shape, in ONE Modal B200 container, we measure:
  (A) evo-style:    1 input, 3 warmup, 15 reps, clone INSIDE the timed window,
                    clear_l2_cache() each rep -> take the MEDIAN  (== current evo_benchmark.py)
  (B) grader-style: drive the canonical eval.py `_run_single_benchmark` with the
                    REAL leaderboard params (recheck, max_repeats=1000, 30s cap),
                    which uses _make_data_batch (multi-input) + reports the MEAN.

Then compare geomean(A medians) vs geomean(B means) on identical hardware.

Usage:
    .venv/bin/python bench_grader.py --target /tmp/exp0038_submission.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import modal

# The 7 ranked leaderboard shapes (seeds match evo_benchmark.py / modal_test.py).
BENCHMARK_SHAPES_7 = [
    {"batch": 20, "n": 32, "cond": 0, "seed": 100, "case": "dense"},
    {"batch": 40, "n": 176, "cond": 0, "seed": 101, "case": "dense"},
    {"batch": 40, "n": 352, "cond": 0, "seed": 102, "case": "dense"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 103, "case": "dense"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 104, "case": "dense"},
    {"batch": 8, "n": 2048, "cond": 0, "seed": 105, "case": "dense"},
    {"batch": 2, "n": 4096, "cond": 0, "seed": 106, "case": "dense"},
]

# Best-guess 12-shape extras (high-batch 512 + n=1024 variants per experiment_plan
# key-finding #6). Used only to estimate a 12-shape geomean for calibration; the
# 7-shape number is the primary anchor.
EXTRA_SHAPES_5 = [
    {"batch": 640, "n": 512, "cond": 2, "seed": 1029, "case": "dense"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"},
    {"batch": 60, "n": 1024, "cond": 4, "seed": 1041, "case": "rankdef"},
    {"batch": 60, "n": 1024, "cond": 6, "seed": 1042, "case": "clustered"},
]

cuda_tag = "13.0.0-devel-ubuntu24.04"
app = modal.App("qr-grader-cal")
image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .apt_install("ninja-build")
    .pip_install("torch", "numpy", "triton")
)


@app.function(image=image, gpu="B200", timeout=590)
def run_remote(submission_code: str, harness: dict, shapes: list):
    import os
    import sys

    os.makedirs("/root/harness", exist_ok=True)
    os.chdir("/root/harness")
    sys.path.insert(0, "/root/harness")
    os.environ["PYTHONPATH"] = "/root/harness"
    for name, code in harness.items():
        with open(f"/root/harness/{name}", "w") as f:
            f.write(code)
    with open("/root/harness/submission.py", "w") as f:
        f.write(submission_code)

    import torch

    out = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }

    from reference import generate_input
    from utils import clear_l2_cache
    import importlib

    # canonical grader module. Neutralize its correctness check during timing:
    # we want the grader's TIMING methodology only (mean over many multi-input
    # reps), and Modal has a known batch=640/n=512 correctness artifact that would
    # otherwise abort the dominant shape. Correctness is validated elsewhere
    # (disjoint-seed sidecar at safe batch + Popcorn). This does NOT change timing.
    import eval as grader

    grader.check_implementation = lambda *a, **k: (True, "")

    _ = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
    torch.cuda.synchronize()

    submission = importlib.import_module("submission")
    custom_kernel = submission.custom_kernel

    results = {}

    # one global warmup like eval.py leaderboard mode does on tests[0]
    warm = grader.TestCase(args=dict(shapes[0]), spec="warmup")
    grader._run_single_benchmark(warm, False, 1000, 5e8)

    for spec in shapes:
        label = f"({spec['batch']},{spec['n']}){'' if spec['case']=='dense' and spec['cond']==0 else ' '+spec['case']+str(spec['cond'])}"

        # ---- (A) evo-style: median of 15 single-input reps, clone in timed region ----
        data = generate_input(**spec)
        torch.cuda.synchronize()
        for _ in range(3):
            _ = custom_kernel(data.clone())
        torch.cuda.synchronize()
        evo_samples = []
        for _ in range(15):
            clear_l2_cache()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = custom_kernel(data.clone())
            e.record()
            torch.cuda.synchronize()
            evo_samples.append(s.elapsed_time(e) * 1000.0)  # ms -> us
        evo_samples.sort()
        evo_median = evo_samples[len(evo_samples) // 2]
        evo_mean = sum(evo_samples) / len(evo_samples)

        # ---- (B) grader-style: real eval.py leaderboard params, mean over many reps ----
        # _run_single_benchmark returns durations in ns (ms * 1e6 / count); mean/1000 = us.
        tc = grader.TestCase(args=dict(spec), spec=label)
        stats = grader._run_single_benchmark(tc, True, 1000, 30e9)
        if hasattr(stats, "mean"):
            grader_mean_us = stats.mean / 1000.0
            grader_best_us = stats.best / 1000.0
            grader_runs = stats.runs
            grader_std_us = stats.std / 1000.0
            grader_err = stats.err / stats.mean if stats.mean else 0.0
        else:
            grader_mean_us = float("nan")
            grader_best_us = float("nan")
            grader_runs = 0
            grader_std_us = float("nan")
            grader_err = float("nan")
            results.setdefault("_grader_errors", {})[label] = str(stats)[:200]

        results[label] = {
            "batch": spec["batch"],
            "n": spec["n"],
            "case": spec["case"],
            "cond": spec["cond"],
            "evo_median_us": evo_median,
            "evo_mean_us": evo_mean,
            "grader_mean_us": grader_mean_us,
            "grader_best_us": grader_best_us,
            "grader_std_us": grader_std_us,
            "grader_rel_err": grader_err,
            "grader_runs": grader_runs,
        }

    out["results"] = results
    return out


def _geomean(xs):
    xs = [x for x in xs if x and x == x and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--twelve", action="store_true", help="also time the 5 extra shapes")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    repo = Path(__file__).resolve().parent
    submission_code = target.read_text()
    harness = {name: (repo / name).read_text() for name in ("task.py", "reference.py", "utils.py", "eval.py")}

    shapes = list(BENCHMARK_SHAPES_7)
    if args.twelve:
        shapes = shapes + EXTRA_SHAPES_5

    print(f"[cal] target={target}")
    print(f"[cal] launching Modal B200 ({len(shapes)} shapes, real eval.py grader vs evo median)...")
    t0 = time.time()
    with modal.enable_output(), app.run():
        res = run_remote.remote(submission_code, harness, shapes)
    print(f"[cal] Modal wall: {time.time()-t0:.0f}s   {res.get('gpu')} torch={res.get('torch')} cuda={res.get('cuda')}")

    r = res["results"]
    print(f"\n{'shape':>22} | {'evo_median':>11} {'evo_mean':>10} | {'GRADER_mean':>12} {'grader_best':>11} {'rel_err':>8} {'runs':>5} | {'mean/med':>8}")
    print("-" * 110)
    for label, t in r.items():
        ratio = t["grader_mean_us"] / t["evo_median_us"] if t["evo_median_us"] else float("nan")
        print(f"{label:>22} | {t['evo_median_us']:11.1f} {t['evo_mean_us']:10.1f} | "
              f"{t['grader_mean_us']:12.1f} {t['grader_best_us']:11.1f} {t['grader_rel_err']:8.4f} {t['grader_runs']:5d} | {ratio:8.3f}")

    labels7 = [f"({s['batch']},{s['n']})" if (s['case']=='dense' and s['cond']==0) else None for s in BENCHMARK_SHAPES_7]
    # match by batch/n for the 7-shape geomean
    def pick(keys_n_b):
        vals_med, vals_grm = [], []
        for label, t in r.items():
            if (t["batch"], t["n"], t["case"], t["cond"]) in keys_n_b:
                vals_med.append(t["evo_median_us"])
                vals_grm.append(t["grader_mean_us"])
        return vals_med, vals_grm

    keys7 = {(s["batch"], s["n"], s["case"], s["cond"]) for s in BENCHMARK_SHAPES_7}
    med7, grm7 = pick(keys7)
    print("\n=== 7-shape geomeans ===")
    print(f"  evo  median-geomean : {_geomean(med7):10.1f} us   (current evo metric; expect ~9752)")
    print(f"  GRADER mean-geomean : {_geomean(grm7):10.1f} us   (leaderboard methodology; expect ~11981)")

    if args.twelve:
        med12 = [t["evo_median_us"] for t in r.values()]
        grm12 = [t["grader_mean_us"] for t in r.values()]
        print("\n=== 12-shape geomeans (best-guess extras) ===")
        print(f"  evo  median-geomean : {_geomean(med12):10.1f} us")
        print(f"  GRADER mean-geomean : {_geomean(grm12):10.1f} us")

    if res.get("results", {}).get("_grader_errors"):
        print("\n[cal] grader errors:", res["results"]["_grader_errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
