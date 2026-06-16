"""Modal test harness for QR kernel development.

Uploads a submission to a Modal A100, runs the correctness checker and
per-shape timing benchmarks, returns results in ~30s instead of the
5-8 minute Popcorn queue round-trip.

Usage:
    modal run modal_test.py --solution v9_cpp_blocked
    modal run modal_test.py --solution v5_adaptive --benchmark
    modal run modal_test.py  # defaults to ./submission.py
"""

import modal
import time
from pathlib import Path

app = modal.App("qr-kernel-test")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
cuda_tag = f"{cuda_version}-{flavor}-{os_version}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .apt_install("ninja-build")
    .pip_install("torch", "numpy")
    .add_local_file("task.py", remote_path="/root/harness/task.py")
    .add_local_file("reference.py", remote_path="/root/harness/reference.py")
    .add_local_file("utils.py", remote_path="/root/harness/utils.py")
)

TEST_CASES = [
    {"batch": 20, "n": 32, "cond": 0, "seed": 1, "case": "dense"},
    {"batch": 40, "n": 176, "cond": 0, "seed": 2, "case": "dense"},
    {"batch": 40, "n": 352, "cond": 0, "seed": 3, "case": "dense"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 4, "case": "dense"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 5, "case": "dense"},
    {"batch": 8, "n": 2048, "cond": 0, "seed": 6, "case": "dense"},
    {"batch": 2, "n": 4096, "cond": 0, "seed": 7, "case": "dense"},
    # Stress tests (various matrix types at n=512)
    {"batch": 16, "n": 512, "cond": 4, "seed": 10, "case": "rankdef"},
    {"batch": 16, "n": 512, "cond": 4, "seed": 11, "case": "nearrank"},
    {"batch": 16, "n": 512, "cond": 6, "seed": 12, "case": "clustered"},
    {"batch": 16, "n": 512, "cond": 4, "seed": 13, "case": "band"},
    {"batch": 16, "n": 512, "cond": 6, "seed": 14, "case": "rowscale"},
    {"batch": 16, "n": 512, "cond": 4, "seed": 15, "case": "nearcollinear"},
    {"batch": 16, "n": 512, "cond": 2, "seed": 16, "case": "upper"},
    # Small matrix stress
    {"batch": 64, "n": 32, "cond": 4, "seed": 20, "case": "band"},
    {"batch": 64, "n": 32, "cond": 6, "seed": 21, "case": "rowscale"},
    {"batch": 64, "n": 32, "cond": 4, "seed": 22, "case": "nearcollinear"},
    # Large matrix stress
    {"batch": 4, "n": 1024, "cond": 4, "seed": 30, "case": "rankdef"},
    {"batch": 4, "n": 1024, "cond": 6, "seed": 31, "case": "clustered"},
    {"batch": 2, "n": 2048, "cond": 4, "seed": 32, "case": "band"},
]

BENCHMARK_SHAPES = [
    {"batch": 20, "n": 32, "cond": 0, "seed": 100, "case": "dense"},
    {"batch": 40, "n": 176, "cond": 0, "seed": 101, "case": "dense"},
    {"batch": 40, "n": 352, "cond": 0, "seed": 102, "case": "dense"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 103, "case": "dense"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 104, "case": "dense"},
    {"batch": 8, "n": 2048, "cond": 0, "seed": 105, "case": "dense"},
    {"batch": 2, "n": 4096, "cond": 0, "seed": 106, "case": "dense"},
]


@app.function(image=image, gpu="A100", timeout=300)
def run_tests(submission_code: str, run_benchmark: bool = False, benchmark_repeats: int = 10):
    """Run correctness tests and optional benchmark on Modal A100."""
    import sys
    import os
    sys.path.insert(0, "/root/harness")
    os.chdir("/root/harness")

    with open("/root/harness/submission.py", "w") as f:
        f.write(submission_code)

    import torch
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print()

    from reference import generate_input, check_implementation

    # Warm up GPU
    _ = torch.randn(100, 100, device="cuda") @ torch.randn(100, 100, device="cuda")
    torch.cuda.synchronize()

    # Import submission (triggers any load_inline compilation)
    compile_start = time.time()
    try:
        import importlib
        submission = importlib.import_module("submission")
        custom_kernel = submission.custom_kernel
    except Exception as e:
        import traceback
        return {"status": "COMPILE_ERROR", "error": str(e), "traceback": traceback.format_exc()}
    compile_time = time.time() - compile_start
    print(f"Compilation time: {compile_time:.1f}s")
    print()

    # Run correctness tests
    results = {"compile_time_s": compile_time, "tests": [], "status": "PASS"}

    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    all_pass = True
    for i, spec in enumerate(TEST_CASES):
        label = f"({spec['batch']},{spec['n']}) {spec['case']}"
        try:
            data = generate_input(**spec)
            torch.cuda.synchronize()
            t0 = time.time()
            output = custom_kernel(data.clone())
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000
            ok, msg = check_implementation(data, output)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {i+1:2d}/{len(TEST_CASES)} {label:30s} {elapsed_ms:8.1f}ms  {msg[:80] if not ok else ''}")
            results["tests"].append({"spec": label, "status": status, "time_ms": elapsed_ms, "msg": msg if not ok else ""})
        except Exception as e:
            all_pass = False
            print(f"  [ERROR] {i+1:2d}/{len(TEST_CASES)} {label:30s} {str(e)[:80]}")
            results["tests"].append({"spec": label, "status": "ERROR", "error": str(e)})

    if not all_pass:
        results["status"] = "FAIL"
    print(f"\nResult: {results['status']} ({sum(1 for t in results['tests'] if t['status'] == 'PASS')}/{len(TEST_CASES)} passed)")

    # Run benchmark if requested and all tests pass
    if run_benchmark and all_pass:
        print()
        print("=" * 60)
        print("BENCHMARK")
        print("=" * 60)
        timings = {}
        for spec in BENCHMARK_SHAPES:
            label = f"({spec['batch']},{spec['n']})"
            data = generate_input(**spec)
            torch.cuda.synchronize()

            # Warmup
            for _ in range(3):
                _ = custom_kernel(data.clone())
                torch.cuda.synchronize()

            # Timed runs
            times_us = []
            for _ in range(benchmark_repeats):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = custom_kernel(data.clone())
                end.record()
                torch.cuda.synchronize()
                times_us.append(start.elapsed_time(end) * 1000)  # ms → μs

            median = sorted(times_us)[len(times_us) // 2]
            mean = sum(times_us) / len(times_us)
            timings[label] = {"median_us": median, "mean_us": mean, "min_us": min(times_us), "max_us": max(times_us)}
            print(f"  {label:12s}  median={median:10.0f}μs  mean={mean:10.0f}μs  min={min(times_us):10.0f}μs")

        # Geometric mean
        import math
        medians = [v["median_us"] for v in timings.values()]
        geomean = math.exp(sum(math.log(x) for x in medians) / len(medians))
        print(f"\n  Geometric mean (median): {geomean:.0f}μs = {geomean/1000:.2f}ms")
        results["benchmark"] = timings
        results["geomean_us"] = geomean

    return results


@app.local_entrypoint()
def main(solution: str = "", benchmark: bool = False, repeats: int = 10):
    """Entry point: specify --solution <name> or defaults to ./submission.py"""
    if solution:
        submission_path = Path(f"solutions/{solution}/submission.py")
    else:
        submission_path = Path("submission.py")

    if not submission_path.exists():
        print(f"ERROR: {submission_path} not found")
        return

    print(f"Testing: {submission_path}")
    print(f"Benchmark: {'yes' if benchmark else 'no'}")
    print(f"Uploading to Modal A100...")
    print()

    code = submission_path.read_text()
    start = time.time()
    results = run_tests.remote(code, run_benchmark=benchmark, benchmark_repeats=repeats)
    wall_time = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Status: {results['status']}")
    if "geomean_us" in results:
        print(f"Geomean: {results['geomean_us']:.0f}μs ({results['geomean_us']/1000:.2f}ms)")
