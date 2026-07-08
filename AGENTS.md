# GPU MODE QR Kernel Competition — Agent Guidelines

## Competition Overview

- **Platform:** GPU MODE (gpumode.com), not Kaggle
- **Problem:** Batched square compact-Householder QR factorization
- **GPU:** NVIDIA B200 (Blackwell)
- **Deadline:** June 30, 2026
- **Submission:** Single `submission.py` file via Popcorn CLI

## Submission Format

```python
import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    # data: batch x n x n float32 CUDA tensor
    # return: (H, tau) matching torch.geqrf convention
    ...
```

## CLI Commands

The leaderboard name is `qr_v2` (not `qr`).

```bash
# Correctness test (19 cases, 240s timeout)
popcorn-cli submit --leaderboard qr_v2 --gpu B200 --mode test --no-tui submission.py

# Benchmark (7 shapes, 480s timeout)
popcorn-cli submit --leaderboard qr_v2 --gpu B200 --mode benchmark --no-tui submission.py

# Leaderboard submission (ranked, 900s timeout)
popcorn-cli submit --leaderboard qr_v2 --gpu B200 --mode leaderboard --no-tui submission.py
```

Always use `--no-tui` when running from agent shells (the TUI requires a real terminal).

## Correctness Criteria

- Factor residual: `rtol = 20 * n * eps32`
- Orthogonality: `rtol = 100 * n * eps32`
- Must pass all matrix types: dense, rankdef, nearrank, clustered, band, rowscale, nearcollinear, upper

## Benchmark Shapes

> ⚠️ **The leaderboard ranks the geometric mean over 12 shapes, not 7** (confirmed
> 2026-06-28 from Popcorn ranked output). The 7 dense shapes below are the base set;
> the ranked set adds 5 stress variants at the two dominant sizes: `(640,512)` ×3
> (mixed/rankdef seed 770003/clustered seed 770004) and `(60,1024)` ×2 (incl.
> nearrank seed 770005). Optimize the **12-shape** geomean. The evo benchmark
> (`evo_benchmark.py`, epoch 2) times all 12; Modal tracks the leaderboard at a
> stable **×1.05** (Modal slightly higher). See `experiment_plan.md` → "Benchmark
> Reliability Fix". Historical "Popcorn ranked" 7-shape numbers in the log were
> actually Modal medians.

7 base dense shapes (the other 5 ranked shapes are variants of (640,512)/(60,1024)):

| batch | n    | Notes                          |
| ----- | ---- | ------------------------------ |
| 20    | 32   | Small                          |
| 40    | 176  | Small                          |
| 40    | 352  | Medium                         |
| 640   | 512  | **Critical** — optimizer-style (×4 in ranked set) |
| 60    | 1024 | Large (×3 in ranked set)       |
| 8     | 2048 | Large                          |
| 2     | 4096 | Large                          |


## Solution Organization

Each solution lives in its own subfolder under `solutions/`. The submission file is always named `submission.py`. Each solution includes a single-file interactive explainer (HTML) that walks through the approach, algorithm, and key design decisions.

```
qrproblem/
├── solutions/
│   ├── v0_baseline/
│   │   ├── submission.py        # torch.geqrf baseline
│   │   └── explainer.html       # Interactive walkthrough
│   ├── v1_blocked_wy/
│   │   ├── submission.py        # Blocked Householder with WY trailing GEMM
│   │   └── explainer.html
│   ├── v2_cuda_panel/
│   │   ├── submission.py        # CUDA inline panel kernel + cuBLAS trailing
│   │   └── explainer.html
│   └── ...
```

To submit a specific solution:

```bash
popcorn-cli submit --leaderboard qr_v2 --gpu B200 --mode test --no-tui solutions/v2_cuda_panel/submission.py
```

## Experiment Workflow

This is NOT a Kaggle ML competition. No MicroCV/CV/LB pipeline.

1. **Write kernel** — create or edit `solutions/<version>/submission.py`
2. **Write explainer** — create `solutions/<version>/explainer.html` explaining the approach
3. **Compile & test on Modal** — use Modal serverless GPUs for CUDA compilation and correctness testing (fast iteration, no Popcorn queue)
4. **Remote test** — `popcorn-cli submit --mode test` (B200, correctness gate)
5. **Remote benchmark** — `popcorn-cli submit --mode benchmark` (B200, timing)
6. **Leaderboard** — `popcorn-cli submit --mode leaderboard` (B200, ranked)

## Modal for Development

Use Modal serverless GPUs (modal.com) for compiling and testing CUDA kernels before submitting to the leaderboard via Popcorn. This is critical because:

- No local CUDA GPU available (Mac development machine)
- Popcorn submissions queue and take 2-8 minutes per round-trip
- CUDA inline (`load_inline`) compilation needs a GPU with nvcc
- Modal provides instant A100/H100 access for fast iteration

```bash
# Run a submission's correctness tests on Modal
modal run modal_test.py --solution v6_optimized

# Interactive shell on a GPU for debugging
modal shell --gpu a100
```

The Modal test script (`modal_test.py`) should:

1. Upload `submission.py`, `task.py`, `reference.py`, `eval.py`, `utils.py`
2. Run the correctness checker on a GPU
3. Report per-shape timings
4. Return results without waiting for Popcorn queue

Use Popcorn only for final leaderboard submissions after Modal testing confirms correctness and promising timings.

## Git

This folder (`qrproblem/`) has its own standalone git repository (separate from the parent `kaggle/` repo). All git operations should run inside this directory, not the parent.

```bash
cd /Users/phani/Work/Adhoc/kaggle/qrproblem
git status   # operates on qrproblem's own .git
```

## Results Tracking

**Log every experiment result immediately** after obtaining benchmark timings — from Popcorn. Do not defer logging; results must be captured in the same commit as the code change.

For each experiment, add a row to the Results Log table in `experiment_plan.md` with:

- All per-shape timings (μs) from the benchmark output
- Computed geometric mean (7-shape and/or 12-shape)
- LB rank or submission status
- Key observations (what improved, what regressed, why)

## Key Constraints

- Single Python file submission
- CUDA inline via `torch.utils.cpp_extension.load_inline` for custom kernels
- FP16/TF32/FP8 allowed internally, output must be FP32
- Modal for dev/test iteration, Popcorn for leaderboard submission

## Competitive Intelligence (GPU MODE Discord, June 2026)

Clues gathered from the competition Discord. Use these to prioritize work; they
independently confirm several of our own findings in `experiment_plan.md`.

### Where the frontier actually is

- **The ~500µs leaderboard leader is almost certainly reward-hacking** (caching
  the probe/test inputs — the harness reuses them). A credible strong competitor
  noted "500/600us is how fast an implementation that caches probes" runs, and
  agents have been caught scraping submissions + reward-hacking blog posts.
  **Do not chase 500µs.** The legit "tight pack" is ~1.5–4k µs (one competitor
  hit 2300 by hand with just Claude Max + free Modal credits).
- **Low precision is explicitly allowed** ("they set the error tolerances kind
  of loose to allow it"). FP8/FP4 can be 3–4× faster, **but only on the trailing
  GEMM** — CUDA cores can't do FP8/FP4, only tensor cores can, and the GEMM is a
  minority of the work (see bottleneck breakdown). Naive `allow_tf32=True` blows
  precision (our V33: 2760× over limit). Winners use **selective/emulated low
  precision (3×TF32 / split-K) on the GEMM only**, keeping panels in FP32/FP64.

### The real bottleneck is the panel factorization, NOT the GEMM

Profiling shared by a top competitor (weighted into the geomean):

| Stage | Share |
|---|---|
| Panel factorization | 38% |
| Trailing GEMM | 20% |
| Trailing apply | 16% |
| Building T matrix | 11% |
| LU stuff | 8% |

- "The limiting factor is the panel factorizations and the dependencies between
  them... it ends up being **latency bound**." If someone is 3–4× faster, "it's
  almost certainly not about the GEMMs." This confirms our V32 finding (the
  sync-bound serial panel is the wall). **The win comes from a parallel/recursive
  panel that breaks the inter-panel dependency chain**, not a better GEMM.

### Shape priorities

- **n=512 is the highest bang-for-buck** ("512 is where most of my time went")
  because the 512 shapes appear most in the geomean. This is our dominant
  `(640,512)` family.
- **Bigger shapes (1024/2048/4096) are GEMM-dominant** → where tensor cores /
  low precision actually pay off. These are still on baseline `geqrf` for us.
- **n=4096 is the hardest.** Use **Cholesky-QR + tall-skinny / CAQR**
  (Cholesky QR is explicitly allowed): form AᵀA → Cholesky → Q = A·R⁻¹ with 1–2
  reorthogonalization passes for stability. Tensor-core friendly, avoids the
  serial Householder panel.

### Roofline

- Model each shape as memory-bandwidth bound: bytes read+written per kernel ÷
  B200 bandwidth → lower bound on time. Note "we can't fully saturate a B200 for
  some of these shapes." Compute this per-shape to know real headroom before
  investing.

### Reading list (from a competitor following the literature)

- arXiv `2203.03341` — tensor cores on TF32 (relevant to 3×TF32 trailing)
- arXiv `0806.2159` — communication-avoiding QR (tall-skinny)
- SIAM `10.1137/18M1218212`
- Berkeley EECS-2013-175 (tall-skinny CAQR)
- **MAGMA batched QR from NVIDIA (~2022)** — most directly applicable
- **Muon / Polar Express** papers — Newton–Schulz polar/orthogonalization
  iterations (tensor-core-heavy alternative orthogonalization)
- IBM Journal (Elmroth) — history, O(n³)
- A tensor-core Householder QR paper (IEEE-locked, done on Ampere; needs scaling
  to Blackwell)

### Workflow tips from competitors

- **Start fresh context often.** Long sessions plateau when the agent decides it
  "reached the speed limit"; restarting with fresh context unblocks progress.
- **Ask the agent to run short tests** to conserve Modal credits (avoids the
  300s Modal timeout on leaderboard mode).
- Local GPU (even a 4090) is great for early iteration; move to Modal around
  ~5000µs. Not strictly required — 2300µs achieved with no local GPU.

### Next actions ranked by leverage

1. **Parallel/recursive panel for `(640,512)`** — break the inter-panel
   dependency chain (the confirmed #1 bottleneck; V32 single-block couldn't).
2. **Cholesky-QR (with reorthogonalization) for `(8,2048)` / `(2,4096)`** —
   both still on baseline `geqrf`; compute-bound → tensor cores should win.
3. **Selective low-precision trailing GEMM (3×TF32 / FP8 split)** on
   GEMM-dominant large shapes only; keep panels FP32 to avoid V33-style blowup.
4. **Compute the bandwidth roofline per shape** to bound the remaining headroom.

