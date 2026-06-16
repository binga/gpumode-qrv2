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

## Benchmark Shapes (ranked by geometric mean)


| batch | n    | Notes                          |
| ----- | ---- | ------------------------------ |
| 20    | 32   | Small                          |
| 40    | 176  | Small                          |
| 40    | 352  | Medium                         |
| 640   | 512  | **Critical** — optimizer-style |
| 60    | 1024 | Large                          |
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

## Key Constraints

- Single Python file submission
- CUDA inline via `torch.utils.cpp_extension.load_inline` for custom kernels
- FP16/TF32/FP8 allowed internally, output must be FP32
- Modal for dev/test iteration, Popcorn for leaderboard submission

