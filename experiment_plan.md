# QR Kernel Competition — Experiment Plan

## Competition

- **Platform:** GPU MODE (gpumode.com)
- **Problem:** Batched Householder QR factorization on B200
- **Metric:** Geometric mean runtime across 7 benchmark shapes
- **Deadline:** June 30, 2026
- **Leaderboard:** https://www.gpumode.com/leaderboard/773?tab=rankings

## Validation Strategy

- **Correctness:** 19 test cases covering dense, rank-deficient, near-rank-deficient, banded, row-scaled, near-collinear, upper-triangular, clustered-scale
- **Performance:** Geometric mean of per-shape runtimes (μs)
- **Tolerances:** Factor residual rtol = 20·n·eps32, orthogonality rtol = 100·n·eps32

## Phases

### Phase 1: Bootstrap & Baseline (U1-U2)
- Set up project, install Popcorn CLI, submit baseline torch.geqrf
- Record per-shape timings, identify bottlenecks

### Phase 2: First Optimization (U3)
- Direct cuSOLVER batched calls, bypass PyTorch overhead
- Size-adaptive dispatch for small vs large matrices

### Phase 3: Custom Kernels (U4-U5)
- Blocked Householder QR with CUDA inline panel factorization
- TF32 trailing matrix update via cuBLAS batched GEMM

### Phase 4: Integration & Tuning (U6-U7)
- Unified size-adaptive dispatch
- Micro-optimizations, profiling, final submission

## Results Log

| ID | Experiment | (20,32) | (40,176) | (40,352) | (640,512) | (60,1024) | (8,2048) | (2,4096) | Geom Mean | LB Rank | Status |
|----|-----------|---------|----------|----------|-----------|-----------|----------|----------|-----------|---------|--------|
| V0-cpu | Baseline torch.geqrf (CPU) | 0.09 | 4.74 | 17.68 | 1154.80 | 698.12 | 642.97 | 964.40 | 62.91 | — | CPU-only reference |
| V0 | Baseline torch.geqrf (B200) | 324 | 22000 | 50900 | 1074000 | 240000 | 76900 | 52100 | — | ~80th | 22/22 tests pass. All timings in μs |
| V3 | Blocked WY (nb=32, FP32) | 324 | 15000 | 37000 | 617000 | 222000 | 202000 | 394000 | — | — | 22/22 pass. 1.7× on (640,512) but 7.6× slower on (2,4096) |
| V5 | Adaptive (WY ≤512, geqrf ≥1024) | 322 | 7780 | 27000 | 616000 | 239000 | 76800 | 52100 | — | — | 22/22 pass. Best of V0 + V3. (40,176) 2.8× faster than V0 |
| V6 | Optimized (vectorized V/T, nb=64) | 322 | 7330 | 34400 | 598000 | 239000 | 76800 | 52100 | — | — | 22/22 pass. Minor gain on (640,512): 616→598ms. (40,352) regressed 27→34ms with nb=64 |
| V7 | torch.compile trailing update | — | — | — | — | — | — | — | — | — | TIMEOUT: torch.compile compilation >300s on Modal |
| V8 | Tuned nb per shape | — | — | — | — | — | — | — | — | — | TIMEOUT: leaderboard mode exceeds 300s Modal limit |

## Key Findings

1. **CPU profiling** (V0): The `(640,512)` shape dominates at 1155ms, followed by `(2,4096)` at 964ms. Small shapes (n≤352) are negligible contributors to geomean.
2. **Blocked Householder (V2)**: Pure-PyTorch blocked QR is ~17% slower on CPU due to panel extraction + householder_product reconstruction overhead. The GEMM-based trailing update only wins on GPU with tensor cores.
3. **Leaderboard snapshot (June 15)**: #1 at 704μs, #2 at 1.2ms, #3 at 1.67ms. Baseline torch.geqrf at ~44.8ms. Cluster at ~5-7ms likely using cuSOLVER optimizations. Sub-2ms requires custom CUDA kernels.
4. **Popcorn working**: Leaderboard is `qr_v2` (not `qr`). Use `--no-tui` flag for agent shells.
5. **B200 baseline**: The `(640,512)` shape at 1074ms dominates — it's 3-14× slower than the next costliest shapes. This is the #1 optimization target.
6. **qr_v2 has 12 benchmark shapes** (not 7): adds mixed/rankdef/clustered/nearrank variants at 512 and 1024. The (640,512) shapes dominate regardless.
7. **V3 blocked WY results**: Major win on the critical shapes: (640,512) dropped from 1074ms→617ms (1.74×), (40,176) from 22ms→15ms (1.47×), (40,352) from 51ms→37ms (1.38×). But large-n/small-batch shapes REGRESSED badly: (2,4096) from 52ms→394ms (7.6× slower), (8,2048) from 77ms→202ms (2.6× slower). The blocked approach adds overhead from Python-level V/T construction and extra bmm calls that dominate when batch is too small to amortize.
8. **V5 is our best leaderboard submission**: ~1.7× over baseline on (640,512). V7/V8 hit 300s Modal timeout on leaderboard mode.
9. **Python-level ceiling reached**: Blocked WY with torch.geqrf panel + torch.bmm trailing update has hit its limit. Further gains require custom CUDA kernels that eliminate Python from the hot loop entirely.
10. **Next step**: Custom CUDA kernel (load_inline) that does the entire blocked QR in C++/CUDA, calling cuSOLVER for panel and cuBLAS for trailing update without Python overhead. This is U4+U5 combined.

## Current Best Configuration

_Updated after each improvement._
