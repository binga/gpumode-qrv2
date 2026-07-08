"""V32: Fully fused persistent CUDA kernel for blocked QR.

One thread block per batch element does the ENTIRE blocked QR factorization:
- Panel factorization in shared memory (FP32)
- T matrix built inline from panel (in shared memory)
- Trailing update: output-stationary parallel GEMM, L1/L2 cache-assisted HBM reads
- ALL blocking steps in a single kernel launch (0 inter-step overhead)

This eliminates 64+ kernel launches, 40MB+ workspace allocations, and all
Python/C++ loop overhead from V27.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


# ============================================================================
# CuTe DSL QR32
# ============================================================================
_cute_qr32_status = "untried"
_cute_qr32_compiled = None
_cute_qr32_defs_ready = False


def _ensure_cute_qr32(data, H, tau):
    global _cute_qr32_status, _cute_qr32_compiled, _cute_qr32_defs_ready
    if _cute_qr32_status in ("ready", "unavailable"):
        return _cute_qr32_status == "ready"
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception:
        _cute_qr32_status = "unavailable"
        return False
    if not _cute_qr32_defs_ready:
        try:
            g = globals()

            @cute.kernel
            def _qr32_kernel(a: cute.Tensor, h: cute.Tensor, tau_out: cute.Tensor):
                bid, _, _ = cute.arch.block_idx()
                tid, _, _ = cute.arch.thread_idx()
                n = 32
                alloc = cutlass.utils.SmemAllocator()
                s_h = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((32, 32)), byte_alignment=16, swizzle=None)
                s_tau = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), byte_alignment=16, swizzle=None)
                for idx in range(tid, 1024, 32):
                    row = idx // n
                    col = idx - row * n
                    s_h[(row, col)] = a[(bid, row, col)]
                if tid < n:
                    s_tau[tid] = 0.0
                cute.arch.sync_threads()
                for j in range(32):
                    my_sq = 0.0
                    if tid > j:
                        v = s_h[(tid, j)]
                        my_sq = v * v
                    sigma_sq = cute.arch.warp_reduction_sum(my_sq, threads_in_group=32)
                    x0 = s_h[(j, j)]
                    tau_j = 0.0
                    if sigma_sq > 0.0 or (sigma_sq == 0.0 and x0 < 0.0):
                        norm_x = 1.0 / cute.math.rsqrt(x0 * x0 + sigma_sq)
                        diag = -norm_x
                        if x0 < 0.0:
                            diag = norm_x
                        v0 = x0 - diag
                        tau_j = (diag - x0) / diag
                        inv_v0 = 1.0 / v0
                        if tid > j:
                            s_h[(tid, j)] = s_h[(tid, j)] * inv_v0
                        if tid == 0:
                            s_h[(j, j)] = diag
                            s_tau[j] = tau_j
                    else:
                        if tid == 0:
                            s_tau[j] = 0.0
                    cute.arch.sync_threads()
                    tau_j = s_tau[j]
                    if tau_j != 0.0:
                        for k in range(j + 1, 32):
                            my_dot = 0.0
                            if tid == j:
                                my_dot = s_h[(j, k)]
                            if tid > j:
                                my_dot = s_h[(tid, j)] * s_h[(tid, k)]
                            dot = cute.arch.warp_reduction_sum(my_dot, threads_in_group=32)
                            scale = tau_j * dot
                            if tid == j:
                                s_h[(j, k)] = s_h[(j, k)] - scale
                            if tid > j:
                                s_h[(tid, k)] = s_h[(tid, k)] - scale * s_h[(tid, j)]
                            cute.arch.sync_threads()
                for idx in range(tid, 1024, 32):
                    row = idx // n
                    col = idx - row * n
                    h[(bid, row, col)] = s_h[(row, col)]
                if tid < n:
                    tau_out[(bid, tid)] = s_tau[tid]

            @cute.jit
            def _qr32_launch(a: cute.Tensor, h: cute.Tensor, tau_out: cute.Tensor):
                _qr32_kernel(a, h, tau_out).launch(grid=(a.shape[0], 1, 1), block=(32, 1, 1))

            g["_cute_qr32_launch"] = _qr32_launch
            g["_cute_from_dlpack"] = from_dlpack
            _cute_qr32_defs_ready = True
        except Exception:
            _cute_qr32_status = "unavailable"
            return False
    try:
        fdl = globals()["_cute_from_dlpack"]
        ac = fdl(data).mark_layout_dynamic()
        hc = fdl(H).mark_layout_dynamic()
        tc = fdl(tau).mark_layout_dynamic()
        _cute_qr32_compiled = cute.compile(globals()["_cute_qr32_launch"], ac, hc, tc)
        _cute_qr32_compiled(ac, hc, tc)
        torch.cuda.synchronize()
        _cute_qr32_status = "ready"
        return True
    except Exception:
        _cute_qr32_status = "unavailable"
        return False


def _try_cute_qr32(data):
    if data.dim() != 3 or data.size(1) != 32 or data.size(2) != 32:
        return None
    H = torch.empty_like(data)
    tau = torch.empty((data.size(0), 32), device=data.device, dtype=data.dtype)
    if not _ensure_cute_qr32(data, H, tau):
        return None
    fdl = globals()["_cute_from_dlpack"]
    _cute_qr32_compiled(fdl(data).mark_layout_dynamic(), fdl(H).mark_layout_dynamic(), fdl(tau).mark_layout_dynamic())
    return H, tau


# ============================================================================
# Fully fused CUDA kernel
# ============================================================================
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define NB 32

__device__ __forceinline__ float warp_sum_all(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ float block_reduce_f(float val, float* scratch, int tid, int nthreads) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    int warp_id = tid / 32, lane = tid % 32, num_warps = (nthreads + 31) / 32;
    if (lane == 0) scratch[warp_id] = val;
    __syncthreads();
    if (tid == 0) { float s = 0; for (int w = 0; w < num_warps; w++) s += scratch[w]; scratch[0] = s; }
    __syncthreads();
    return scratch[0];
}

template<int NTHREADS, int TILE_T>
__global__ void fused_blocked_qr(
    float* __restrict__ H,
    float* __restrict__ tau_out,
    const int n
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int num_warps = NTHREADS / 32;

    float* mat = H + (size_t)bid * n * n;
    float* tau_base = tau_out + (size_t)bid * n;

    extern __shared__ char smem_bytes[];
    float* sPanel = (float*)smem_bytes;
    float* sTau = sPanel + n * NB;
    float* sT = sTau + NB;
    float* sW = sT + NB * NB;
    float* sW2 = sW + NB * TILE_T;
    float* scratch = sW2 + NB * TILE_T;

    for (int j_start = 0; j_start < n; j_start += NB) {
        const int actual_nb = min(NB, n - j_start);
        const int j_end = j_start + actual_nb;
        const int m_panel = n - j_start;
        const int trailing_cols = n - j_end;

        // === PHASE 1: Panel factorization in SMEM ===
        for (int idx = tid; idx < m_panel * actual_nb; idx += NTHREADS) {
            int row = idx / actual_nb, col = idx % actual_nb;
            sPanel[row * NB + col] = mat[(j_start + row) * n + (j_start + col)];
        }
        if (tid < actual_nb) sTau[tid] = 0.0f;
        __syncthreads();

        for (int j = 0; j < actual_nb; j++) {
            float my_sq = 0.0f;
            for (int i = j + 1 + tid; i < m_panel; i += NTHREADS) {
                float v = sPanel[i * NB + j]; my_sq += v * v;
            }
            float sigma_sq = block_reduce_f(my_sq, scratch, tid, NTHREADS);
            float x0 = sPanel[j * NB + j], tau_j = 0.0f;
            if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
                float norm_x = sqrtf(x0 * x0 + sigma_sq);
                float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
                float v0 = x0 - diag;
                tau_j = (diag - x0) / diag;
                float inv_v0 = 1.0f / v0;
                for (int i = j + 1 + tid; i < m_panel; i += NTHREADS)
                    sPanel[i * NB + j] *= inv_v0;
                if (tid == 0) { sPanel[j * NB + j] = diag; sTau[j] = tau_j; }
            } else {
                if (tid == 0) sTau[j] = 0.0f;
            }
            __syncthreads();
            tau_j = sTau[j];
            if (tau_j == 0.0f) continue;
            for (int k = j + 1; k < actual_nb; k++) {
                float my_dot = 0.0f;
                if (tid == 0) my_dot = sPanel[j * NB + k];
                for (int i = j + 1 + tid; i < m_panel; i += NTHREADS)
                    my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
                float dot = block_reduce_f(my_dot, scratch, tid, NTHREADS);
                float scale = tau_j * dot;
                if (tid == 0) sPanel[j * NB + k] -= scale;
                for (int i = j + 1 + tid; i < m_panel; i += NTHREADS)
                    sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
                __syncthreads();
            }
        }

        // Write panel back to HBM
        for (int idx = tid; idx < m_panel * actual_nb; idx += NTHREADS) {
            int row = idx / actual_nb, col = idx % actual_nb;
            mat[(j_start + row) * n + (j_start + col)] = sPanel[row * NB + col];
        }
        if (tid < actual_nb) tau_base[j_start + tid] = sTau[tid];
        __syncthreads();

        if (trailing_cols <= 0) break;

        // === PHASE 2: Build T in SMEM ===
        for (int idx = tid; idx < NB * NB; idx += NTHREADS) sT[idx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < actual_nb; k++) {
            float tau_k = sTau[k];
            if (tid == 0) sT[k * NB + k] = tau_k;
            if (k == 0) { __syncthreads(); continue; }
            for (int i = 0; i < k; i++) {
                float my_dot = 0.0f;
                if (tid == 0) my_dot = sPanel[k * NB + i];
                for (int r = k + 1 + tid; r < m_panel; r += NTHREADS)
                    my_dot += sPanel[r * NB + i] * sPanel[r * NB + k];
                float dot = block_reduce_f(my_dot, scratch, tid, NTHREADS);
                if (tid == 0) sW[i] = dot;
                __syncthreads();
            }
            if (tid == 0) {
                for (int i = 0; i < k; i++) {
                    double acc = 0.0;
                    for (int p = 0; p < k; p++)
                        acc += (double)sT[i * NB + p] * (double)sW[p];
                    sT[i * NB + k] = (float)(-(double)tau_k * acc);
                }
            }
            __syncthreads();
        }

        // === PHASE 3: Trailing update (register-blocked) ===
        // trail := (I - V T^T V^T) trail = trail - V @ (T^T @ (V^T @ trail))
        // Each thread owns one panel-column index i (or row r) and a strip of
        // TN consecutive output columns held in registers, so each loaded V
        // value is reused across TN columns -> ~TN x less SMEM/HBM traffic.
        constexpr int CHUNK_R = 32;
        constexpr int TN = 8;
        constexpr int N_CG = TILE_T / TN;                 // column groups per tile
        float* sTrail = scratch + (NTHREADS / 32);        // CHUNK_R * TILE_T floats

        for (int tile_start = 0; tile_start < trailing_cols; tile_start += TILE_T) {
            int tile_cols = min(TILE_T, trailing_cols - tile_start);
            int col_offset = j_end + tile_start;
            int n_cg = (tile_cols + TN - 1) / TN;

            // --- Step A: W1 = V^T @ trail  (actual_nb x tile_cols) ---
            for (int idx = tid; idx < NB * TILE_T; idx += NTHREADS) sW[idx] = 0.0f;
            __syncthreads();

            for (int r_start = 0; r_start < m_panel; r_start += CHUNK_R) {
                int chunk_rows = min(CHUNK_R, m_panel - r_start);

                for (int idx = tid; idx < chunk_rows * tile_cols; idx += NTHREADS) {
                    int lr = idx / tile_cols, j = idx % tile_cols;
                    sTrail[lr * TILE_T + j] = mat[(j_start + r_start + lr) * n + col_offset + j];
                }
                __syncthreads();

                // Thread owns (i, column-group cg): accumulate TN cols in registers.
                for (int pair = tid; pair < actual_nb * n_cg; pair += NTHREADS) {
                    int i = pair / n_cg;
                    int j0 = (pair % n_cg) * TN;
                    float acc[TN];
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++) acc[jj] = 0.0f;
                    for (int lr = 0; lr < chunk_rows; lr++) {
                        int r = r_start + lr;
                        if (r < i) continue;
                        float v = (r == i) ? 1.0f : sPanel[r * NB + i];
                        const float* trow = &sTrail[lr * TILE_T + j0];
                        #pragma unroll
                        for (int jj = 0; jj < TN; jj++) acc[jj] += v * trow[jj];
                    }
                    float* wrow = &sW[i * TILE_T + j0];
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++)
                        if (j0 + jj < tile_cols) wrow[jj] += acc[jj];
                }
                __syncthreads();
            }

            // --- Step B: W2 = T^T @ W1  (actual_nb x tile_cols) ---
            for (int pair = tid; pair < actual_nb * n_cg; pair += NTHREADS) {
                int i = pair / n_cg;
                int j0 = (pair % n_cg) * TN;
                float acc[TN];
                #pragma unroll
                for (int jj = 0; jj < TN; jj++) acc[jj] = 0.0f;
                for (int p = 0; p <= i; p++) {                   // T upper-tri: T[p][i]=0 for p>i
                    float t = sT[p * NB + i];
                    const float* wrow = &sW[p * TILE_T + j0];
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++) acc[jj] += t * wrow[jj];
                }
                float* w2row = &sW2[i * TILE_T + j0];
                #pragma unroll
                for (int jj = 0; jj < TN; jj++)
                    if (j0 + jj < tile_cols) w2row[jj] = acc[jj];
            }
            __syncthreads();

            // --- Step C: trail -= V @ W2  (m_panel x tile_cols), 2D register-blocked ---
            // Thread owns (TM consecutive rows, TN cols): each W2 load reused across
            // TM rows, each V load reused across TN cols.
            constexpr int TM = 4;
            int n_rg = (m_panel + TM - 1) / TM;
            for (int pair = tid; pair < n_rg * n_cg; pair += NTHREADS) {
                int r0 = (pair / n_cg) * TM;
                int j0 = (pair % n_cg) * TN;
                float acc[TM][TN];
                #pragma unroll
                for (int tm = 0; tm < TM; tm++)
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++) acc[tm][jj] = 0.0f;
                for (int i = 0; i < actual_nb; i++) {
                    float w[TN];
                    const float* w2row = &sW2[i * TILE_T + j0];
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++) w[jj] = w2row[jj];
                    #pragma unroll
                    for (int tm = 0; tm < TM; tm++) {
                        int r = r0 + tm;
                        if (r >= m_panel || i > r) continue;
                        float v = (r == i) ? 1.0f : sPanel[r * NB + i];
                        #pragma unroll
                        for (int jj = 0; jj < TN; jj++) acc[tm][jj] += v * w[jj];
                    }
                }
                #pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    int r = r0 + tm;
                    if (r >= m_panel) continue;
                    float* dst = &mat[(j_start + r) * n + col_offset + j0];
                    #pragma unroll
                    for (int jj = 0; jj < TN; jj++)
                        if (j0 + jj < tile_cols) dst[jj] -= acc[tm][jj];
                }
            }
            __syncthreads();
        }
    }
}

__global__ void fused_qr_small(
    const float* __restrict__ A_in, float* __restrict__ H_out,
    float* __restrict__ tau_out, const int n
) {
    const int bid = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float smem[];
    float* sA = smem, *sTau = smem + n * n;
    const float* src = A_in + (size_t)bid * n * n;
    for (int idx = tid; idx < n * n; idx += 32) sA[idx] = src[idx];
    if (tid < n) sTau[tid] = 0.0f;
    __syncwarp();
    for (int j = 0; j < n; j++) {
        float my_sq = 0.0f;
        if (tid > j && tid < n) { float v = sA[tid * n + j]; my_sq = v * v; }
        float sigma_sq = warp_sum_all(my_sq);
        float x0 = sA[j * n + j], tau_j = 0.0f;
        if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
            float v0 = x0 - diag; tau_j = (diag - x0) / diag;
            if (tid > j && tid < n) sA[tid * n + j] *= (1.0f / v0);
            if (tid == 0) { sA[j * n + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0f; }
        __syncwarp(); tau_j = sTau[j]; if (tau_j == 0.0f) continue;
        for (int k = j + 1; k < n; k++) {
            float my_dot = 0.0f;
            if (tid == j) my_dot = sA[j * n + k];
            if (tid > j && tid < n) my_dot = sA[tid * n + j] * sA[tid * n + k];
            float dot = warp_sum_all(my_dot);
            float scale = tau_j * dot;
            if (tid == j) sA[j * n + k] -= scale;
            if (tid > j && tid < n) sA[tid * n + k] -= scale * sA[tid * n + j];
            __syncwarp();
        }
    }
    float* dst_h = H_out + (size_t)bid * n * n;
    float* dst_t = tau_out + (size_t)bid * n;
    for (int idx = tid; idx < n * n; idx += 32) dst_h[idx] = sA[idx];
    if (tid < n) dst_t[tid] = sTau[tid];
}

std::vector<torch::Tensor> dispatch_qr(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 3 && A.size(1) == A.size(2));
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32);
    const int batch = A.size(0), n = A.size(1);

    auto H = A.clone();
    auto tau = torch::zeros({batch, n}, A.options());

    if (n <= 32) {
        auto H2 = torch::empty_like(A);
        fused_qr_small<<<batch, 32, (n*n+n)*sizeof(float)>>>(
            A.data_ptr<float>(), H2.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H2, tau};
    }

    if (n <= 512 && batch >= 16) {
        constexpr int NTHREADS = 256;
        constexpr int TILE_T = 64;
        constexpr int CHUNK_R = 32;

        // sPanel(n*NB) + sTau(NB) + sT(NB*NB) + sW(NB*TILE_T) + sW2(NB*TILE_T)
        // + scratch(num_warps) + sTrail(CHUNK_R*TILE_T)
        size_t smem = ((size_t)n * NB + NB + NB * NB + NB * TILE_T + NB * TILE_T
                       + NTHREADS / 32 + CHUNK_R * TILE_T) * sizeof(float);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(
                fused_blocked_qr<NTHREADS, TILE_T>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        fused_blocked_qr<NTHREADS, TILE_T><<<batch, NTHREADS, smem>>>(
            H.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H, tau};
    }

    auto result = at::geqrf(A);
    return {std::get<0>(result), std::get<1>(result)};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dispatch_qr(torch::Tensor A);
"""

module = load_inline(
    name="qr_v32_fused",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    cute_qr32 = _try_cute_qr32(data)
    if cute_qr32 is not None:
        return cute_qr32
    result = module.dispatch_qr(data)
    return (result[0], result[1])
