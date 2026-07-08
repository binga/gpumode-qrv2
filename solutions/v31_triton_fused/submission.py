"""V31: Triton-fused trailing update with FP32 precision fix.

Key changes from V30 (which passed 22/22 but failed LB stress tests):
- All tl.dot calls use input_precision="ieee" to avoid TF32 rounding
- Fused trailing: V^T@trail, T^T@W1, trail-=V@W2 in ONE kernel launch
- CUDA panel (double-precision for n=512) stays for numerical stability

Architecture:
- n=32: CuTe DSL QR32 (proven ~84μs)
- n<=512, batch>=16: CUDA panel + Triton fused trailing
- else: torch.geqrf fallback
"""
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


# ============================================================================
# CuTe DSL QR32 (proven fast for n=32)
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
# Triton fused trailing update kernel
# Fuses: W1 = V^T @ trailing, W2 = T^T @ W1, trailing -= V @ W2
# into ONE kernel launch. Each program handles one (batch, trailing_tile).
# ============================================================================
@triton.jit
def fused_trailing_kernel(
    H_ptr,
    V_ptr,
    T_ptr,
    n: tl.constexpr,
    j_start,
    m_panel,
    trailing_cols,
    v_batch_stride,
    NB: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_warps: tl.constexpr,
):
    """Fused trailing update: one program per (batch, trailing_tile).
    Computes trail -= V @ T^T @ V^T @ trail in one pass."""
    pid_batch = tl.program_id(0)
    pid_tile = tl.program_id(1)

    tile_start = pid_tile * TILE_N
    if tile_start >= trailing_cols:
        return

    j_end = j_start + NB
    col_offset = j_end + tile_start
    actual_tile_n = tl.minimum(TILE_N, trailing_cols - tile_start)

    h_base = H_ptr + pid_batch * n * n
    v_base = V_ptr + pid_batch * v_batch_stride
    t_base = T_ptr + pid_batch * NB * NB

    offs_m = tl.arange(0, BLOCK_M)
    offs_nb = tl.arange(0, NB)
    offs_tile = tl.arange(0, TILE_N)

    mask_m = offs_m < m_panel
    mask_tile = offs_tile < actual_tile_n

    # Load V: (BLOCK_M, NB)
    v = tl.load(
        v_base + offs_m[:, None] * NB + offs_nb[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # Load T: (NB, NB)
    t_mat = tl.load(t_base + offs_nb[:, None] * NB + offs_nb[None, :])

    # Load trailing tile: (BLOCK_M, TILE_N) from row-major H
    safe_col = tl.where(mask_tile, col_offset + offs_tile, 0)
    trail_ptrs = h_base + (j_start + offs_m[:, None]) * n + safe_col[None, :]
    trail = tl.load(
        trail_ptrs,
        mask=mask_m[:, None] & mask_tile[None, :],
        other=0.0,
    )

    # W1 = V^T @ trail: (NB, BLOCK_M) x (BLOCK_M, TILE_N) -> (NB, TILE_N)
    w1 = tl.dot(tl.trans(v), trail, input_precision="ieee")

    # W2 = T^T @ W1: (NB, NB) x (NB, TILE_N) -> (NB, TILE_N)
    w2 = tl.dot(tl.trans(t_mat), w1, input_precision="ieee")

    # trail -= V @ W2: (BLOCK_M, NB) x (NB, TILE_N) -> (BLOCK_M, TILE_N)
    trail -= tl.dot(v, w2, input_precision="ieee")

    # Store result
    tl.store(
        trail_ptrs,
        trail,
        mask=mask_m[:, None] & mask_tile[None, :],
    )


# ============================================================================
# CUDA panel + V/T builders (proven from V27)
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

__device__ double block_reduce_d(double val, double* scratch, int tid, int nthreads) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    int warp_id = tid / 32, lane = tid % 32, num_warps = (nthreads + 31) / 32;
    if (lane == 0) scratch[warp_id] = val;
    __syncthreads();
    if (tid == 0) { double s = 0; for (int w = 0; w < num_warps; w++) s += scratch[w]; scratch[0] = s; }
    __syncthreads();
    return scratch[0];
}

__global__ void fused_panel_fp32(
    float* __restrict__ A, float* __restrict__ tau_out,
    const int n, const int j_start, const int j_end
) {
    const int bid = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const int actual_nb = j_end - j_start, m = n - j_start;
    float* mat = A + (size_t)bid * n * n;
    float* tau_base = tau_out + (size_t)bid * n;
    extern __shared__ float smem[];
    float* sPanel = smem, *sTau = sPanel + m * NB, *scratch = sTau + NB;
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        sPanel[row * NB + col] = mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0f;
    __syncthreads();
    for (int j = 0; j < actual_nb; j++) {
        float my_sq = 0.0f;
        for (int i = j + 1 + tid; i < m; i += nthreads) { float v = sPanel[i * NB + j]; my_sq += v * v; }
        float sigma_sq = block_reduce_f(my_sq, scratch, tid, nthreads);
        float x0 = sPanel[j * NB + j], tau_j = 0.0f;
        if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
            float v0 = x0 - diag; tau_j = (diag - x0) / diag;
            float inv_v0 = 1.0f / v0;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * NB + j] *= inv_v0;
            if (tid == 0) { sPanel[j * NB + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0f; }
        __syncthreads();
        tau_j = sTau[j]; if (tau_j == 0.0f) continue;
        for (int k = j + 1; k < actual_nb; k++) {
            float my_dot = 0.0f;
            if (tid == 0) my_dot = sPanel[j * NB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
            float dot = block_reduce_f(my_dot, scratch, tid, nthreads);
            float scale = tau_j * dot;
            if (tid == 0) sPanel[j * NB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
            __syncthreads();
        }
    }
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = sPanel[row * NB + col];
    }
    if (tid < actual_nb) tau_base[j_start + tid] = sTau[tid];
}

__global__ void fused_panel_dbl(
    float* __restrict__ A, float* __restrict__ tau_out,
    const int n, const int j_start, const int j_end
) {
    const int bid = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const int actual_nb = j_end - j_start, m = n - j_start;
    float* mat = A + (size_t)bid * n * n;
    float* tau_base = tau_out + (size_t)bid * n;
    extern __shared__ char smem_raw[];
    double* sPanel = (double*)smem_raw, *sTau = sPanel + m * NB, *scratch = sTau + NB;
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        sPanel[row * NB + col] = (double)mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0;
    __syncthreads();
    for (int j = 0; j < actual_nb; j++) {
        double my_sq = 0.0;
        for (int i = j + 1 + tid; i < m; i += nthreads) { double v = sPanel[i * NB + j]; my_sq += v * v; }
        double sigma_sq = block_reduce_d(my_sq, scratch, tid, nthreads);
        double x0 = sPanel[j * NB + j], tau_j = 0.0;
        if (sigma_sq > 0.0 || (sigma_sq == 0.0 && x0 < 0.0)) {
            double norm_x = sqrt(x0 * x0 + sigma_sq);
            double diag = (x0 >= 0.0) ? -norm_x : norm_x;
            double v0 = x0 - diag; tau_j = (diag - x0) / diag;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * NB + j] *= (1.0 / v0);
            if (tid == 0) { sPanel[j * NB + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0; }
        __syncthreads();
        tau_j = sTau[j]; if (tau_j == 0.0) continue;
        for (int k = j + 1; k < actual_nb; k++) {
            double my_dot = 0.0;
            if (tid == 0) my_dot = sPanel[j * NB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
            double dot = block_reduce_d(my_dot, scratch, tid, nthreads);
            double scale = tau_j * dot;
            if (tid == 0) sPanel[j * NB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
            __syncthreads();
        }
    }
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = (float)sPanel[row * NB + col];
    }
    if (tid < actual_nb) tau_base[j_start + tid] = (float)sTau[tid];
}

__global__ void build_v_kernel(
    const float* __restrict__ H, float* __restrict__ V,
    const int n, const int j_start, const int actual_nb, const int m_panel,
    const int v_ld, const int v_batch_stride
) {
    const int bid = blockIdx.y;
    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < m_panel * actual_nb; linear += (size_t)gridDim.x * blockDim.x) {
        int row = linear / actual_nb, col = linear % actual_nb;
        float value = 0.0f;
        if (row == col) value = 1.0f;
        else if (row > col) value = H[(size_t)bid * n * n + (j_start + row) * n + (j_start + col)];
        V[(size_t)bid * v_batch_stride + row * v_ld + col] = value;
    }
}

__global__ void build_t_parallel(
    const float* __restrict__ V, const float* __restrict__ tau,
    float* __restrict__ T,
    const int m_panel, const int actual_nb, const int v_ld, const int t_ld,
    const int j_start, const int n, const int v_batch_stride
) {
    const int bid = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const float* v = V + (size_t)bid * v_batch_stride;
    const float* tau_b = tau + (size_t)bid * n;
    float* t = T + (size_t)bid * t_ld * t_ld;
    extern __shared__ float smem_t[];
    float* s_dots = smem_t, *s_work = s_dots + t_ld;
    for (int idx = tid; idx < t_ld * t_ld; idx += nthreads) t[idx] = 0.0f;
    __syncthreads();
    for (int k = 0; k < actual_nb; k++) {
        float tau_k = tau_b[j_start + k];
        if (tid == 0) t[k * t_ld + k] = tau_k;
        if (k == 0) { __syncthreads(); continue; }
        for (int i = 0; i < k; i++) {
            float my_dot = 0.0f;
            for (int r = tid; r < m_panel; r += nthreads)
                my_dot += v[r * v_ld + i] * v[r * v_ld + k];
            float dot = block_reduce_f(my_dot, s_work, tid, nthreads);
            if (tid == 0) s_dots[i] = dot;
            __syncthreads();
        }
        if (tid == 0) {
            for (int i = 0; i < k; i++) {
                double acc = 0.0;
                for (int p = 0; p < k; p++) acc += (double)t[i * t_ld + p] * (double)s_dots[p];
                t[i * t_ld + k] = (float)(-(double)tau_k * acc);
            }
        }
        __syncthreads();
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

std::vector<torch::Tensor> panel_and_v(torch::Tensor H, torch::Tensor tau,
    torch::Tensor V, int n, int j, int j_end, int batch, bool use_dbl) {

    int nb = j_end - j;
    int m_panel = n - j;
    int threads = 256;
    int num_warps = threads / 32;

    if (use_dbl) {
        size_t smem = ((size_t)m_panel * NB + NB + num_warps) * sizeof(double);
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(fused_panel_dbl, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fused_panel_dbl<<<batch, threads, smem>>>(
            H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
    } else {
        size_t smem = ((size_t)m_panel * NB + NB + num_warps) * sizeof(float);
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(fused_panel_fp32, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fused_panel_fp32<<<batch, threads, smem>>>(
            H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
    }

    int v_batch_stride = m_panel * nb;
    int per_matrix_v = m_panel * nb;
    int v_blocks = std::min<int>((per_matrix_v + threads - 1) / threads, 65535);
    build_v_kernel<<<dim3(v_blocks, batch), threads>>>(
        H.data_ptr<float>(), V.data_ptr<float>(),
        n, j, nb, m_panel, nb, v_batch_stride);

    return {V};
}

std::vector<torch::Tensor> build_t(torch::Tensor V, torch::Tensor tau,
    torch::Tensor T, int n, int j, int j_end, int batch, int m_panel) {

    int nb = j_end - j;
    int threads = 256;
    int num_warps = threads / 32;
    int v_batch_stride = m_panel * nb;
    size_t t_smem = (nb + num_warps) * sizeof(float);
    build_t_parallel<<<batch, threads, t_smem>>>(
        V.data_ptr<float>(), tau.data_ptr<float>(), T.data_ptr<float>(),
        m_panel, nb, nb, nb, j, n, v_batch_stride);
    return {T};
}

std::vector<torch::Tensor> dispatch_qr_small(torch::Tensor A) {
    const int batch = A.size(0), n = A.size(1);
    auto H = torch::empty_like(A);
    auto tau = torch::zeros({batch, n}, A.options());
    fused_qr_small<<<batch, 32, (n*n+n)*sizeof(float)>>>(
        A.data_ptr<float>(), H.data_ptr<float>(), tau.data_ptr<float>(), n);
    return {H, tau};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> panel_and_v(torch::Tensor H, torch::Tensor tau,
    torch::Tensor V, int n, int j, int j_end, int batch, bool use_dbl);
std::vector<torch::Tensor> build_t(torch::Tensor V, torch::Tensor tau,
    torch::Tensor T, int n, int j, int j_end, int batch, int m_panel);
std::vector<torch::Tensor> dispatch_qr_small(torch::Tensor A);
"""

module = load_inline(
    name="qr_v31_triton",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["panel_and_v", "build_t", "dispatch_qr_small"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def _blocked_qr_triton(data: torch.Tensor) -> output_t:
    batch, n, _ = data.shape
    nb = 32
    H = data.clone()
    tau = torch.zeros(batch, n, device=data.device, dtype=data.dtype)

    is_medium = n <= 352

    # Preallocate workspaces at maximum size
    # V: (batch, n, nb) with v_batch_stride = n * nb (fixed across steps)
    V = torch.empty(batch, n * nb, device=data.device, dtype=data.dtype)
    T = torch.empty(batch, nb * nb, device=data.device, dtype=data.dtype)

    # BLOCK_M: next power of 2 >= n (fixed for all steps — avoids recompilation)
    block_m = 1
    while block_m < n:
        block_m *= 2
    block_m = min(block_m, 512)

    # TILE_N: larger tiles reduce V re-reads from HBM
    tile_n = 128 if n > 256 else 64

    v_batch_stride = n * nb  # fixed stride for preallocated V

    for j in range(0, n, nb):
        actual_nb = min(nb, n - j)
        j_end = j + actual_nb
        m_panel = n - j
        trailing_cols = n - j_end
        use_dbl = not is_medium and m_panel > 256

        # Panel factorization + V construction (CUDA)
        # Reshape V for the C++ interface: (batch, n, nb) view
        V_3d = V.view(batch, n, nb)
        module.panel_and_v(H, tau, V_3d, n, j, j_end, batch, use_dbl)

        if trailing_cols <= 0:
            break

        # T construction (CUDA)
        T_3d = T.view(batch, nb, nb)
        module.build_t(V_3d, tau, T_3d, n, j, j_end, batch, m_panel)

        # Fused trailing update (Triton)
        num_tiles = (trailing_cols + tile_n - 1) // tile_n

        grid = (batch, num_tiles)
        fused_trailing_kernel[grid](
            H, V, T,
            n, j, m_panel, trailing_cols,
            v_batch_stride,
            NB=nb,
            TILE_N=tile_n,
            BLOCK_M=block_m,
            num_warps=8,
        )

    return H, tau


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    # Tier 1: CuTe QR32
    cute_qr32 = _try_cute_qr32(data)
    if cute_qr32 is not None:
        return cute_qr32

    # Tier 2: n<=32 CUDA fused
    if n <= 32:
        result = module.dispatch_qr_small(data)
        return (result[0], result[1])

    # Tier 3: Triton blocked QR for medium/large batch
    if n <= 512 and batch >= 16:
        return _blocked_qr_triton(data)

    # Tier 4: torch.geqrf fallback
    return torch.geqrf(data)
