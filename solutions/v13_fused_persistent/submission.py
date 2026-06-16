"""V13: Fused CUDA panel kernel with double-precision accumulation.

Uses double-precision for norm and dot-product reductions in the panel
factorization to match cuSOLVER's numerical stability for tall panels
(m up to 512). Results are stored in float32.

Tier 1 (n<=32): fused warp-level kernel in SMEM
Tier 2 (n<=512, batch>=16): fused panel (double accum) + ATen trailing
Tier 3: torch.geqrf fallback
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

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

// Double-precision block reduction for numerical stability on tall panels
__device__ double block_reduce_d(double val, double* scratch, int tid, int nthreads) {
    // Intra-warp reduction in double
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    int warp_id = tid / 32;
    int lane = tid % 32;
    int num_warps = (nthreads + 31) / 32;

    if (lane == 0) scratch[warp_id] = val;
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        for (int w = 0; w < num_warps; w++) sum += scratch[w];
        scratch[0] = sum;
    }
    __syncthreads();
    return scratch[0];
}

// Fused panel kernel: FULL double-precision internal computation.
// Panel stored as doubles in SMEM. Converts to float32 only on write-back.
// This matches cuSOLVER's internal precision for numerical stability.
__global__ void fused_panel_kernel(
    float* __restrict__ A,
    float* __restrict__ tau_out,
    const int n,
    const int j_start,
    const int j_end
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int actual_nb = j_end - j_start;
    const int m = n - j_start;

    float* mat = A + (size_t)bid * n * n;
    float* tau_base = tau_out + (size_t)bid * n;

    extern __shared__ char smem_raw[];
    // Layout: sPanel (m*NB doubles) | sTau (NB doubles) | scratch (8 doubles)
    double* sPanel = (double*)smem_raw;
    double* sTau = sPanel + m * NB;
    double* scratch = sTau + NB;

    // Load panel (float → double)
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb;
        int col = idx % actual_nb;
        sPanel[row * NB + col] = (double)mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0;
    __syncthreads();

    for (int j = 0; j < actual_nb; j++) {
        double my_sq = 0.0;
        for (int i = j + 1 + tid; i < m; i += nthreads) {
            double v = sPanel[i * NB + j];
            my_sq += v * v;
        }
        double sigma_sq = block_reduce_d(my_sq, scratch, tid, nthreads);

        double x0 = sPanel[j * NB + j];
        double tau_j = 0.0;

        if (sigma_sq > 0.0 || (sigma_sq == 0.0 && x0 < 0.0)) {
            double norm_x = sqrt(x0 * x0 + sigma_sq);
            double diag = (x0 >= 0.0) ? -norm_x : norm_x;
            double v0 = x0 - diag;
            tau_j = (diag - x0) / diag;
            double inv_v0 = 1.0 / v0;

            for (int i = j + 1 + tid; i < m; i += nthreads) {
                sPanel[i * NB + j] *= inv_v0;
            }
            if (tid == 0) {
                sPanel[j * NB + j] = diag;
                sTau[j] = tau_j;
            }
        } else {
            if (tid == 0) sTau[j] = 0.0;
        }
        __syncthreads();

        tau_j = sTau[j];
        if (tau_j == 0.0) continue;

        for (int k = j + 1; k < actual_nb; k++) {
            double my_dot = 0.0;
            if (tid == 0) my_dot = sPanel[j * NB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) {
                my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
            }
            double dot = block_reduce_d(my_dot, scratch, tid, nthreads);
            double scale = tau_j * dot;

            if (tid == 0) sPanel[j * NB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) {
                sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
            }
            __syncthreads();
        }
    }

    // Write back (double → float)
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb;
        int col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = (float)sPanel[row * NB + col];
    }
    if (tid < actual_nb) tau_base[j_start + tid] = (float)sTau[tid];
}

// Small fused kernel (n<=32, proven correct)
__global__ void fused_qr_small(
    const float* __restrict__ A_in,
    float* __restrict__ H_out,
    float* __restrict__ tau_out,
    const int n
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* sA = smem;
    float* sTau = smem + n * n;
    const float* src = A_in + (size_t)bid * n * n;
    for (int idx = tid; idx < n * n; idx += 32) sA[idx] = src[idx];
    if (tid < n) sTau[tid] = 0.0f;
    __syncwarp();
    for (int j = 0; j < n; j++) {
        float my_sq = 0.0f;
        if (tid > j && tid < n) { float v = sA[tid * n + j]; my_sq = v * v; }
        float sigma_sq = warp_sum_all(my_sq);
        float x0 = sA[j * n + j];
        float tau_j = 0.0f;
        if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
            float v0 = x0 - diag;
            tau_j = (diag - x0) / diag;
            float inv_v0 = 1.0f / v0;
            if (tid > j && tid < n) sA[tid * n + j] *= inv_v0;
            if (tid == 0) { sA[j * n + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0f; }
        __syncwarp();
        tau_j = sTau[j];
        if (tau_j == 0.0f) continue;
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
    const int batch = A.size(0);
    const int n = A.size(1);

    auto H = A.clone();
    auto tau = torch::zeros({batch, n}, A.options());

    if (n <= 32) {
        auto H2 = torch::empty_like(A);
        size_t smem_sz = (n * n + n) * sizeof(float);
        fused_qr_small<<<batch, 32, smem_sz>>>(
            A.data_ptr<float>(), H2.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H2, tau};
    }

    if (n <= 512 && batch >= 16) {
        int nb = NB;
        int threads = 256;
        int num_warps = threads / 32;

        // Set max shared memory for the panel kernel
        // Panel stored as doubles: (m*NB + NB + num_warps) doubles
        size_t max_smem = (n * NB + NB + num_warps) * sizeof(double);
        cudaFuncSetAttribute(fused_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);

        for (int j = 0; j < n; j += nb) {
            int actual_nb = std::min(nb, n - j);
            int j_end = j + actual_nb;
            int m_panel = n - j;

            // Panel in double: (m_panel*NB + NB + num_warps) doubles
            size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(double);
            fused_panel_kernel<<<batch, threads, panel_smem>>>(
                H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);

            int trailing_cols = n - j_end;
            if (trailing_cols <= 0) break;

            // ATen trailing update (proven correct)
            auto panel_h = H.slice(1, j, n).slice(2, j, j_end).contiguous();
            auto panel_tau = tau.slice(1, j, j_end);
            auto V = torch::zeros({batch, m_panel, actual_nb}, A.options());
            for (int k = 0; k < actual_nb; k++) {
                V.select(1, k).select(1, k).fill_(1.0f);
                if (k + 1 < m_panel)
                    V.slice(1, k+1, m_panel).select(2, k).copy_(panel_h.slice(1, k+1, m_panel).select(2, k));
            }
            auto T = torch::zeros({batch, actual_nb, actual_nb}, A.options());
            for (int k = 0; k < actual_nb; k++) {
                T.select(1, k).select(1, k).copy_(panel_tau.select(1, k));
                if (k > 0) {
                    auto Vk = V.slice(2, k, k+1);
                    auto Vprev = V.slice(2, 0, k);
                    auto z = at::bmm(Vprev.transpose(1,2), Vk).squeeze(2);
                    auto Tprev = T.slice(1, 0, k).slice(2, 0, k);
                    z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                    auto tau_k = panel_tau.select(1, k).unsqueeze(1);
                    T.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
                }
            }
            auto trailing = H.slice(1, j, n).slice(2, j_end, n).contiguous();
            auto W1 = at::bmm(V.transpose(1,2), trailing);
            auto W2 = at::bmm(T.transpose(1,2), W1);
            H.slice(1, j, n).slice(2, j_end, n).sub_(at::bmm(V, W2));
        }
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
    name="qr_v13_full_dbl",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    result = module.dispatch_qr(data)
    return (result[0], result[1])
