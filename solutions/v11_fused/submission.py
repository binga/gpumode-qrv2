"""V11: Fused small kernel + C++ blocked QR for medium + geqrf fallback.

Tier 1 (n<=32): Fused CUDA kernel, entire matrix in shared memory.
Tier 2 (n<=512, batch>=16): C++ blocked QR (ATen geqrf panel + ATen bmm trailing).
Tier 3: torch.geqrf fallback.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

__device__ __forceinline__ float warp_sum_all(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

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
        } else {
            if (tid == 0) sTau[j] = 0.0f;
        }
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

// C++ blocked QR (same as V9): ATen geqrf panel + ATen bmm trailing
std::vector<torch::Tensor> blocked_qr(torch::Tensor A, int nb) {
    const int batch = A.size(0);
    const int n = A.size(1);
    auto H = A.clone();
    auto tau_all = torch::zeros({batch, n}, A.options());

    for (int j = 0; j < n; j += nb) {
        int actual_nb = std::min(nb, n - j);
        int j_end = j + actual_nb;
        int m = n - j;

        auto panel = H.slice(1, j, n).slice(2, j, j_end).contiguous();
        auto geqrf_result = at::geqrf(panel);
        auto panel_h = std::get<0>(geqrf_result);
        auto panel_tau = std::get<1>(geqrf_result);
        H.slice(1, j, n).slice(2, j, j_end).copy_(panel_h);
        tau_all.slice(1, j, j_end).copy_(panel_tau);

        int trailing_cols = n - j_end;
        if (trailing_cols <= 0) break;

        auto V = torch::zeros({batch, m, actual_nb}, A.options());
        for (int k = 0; k < actual_nb; k++) {
            V.select(1, k).select(1, k).fill_(1.0f);
            if (k + 1 < m) {
                V.slice(1, k + 1, m).select(2, k).copy_(
                    panel_h.slice(1, k + 1, m).select(2, k));
            }
        }

        auto T = torch::zeros({batch, actual_nb, actual_nb}, A.options());
        for (int k = 0; k < actual_nb; k++) {
            T.select(1, k).select(1, k).copy_(panel_tau.select(1, k));
            if (k > 0) {
                auto Vk = V.slice(2, k, k + 1);
                auto Vprev = V.slice(2, 0, k);
                auto z = at::bmm(Vprev.transpose(1, 2), Vk).squeeze(2);
                auto Tprev = T.slice(1, 0, k).slice(2, 0, k);
                z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                auto tau_k = panel_tau.select(1, k).unsqueeze(1);
                T.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
            }
        }

        auto trailing = H.slice(1, j, n).slice(2, j_end, n).contiguous();
        auto W1 = at::bmm(V.transpose(1, 2), trailing);
        auto W2 = at::bmm(T.transpose(1, 2), W1);
        H.slice(1, j, n).slice(2, j_end, n).sub_(at::bmm(V, W2));
    }
    return {H, tau_all};
}

std::vector<torch::Tensor> dispatch_qr(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 3 && A.size(1) == A.size(2));
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32);
    const int batch = A.size(0);
    const int n = A.size(1);

    if (n <= 32) {
        auto H = torch::empty_like(A);
        auto tau = torch::zeros({batch, n}, A.options());
        size_t smem_sz = (n * n + n) * sizeof(float);
        fused_qr_small<<<batch, 32, smem_sz>>>(
            A.data_ptr<float>(), H.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H, tau};
    }
    if (n <= 512 && batch >= 16) {
        return blocked_qr(A, 32);
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
    name="qr_v11_combined",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    result = module.dispatch_qr(data)
    return (result[0], result[1])
