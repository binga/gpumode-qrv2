"""V12: Direct cuSOLVER batched geqrf + fused n<=32.

Tests whether calling cusolverDnSgeqrfBatched directly (array-of-pointers
interface, single API call for all 640 matrices) is faster than ATen's
at::geqrf path which may loop internally.

The blocked QR with cuSOLVER panel + cuBLAS trailing is replaced with
a SINGLE cusolverDnSgeqrfBatched call that factors all matrices at once.
No blocking, no V/T construction, no trailing update — just one library call.

If this is fast, it means cuSOLVER's internal batched implementation
is already doing the right thing and we should just call it directly.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>

// Fused kernel for n<=32 (same as V11)
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

// Direct cuSOLVER batched geqrf
static cusolverDnHandle_t g_handle = nullptr;

std::vector<torch::Tensor> cusolver_batched_qr(torch::Tensor A) {
    const int batch = A.size(0);
    const int n = A.size(1);

    // cuSOLVER expects column-major. PyTorch tensors are row-major.
    // Transpose to get column-major layout for cuSOLVER, then transpose back.
    // A_col = A.transpose(1,2).contiguous() gives column-major in memory.
    auto H = A.transpose(1, 2).contiguous();  // now (batch, n, n) column-major equivalent
    auto tau = torch::zeros({batch, n}, A.options());

    if (!g_handle) {
        cusolverDnCreate(&g_handle);
    }

    // Allocate array of pointers for batched API
    std::vector<float*> h_ptrs(batch);
    std::vector<float*> tau_ptrs(batch);
    for (int i = 0; i < batch; i++) {
        h_ptrs[i] = H.data_ptr<float>() + (size_t)i * n * n;
        tau_ptrs[i] = tau.data_ptr<float>() + (size_t)i * n;
    }

    // Copy pointer arrays to device
    float** d_A_array;
    float** d_tau_array;
    cudaMalloc(&d_A_array, batch * sizeof(float*));
    cudaMalloc(&d_tau_array, batch * sizeof(float*));
    cudaMemcpy(d_A_array, h_ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau_array, tau_ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice);

    // Query workspace
    int lwork = 0;
    cusolverDnSgeqrf_bufferSize(g_handle, n, n, H.data_ptr<float>(), n, &lwork);

    auto workspace = torch::empty({lwork}, A.options());
    auto info = torch::zeros({batch}, A.options().dtype(torch::kInt32));

    // Call batched geqrf
    // Note: cusolverDnSgeqrfBatched processes all batch elements
    cusolverDnSgeqrfBatched(
        g_handle,
        n, n,
        d_A_array, n,
        d_tau_array,
        info.data_ptr<int>(),
        batch
    );
    cudaDeviceSynchronize();

    cudaFree(d_A_array);
    cudaFree(d_tau_array);

    // Transpose result back to row-major
    H = H.transpose(1, 2).contiguous();

    return {H, tau};
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
    // Try direct cuSOLVER batched for everything else
    return cusolver_batched_qr(A);
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dispatch_qr(torch::Tensor A);
"""

module = load_inline(
    name="qr_v12_cusolver_direct",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_ldflags=["-lcusolver"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    result = module.dispatch_qr(data)
    return (result[0], result[1])
