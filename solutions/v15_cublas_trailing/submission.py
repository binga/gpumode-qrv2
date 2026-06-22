"""V15: V13 fused panel with direct cuBLAS trailing updates.

V13's largest remaining overhead is the per-block ATen path used to build
V/T and apply the WY trailing update. This version keeps the numerically
stable double-accumulation panel, but moves the medium-size trailing path to
preallocated workspaces plus direct strided-batched cuBLAS calls.

Tier 1 (n<=32): fused warp-level kernel in SMEM
Tier 2 (n<=512, batch>=16): fused panel + CUDA V/T build + cuBLAS trailing
Tier 3: torch.geqrf fallback
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define NB 32

#define CUBLAS_CHECK(expr)                                                    \
    do {                                                                      \
        cublasStatus_t status = (expr);                                       \
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed: ",  \
                    static_cast<int>(status));                                \
    } while (0)

static cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    }
    return g_cublas_handle;
}

__device__ __forceinline__ float warp_sum_all(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ double block_reduce_d(double val, double* scratch, int tid, int nthreads) {
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
    double* sPanel = (double*)smem_raw;
    double* sTau = sPanel + m * NB;
    double* scratch = sTau + NB;

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

    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb;
        int col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = (float)sPanel[row * NB + col];
    }
    if (tid < actual_nb) tau_base[j_start + tid] = (float)sTau[tid];
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

__global__ void build_v_kernel(
    const float* __restrict__ H,
    float* __restrict__ V,
    const int n,
    const int j_start,
    const int actual_nb
) {
    const int bid = blockIdx.y;
    const int m = n - j_start;
    const int per_matrix = m * actual_nb;
    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < per_matrix;
         linear += (size_t)gridDim.x * blockDim.x) {
        int row = linear / actual_nb;
        int col = linear % actual_nb;

        float value = 0.0f;
        if (row == col) {
            value = 1.0f;
        } else if (row > col) {
            value = H[(size_t)bid * n * n + (j_start + row) * n + (j_start + col)];
        }
        V[(size_t)bid * n * NB + row * NB + col] = value;
    }
}

__global__ void build_t_kernel(
    const float* __restrict__ V,
    const float* __restrict__ tau,
    float* __restrict__ T,
    const int n,
    const int j_start,
    const int actual_nb
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int m = n - j_start;
    float* t = T + (size_t)bid * NB * NB;
    const float* v = V + (size_t)bid * n * NB;
    const float* tau_b = tau + (size_t)bid * n + j_start;

    for (int idx = tid; idx < NB * NB; idx += blockDim.x) {
        t[idx] = 0.0f;
    }
    __syncthreads();

    if (tid != 0) return;

    double dots[NB];
    double work[NB];

    for (int k = 0; k < actual_nb; k++) {
        float tau_k = tau_b[k];
        t[k * NB + k] = tau_k;

        if (k == 0) continue;

        for (int i = 0; i < k; i++) {
            double dot = 0.0;
            for (int r = 0; r < m; r++) {
                dot += (double)v[r * NB + i] * (double)v[r * NB + k];
            }
            dots[i] = dot;
        }

        for (int i = 0; i < k; i++) {
            double acc = 0.0;
            for (int p = 0; p < k; p++) {
                acc += (double)t[i * NB + p] * dots[p];
            }
            work[i] = acc;
        }

        for (int i = 0; i < k; i++) {
            t[i * NB + k] = (float)(-(double)tau_k * work[i]);
        }
    }
}

void row_major_gemm_strided_batched(
    cublasHandle_t handle,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    int lda_row,
    long long strideA,
    const float* B,
    int ldb_row,
    long long strideB,
    float beta,
    float* C,
    int ldc_row,
    long long strideC,
    int batch
) {
    cublasOperation_t opA_col = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB_col = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        opB_col,
        opA_col,
        N,
        M,
        K,
        &alpha,
        B,
        ldb_row,
        strideB,
        A,
        lda_row,
        strideA,
        &beta,
        C,
        ldc_row,
        strideC,
        batch
    ));
}

std::vector<torch::Tensor> dispatch_qr(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 3 && A.size(1) == A.size(2));
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32);
    const c10::cuda::CUDAGuard device_guard(A.device());

    const int batch = A.size(0);
    const int n = A.size(1);

    auto tau = torch::zeros({batch, n}, A.options());

    if (n <= 32) {
        auto H_small = torch::empty_like(A);
        size_t smem_sz = (n * n + n) * sizeof(float);
        fused_qr_small<<<batch, 32, smem_sz>>>(
            A.data_ptr<float>(), H_small.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H_small, tau};
    }

    if (n <= 512 && batch >= 16) {
        auto H = A.clone();
        if (n == 512 && batch >= 128) {
            int threads = 256;
            int num_warps = threads / 32;
            size_t max_smem = (n * NB + NB + num_warps) * sizeof(double);
            cudaFuncSetAttribute(fused_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);

            for (int j = 0; j < n; j += NB) {
                int actual_nb = std::min(NB, n - j);
                int j_end = j + actual_nb;
                int m_panel = n - j;

                size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(double);
                fused_panel_kernel<<<batch, threads, panel_smem>>>(
                    H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                int trailing_cols = n - j_end;
                if (trailing_cols <= 0) break;

                auto panel_h = H.slice(1, j, n).slice(2, j, j_end).contiguous();
                auto panel_tau = tau.slice(1, j, j_end);
                auto V_aten = torch::zeros({batch, m_panel, actual_nb}, A.options());
                for (int k = 0; k < actual_nb; k++) {
                    V_aten.select(1, k).select(1, k).fill_(1.0f);
                    if (k + 1 < m_panel)
                        V_aten.slice(1, k + 1, m_panel).select(2, k).copy_(
                            panel_h.slice(1, k + 1, m_panel).select(2, k));
                }
                auto T_aten = torch::zeros({batch, actual_nb, actual_nb}, A.options());
                for (int k = 0; k < actual_nb; k++) {
                    T_aten.select(1, k).select(1, k).copy_(panel_tau.select(1, k));
                    if (k > 0) {
                        auto Vk = V_aten.slice(2, k, k + 1);
                        auto Vprev = V_aten.slice(2, 0, k);
                        auto z = at::bmm(Vprev.transpose(1, 2), Vk).squeeze(2);
                        auto Tprev = T_aten.slice(1, 0, k).slice(2, 0, k);
                        z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                        auto tau_k = panel_tau.select(1, k).unsqueeze(1);
                        T_aten.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
                    }
                }
                auto trailing = H.slice(1, j, n).slice(2, j_end, n).contiguous();
                auto W1_aten = at::bmm(V_aten.transpose(1, 2), trailing);
                auto W2_aten = at::bmm(T_aten.transpose(1, 2), W1_aten);
                H.slice(1, j, n).slice(2, j_end, n).sub_(at::bmm(V_aten, W2_aten));
            }
            return {H, tau};
        }

        auto V = torch::empty({batch, n, NB}, A.options());
        auto T = torch::empty({batch, NB, NB}, A.options());
        auto W1 = torch::empty({batch, NB, n}, A.options());
        auto W2 = torch::empty({batch, NB, n}, A.options());

        int threads = 256;
        int num_warps = threads / 32;
        size_t max_smem = (n * NB + NB + num_warps) * sizeof(double);
        cudaFuncSetAttribute(fused_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);

        cublasHandle_t handle = get_cublas_handle();
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

        const long long h_stride = (long long)n * n;
        const long long v_stride = (long long)n * NB;
        const long long t_stride = (long long)NB * NB;
        const long long w_stride = (long long)NB * n;

        for (int j = 0; j < n; j += NB) {
            int actual_nb = std::min(NB, n - j);
            int j_end = j + actual_nb;
            int m_panel = n - j;
            int trailing_cols = n - j_end;

            size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(double);
            fused_panel_kernel<<<batch, threads, panel_smem>>>(
                H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            if (trailing_cols <= 0) break;

            int per_matrix_v = m_panel * actual_nb;
            int v_blocks = std::min<int>((per_matrix_v + threads - 1) / threads, 65535);
            build_v_kernel<<<dim3(v_blocks, batch), threads>>>(
                H.data_ptr<float>(), V.data_ptr<float>(), n, j, actual_nb);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            build_t_kernel<<<batch, 128>>>(
                V.data_ptr<float>(), tau.data_ptr<float>(), T.data_ptr<float>(), n, j, actual_nb);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            const float one = 1.0f;
            const float zero = 0.0f;
            const float neg_one = -1.0f;
            float* trailing = H.data_ptr<float>() + (size_t)j * n + j_end;

            // W1 = V^T @ A[j:, j_end:]
            row_major_gemm_strided_batched(
                handle, true, false,
                actual_nb, trailing_cols, m_panel,
                one,
                V.data_ptr<float>(), NB, v_stride,
                trailing, n, h_stride,
                zero,
                W1.data_ptr<float>(), n, w_stride,
                batch);

            // W2 = T^T @ W1
            row_major_gemm_strided_batched(
                handle, true, false,
                actual_nb, trailing_cols, actual_nb,
                one,
                T.data_ptr<float>(), NB, t_stride,
                W1.data_ptr<float>(), n, w_stride,
                zero,
                W2.data_ptr<float>(), n, w_stride,
                batch);

            // A[j:, j_end:] -= V @ W2
            row_major_gemm_strided_batched(
                handle, false, false,
                m_panel, trailing_cols, actual_nb,
                neg_one,
                V.data_ptr<float>(), NB, v_stride,
                W2.data_ptr<float>(), n, w_stride,
                one,
                trailing, n, h_stride,
                batch);
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
    name="qr_v15_cublas_trailing",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_ldflags=["-lcublas"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    result = module.dispatch_qr(data)
    return (result[0], result[1])
