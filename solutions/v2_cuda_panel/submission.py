"""V3: CUDA inline blocked Householder QR with cuBLAS trailing update.

Panel factorization: custom CUDA kernel with one thread-block per batch matrix,
multiple threads cooperating on the column operations within each panel.

Trailing update: cuBLAS batched GEMM with TF32 math mode for ~2x throughput.

Architecture:
  For each block of nb columns:
    1. Panel factorize (custom kernel): compute Householder vectors + tau
    2. Build WY compact form: T matrix (nb x nb upper triangular)
    3. Trailing update via GEMM: A[:,j+nb:] -= V @ T @ V^T @ A[:,j+nb:]
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_kernel = None

def _build_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel

    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// ------------------------------------------------------------------
// Panel factorization kernel
//
// Each thread-block handles one matrix in the batch.
// Within a block, threads cooperate on column norms and reflector
// application using shared memory.
//
// Operates in-place on A (batch, n, n) stored row-major.
// Produces Householder vectors below diagonal, R on/above diagonal,
// and tau vector — matching LAPACK's DGEQRF compact storage.
// ------------------------------------------------------------------

__global__ void panel_factorize(
    float* __restrict__ A,
    float* __restrict__ tau,
    int n,
    int col_start,
    int col_end
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float* a = A + (size_t)bid * n * n;
    float* t = tau + (size_t)bid * n;

    extern __shared__ float smem[];
    // smem layout: [partial_sums: nthreads] [dot_results: nthreads]
    float* partial = smem;
    float* dots    = smem + nthreads;

    for (int j = col_start; j < col_end && j < n; j++) {
        int rows_below = n - j - 1;

        // --- Step 1: Compute squared norm of a[j+1:n, j] ---
        float my_sum = 0.0f;
        for (int i = j + 1 + tid; i < n; i += nthreads) {
            float v = a[i * n + j];
            my_sum += v * v;
        }
        partial[tid] = my_sum;
        __syncthreads();

        // Reduce
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        float sigma_sq = partial[0];

        // --- Step 2: Compute Householder vector and tau ---
        float x0 = a[j * n + j];
        float tau_j = 0.0f;
        float inv_v0 = 0.0f;
        float r_jj = x0;

        if (sigma_sq > 1e-30f) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;

            float v_norm_sq = v0 * v0 + sigma_sq;
            tau_j = 2.0f / v_norm_sq;
            inv_v0 = 1.0f / v0;
            r_jj = -sign * norm_x;
        }

        // Store tau and diagonal
        if (tid == 0) {
            t[j] = tau_j;
            a[j * n + j] = r_jj;
        }

        // Normalize Householder vector below diagonal: v[i] /= v0
        for (int i = j + 1 + tid; i < n; i += nthreads) {
            a[i * n + j] *= inv_v0;
        }
        __syncthreads();

        if (tau_j == 0.0f) continue;

        // --- Step 3: Apply reflection to columns j+1..col_end-1 ---
        // For each column k: compute dot = v^T * A[:, k], then A[:, k] -= tau * dot * v
        // where v = [1, a[j+1,j], ..., a[n-1,j]]
        for (int k = j + 1; k < col_end && k < n; k++) {
            // Compute v^T * A[j:n, k]
            float my_dot = 0.0f;
            if (tid == 0) my_dot = a[j * n + k];  // v[0] = 1
            for (int i = j + 1 + tid; i < n; i += nthreads) {
                my_dot += a[i * n + j] * a[i * n + k];
            }
            dots[tid] = my_dot;
            __syncthreads();

            for (int s = nthreads / 2; s > 0; s >>= 1) {
                if (tid < s) dots[tid] += dots[tid + s];
                __syncthreads();
            }
            float dot_val = dots[0] * tau_j;

            // Update column k
            if (tid == 0) a[j * n + k] -= dot_val;
            for (int i = j + 1 + tid; i < n; i += nthreads) {
                a[i * n + k] -= a[i * n + j] * dot_val;
            }
            __syncthreads();
        }
    }
}

// ------------------------------------------------------------------
// Build WY compact form T matrix from Householder vectors and tau.
//
// T is nb x nb upper triangular such that the block reflector is:
//   I - V @ T @ V^T
// where V has the Householder vectors as columns.
//
// T[0,0] = tau[0]
// T[j,j] = tau[j]
// T[i,j] = -tau[j] * (T[0:j, 0:j] @ V[:, 0:j]^T @ v_j)[i] for i < j
//
// This is computed column by column on the host since nb is small (32).
// ------------------------------------------------------------------

torch::Tensor build_T_matrix(
    torch::Tensor H,       // (batch, n, n) with V below diagonal
    torch::Tensor tau,      // (batch, n)
    int j_start, int nb, int n
) {
    int batch = H.size(0);
    int actual_nb = std::min(nb, n - j_start);

    // T: (batch, actual_nb, actual_nb) upper triangular
    auto T = torch::zeros({batch, actual_nb, actual_nb}, H.options());

    // V: extract the Householder vectors from H
    // V[:, i] = [0...0, 1, H[j_start+i+1, j_start+i], ..., H[n-1, j_start+i]]
    int m = n - j_start;
    auto V = torch::zeros({batch, m, actual_nb}, H.options());
    for (int i = 0; i < actual_nb; i++) {
        V.index({torch::indexing::Slice(), i, i}) = 1.0f;
        if (j_start + i + 1 < n) {
            V.index({torch::indexing::Slice(),
                     torch::indexing::Slice(i + 1, m),
                     i}) =
                H.index({torch::indexing::Slice(),
                         torch::indexing::Slice(j_start + i + 1, n),
                         j_start + i});
        }
    }

    // Build T column by column
    for (int j = 0; j < actual_nb; j++) {
        auto tau_j = tau.index({torch::indexing::Slice(), j_start + j});
        T.index({torch::indexing::Slice(), j, j}) = tau_j;

        if (j > 0) {
            // z = -tau_j * T[0:j, 0:j] @ V[:, 0:j]^T @ V[:, j]
            auto Vj = V.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               j}).unsqueeze(2);  // (batch, m, 1)
            auto Vprev = V.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, j)});  // (batch, m, j)
            auto VtVj = torch::bmm(Vprev.transpose(1, 2), Vj).squeeze(2);  // (batch, j)

            auto Tprev = T.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(0, j),
                                  torch::indexing::Slice(0, j)});  // (batch, j, j)

            auto z = torch::bmm(Tprev, VtVj.unsqueeze(2)).squeeze(2);  // (batch, j)
            z = z * (-tau_j.unsqueeze(1));

            T.index({torch::indexing::Slice(),
                     torch::indexing::Slice(0, j),
                     j}) = z;
        }
    }

    return T;
}

// ------------------------------------------------------------------
// Main entry: blocked Householder QR
// ------------------------------------------------------------------

std::vector<torch::Tensor> blocked_qr(torch::Tensor A, int nb) {
    int batch = A.size(0);
    int n = A.size(1);

    auto H = A.clone();
    auto tau_out = torch::zeros({batch, n}, A.options());

    // Thread config for panel kernel
    int threads = std::min(256, ((n + 31) / 32) * 32);
    threads = std::max(threads, 32);
    size_t smem_bytes = 2 * threads * sizeof(float);

    for (int j = 0; j < n; j += nb) {
        int j_end = std::min(j + nb, n);

        // Panel factorization: factor columns j..j_end in-place
        panel_factorize<<<batch, threads, smem_bytes,
                          at::cuda::getCurrentCUDAStream()>>>(
            H.data_ptr<float>(),
            tau_out.data_ptr<float>(),
            n, j, j_end
        );

        // Trailing update: apply block reflector to columns j_end..n-1
        int trailing_cols = n - j_end;
        if (trailing_cols <= 0) continue;

        int actual_nb = j_end - j;
        int m = n - j;

        // Build WY compact form
        auto T = build_T_matrix(H, tau_out, j, nb, n);

        // Extract V from H (unit lower triangular in the panel region)
        auto V = torch::zeros({batch, m, actual_nb}, H.options());
        for (int i = 0; i < actual_nb; i++) {
            V.index({torch::indexing::Slice(), i, i}) = 1.0f;
            if (i + 1 < m) {
                V.index({torch::indexing::Slice(),
                         torch::indexing::Slice(i + 1, m),
                         i}) =
                    H.index({torch::indexing::Slice(),
                             torch::indexing::Slice(j + i + 1, n),
                             j + i});
            }
        }

        // W = V^T @ A[j:, j_end:]  — shape (batch, actual_nb, trailing_cols)
        auto trailing = H.index({torch::indexing::Slice(),
                                 torch::indexing::Slice(j, n),
                                 torch::indexing::Slice(j_end, n)}).clone();
        auto W = torch::bmm(V.transpose(1, 2), trailing);

        // W = T @ W  — shape (batch, actual_nb, trailing_cols)
        W = torch::bmm(T, W);

        // A[j:, j_end:] -= V @ W
        auto update = torch::bmm(V, W);
        H.index({torch::indexing::Slice(),
                 torch::indexing::Slice(j, n),
                 torch::indexing::Slice(j_end, n)}) -= update;
    }

    return {H, tau_out};
}

"""

    try:
        _kernel = load_inline(
            name="qr_v3",
            cpp_sources=[],
            cuda_sources=[cuda_src],
            functions=["blocked_qr"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            extra_ldflags=["-lcublas"],
            verbose=False,
        )
    except Exception as e:
        print(f"CUDA compile failed: {e}")
        _kernel = None
    return _kernel


def custom_kernel(data: input_t) -> output_t:
    if not data.is_cuda:
        return torch.geqrf(data)

    n = data.shape[-1]
    kernel = _build_kernel()
    if kernel is not None:
        nb = 32 if n >= 256 else n
        result = kernel.blocked_qr(data, nb)
        return (result[0], result[1])

    return torch.geqrf(data)
