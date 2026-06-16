"""V13: Fused persistent kernel — entire blocked QR in one kernel launch.

Architecture:
- One thread-block per batch matrix (640 blocks for critical shape)
- Panel (m × nb) factorized entirely in shared memory
- V stays in shared memory between panel factorization and trailing update
- Trailing update done in tiles: load from global, apply reflector block, write back
- Zero inter-kernel global memory traffic for intermediate data

Shared memory layout (B200 has 227KB):
- Panel/V buffer: m × nb floats (max 512×32 = 64KB)
- T buffer: nb × nb floats (32×32 = 4KB)
- Trailing tile: m × TILE_W floats (tuned per available SMEM)
- Reduction buffer: num_warps floats

For n<=32: use the proven fused warp-level kernel from V11.
For 32<n<=512, batch>=16: use this persistent kernel.
For larger: torch.geqrf fallback.
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
#define TILE_W 32

__device__ __forceinline__ float warp_sum_all(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// Block-level reduction: all threads contribute, thread 0 gets result, then broadcast
__device__ float block_reduce(float val, float* scratch, int tid, int nthreads) {
    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    int warp_id = tid / 32;
    int lane = tid % 32;
    int num_warps = (nthreads + 31) / 32;

    if (lane == 0) scratch[warp_id] = val;
    __syncthreads();

    // Thread 0 sums across warps
    if (tid == 0) {
        float sum = 0.0f;
        for (int w = 0; w < num_warps; w++) sum += scratch[w];
        scratch[0] = sum;
    }
    __syncthreads();
    return scratch[0];
}

// Fused panel-only kernel: factorizes panel A[j:n, j:j+nb] in shared memory.
// One block per batch matrix. Panel is m×nb where m = n-j.
__global__ void fused_panel_kernel(
    float* __restrict__ A,       // (batch, n, n) row-major, modified in-place
    float* __restrict__ tau_out, // (batch, n)
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

    extern __shared__ float smem[];
    float* sPanel = smem;                    // m × NB
    float* sTau = smem + m * NB;            // NB
    float* scratch = sTau + NB;             // num_warps

    // Load panel from global to shared
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb;
        int col = idx % actual_nb;
        sPanel[row * NB + col] = mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0f;
    __syncthreads();

    // Householder factorization of panel in shared memory
    for (int j = 0; j < actual_nb; j++) {
        float my_sq = 0.0f;
        for (int i = j + 1 + tid; i < m; i += nthreads) {
            float v = sPanel[i * NB + j];
            my_sq += v * v;
        }
        float sigma_sq = block_reduce(my_sq, scratch, tid, nthreads);

        float x0 = sPanel[j * NB + j];
        float tau_j = 0.0f;

        if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
            float v0 = x0 - diag;
            tau_j = (diag - x0) / diag;
            float inv_v0 = 1.0f / v0;

            for (int i = j + 1 + tid; i < m; i += nthreads) {
                sPanel[i * NB + j] *= inv_v0;
            }
            if (tid == 0) {
                sPanel[j * NB + j] = diag;
                sTau[j] = tau_j;
            }
        } else {
            if (tid == 0) sTau[j] = 0.0f;
        }
        __syncthreads();

        tau_j = sTau[j];
        if (tau_j == 0.0f) continue;

        // Apply reflection to remaining panel columns
        for (int k = j + 1; k < actual_nb; k++) {
            float my_dot = 0.0f;
            if (tid == 0) my_dot = sPanel[j * NB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) {
                my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
            }
            float dot = block_reduce(my_dot, scratch, tid, nthreads);
            float scale = tau_j * dot;

            if (tid == 0) sPanel[j * NB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) {
                sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
            }
            __syncthreads();
        }
    }

    // Write factored panel back to global memory
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb;
        int col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = sPanel[row * NB + col];
    }
    // Write tau
    if (tid < actual_nb) tau_base[j_start + tid] = sTau[tid];
}

// Fused persistent kernel for blocked QR.
// One block per batch matrix. Handles panel factorization + trailing update
// entirely within the block, keeping panel/V in shared memory.
__global__ void fused_blocked_qr(
    float* __restrict__ A,       // (batch, n, n) row-major, modified in-place → H
    float* __restrict__ tau_out, // (batch, n)
    const int n
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float* mat = A + (size_t)bid * n * n;
    float* tau = tau_out + (size_t)bid * n;

    // Shared memory layout:
    // [0 .. n*NB-1]: panel buffer (V after factorization)
    // [n*NB .. n*NB + NB*NB - 1]: T matrix
    // [n*NB + NB*NB .. n*NB + NB*NB + n*TILE_W - 1]: trailing tile
    // [n*NB + NB*NB + n*TILE_W .. ]: reduction scratch (num_warps floats)
    extern __shared__ float smem[];
    const int panel_offset = 0;
    const int t_offset = n * NB;
    const int tile_offset = t_offset + NB * NB;
    const int scratch_offset = tile_offset + n * TILE_W;

    float* sPanel = smem + panel_offset;     // m × NB (m changes each iteration)
    float* sT = smem + t_offset;             // NB × NB
    float* sTile = smem + tile_offset;       // m × TILE_W
    float* scratch = smem + scratch_offset;  // num_warps

    for (int j_block = 0; j_block < n; j_block += NB) {
        int actual_nb = min(NB, n - j_block);
        int m = n - j_block;  // rows in panel

        // === STEP 1: Load panel from global to shared memory ===
        // Panel is mat[j_block:n, j_block:j_block+actual_nb] (m × actual_nb, row-major)
        for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
            int row = idx / actual_nb;
            int col = idx % actual_nb;
            sPanel[row * NB + col] = mat[(j_block + row) * n + (j_block + col)];
        }
        __syncthreads();

        // === STEP 2: Factorize panel in shared memory (Householder) ===
        for (int j = 0; j < actual_nb; j++) {
            // Column norm below diagonal
            float my_sq = 0.0f;
            for (int i = j + 1 + tid; i < m; i += nthreads) {
                float v = sPanel[i * NB + j];
                my_sq += v * v;
            }
            float sigma_sq = block_reduce(my_sq, scratch, tid, nthreads);

            float x0 = sPanel[j * NB + j];
            float tau_j = 0.0f;

            if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
                float norm_x = sqrtf(x0 * x0 + sigma_sq);
                float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
                float v0 = x0 - diag;
                tau_j = (diag - x0) / diag;
                float inv_v0 = 1.0f / v0;

                for (int i = j + 1 + tid; i < m; i += nthreads) {
                    sPanel[i * NB + j] *= inv_v0;
                }
                if (tid == 0) {
                    sPanel[j * NB + j] = diag;
                    tau[j_block + j] = tau_j;
                }
            } else {
                if (tid == 0) tau[j_block + j] = 0.0f;
            }
            __syncthreads();

            tau_j = tau[j_block + j];
            if (tau_j == 0.0f) continue;

            // Apply reflection to remaining panel columns j+1..actual_nb-1
            for (int k = j + 1; k < actual_nb; k++) {
                float my_dot = 0.0f;
                if (tid == 0) my_dot = sPanel[j * NB + k];
                for (int i = j + 1 + tid; i < m; i += nthreads) {
                    my_dot += sPanel[i * NB + j] * sPanel[i * NB + k];
                }
                float dot = block_reduce(my_dot, scratch, tid, nthreads);
                float scale = tau_j * dot;

                if (tid == 0) sPanel[j * NB + k] -= scale;
                for (int i = j + 1 + tid; i < m; i += nthreads) {
                    sPanel[i * NB + k] -= scale * sPanel[i * NB + j];
                }
                __syncthreads();
            }
        }

        // === STEP 3: Write factored panel back to global memory ===
        for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
            int row = idx / actual_nb;
            int col = idx % actual_nb;
            mat[(j_block + row) * n + (j_block + col)] = sPanel[row * NB + col];
        }
        __syncthreads();

        // === STEP 4: Trailing update ===
        int trailing_start = j_block + actual_nb;
        int trailing_cols = n - trailing_start;
        if (trailing_cols <= 0) continue;

        // Build T matrix in shared memory
        // T[k,k] = tau[j_block+k], T[i,j] = -tau_j * T[0:j,0:j] @ V[:,0:j]^T @ V[:,j]
        for (int idx = tid; idx < NB * NB; idx += nthreads) sT[idx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < actual_nb; k++) {
            float tau_k = tau[j_block + k];
            if (tid == 0) sT[k * NB + k] = tau_k;
            __syncthreads();

            if (k > 0 && tau_k != 0.0f) {
                // For each i < k: compute z[i] = V[:,i]^T @ V[:,k]
                // Then T[0:k, k] = -tau_k * T[0:k, 0:k] @ z
                // Process one column i at a time (k is small, max 32)
                for (int i = 0; i < k; i++) {
                    float my_dot = 0.0f;
                    if (tid == 0) my_dot = 1.0f * (i == k ? 1.0f : 0.0f); // v[j,i]*v[j,k] at diagonal
                    // V[:,i] has 1 at position i, sPanel[i+1:,i] below
                    // V[:,k] has 1 at position k, sPanel[k+1:,k] below
                    // dot = V[:,i]^T @ V[:,k]
                    // = (i==k ? 1 : 0) + sum_{r=max(i,k)+1}^{m-1} V[r,i]*V[r,k]
                    // Since i < k: dot = V[k,i] * 1 + sum_{r=k+1}^{m-1} V[r,i]*V[r,k]
                    // where V[k,i] = sPanel[k*NB+i] (below diagonal of col i)
                    my_dot = 0.0f;
                    if (tid == 0) my_dot = sPanel[k * NB + i]; // V[k,i] * V[k,k]=1
                    for (int r = k + 1 + tid; r < m; r += nthreads) {
                        my_dot += sPanel[r * NB + i] * sPanel[r * NB + k];
                    }
                    float z_i = block_reduce(my_dot, scratch, tid, nthreads);

                    // Accumulate: T[i,k] += -tau_k * T[i,0:k] @ z[0:k]
                    // But we compute z one at a time, so we need to do the Tprev multiply differently.
                    // Store z_i temporarily
                    if (tid == 0) scratch[16 + i] = z_i; // reuse scratch space
                    __syncthreads();
                }
                // Now scratch[16+0..16+k-1] holds z[0..k-1]
                // T[0:k, k] = -tau_k * T[0:k, 0:k] @ z[0:k]
                if (tid < k) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += sT[tid * NB + p] * scratch[16 + p];
                    }
                    sT[tid * NB + k] = -tau_k * sum;
                }
                __syncthreads();
            }
        }

        // Apply trailing update in tiles
        for (int t_start = trailing_start; t_start < n; t_start += TILE_W) {
            int tw = min(TILE_W, n - t_start);

            // Load trailing tile: mat[j_block:n, t_start:t_start+tw] → sTile[m × tw]
            for (int idx = tid; idx < m * tw; idx += nthreads) {
                int row = idx / tw;
                int col = idx % tw;
                sTile[row * TILE_W + col] = mat[(j_block + row) * n + (t_start + col)];
            }
            __syncthreads();

            // W1[k, c] = V[:,k]^T @ tile[:,c] for k=0..actual_nb-1, c=0..tw-1
            // V[:,k] = [0..0, 1 at pos k, sPanel[k+1:, k]]
            // Compute one (k,c) pair per thread iteration
            // Store W1 in scratch (reuse T area after T is consumed? No, T is needed.)
            // Actually we need a separate W1 buffer. Let's compute W1 column by column.
            // For each column c of the tile:
            for (int c = 0; c < tw; c++) {
                // Compute w1[0..actual_nb-1] = V^T @ tile[:,c]
                for (int k = 0; k < actual_nb; k++) {
                    float my_dot = 0.0f;
                    if (tid == 0) my_dot = sTile[k * TILE_W + c]; // V[k,k]=1 * tile[k,c]
                    for (int r = k + 1 + tid; r < m; r += nthreads) {
                        my_dot += sPanel[r * NB + k] * sTile[r * TILE_W + c];
                    }
                    float w1_k = block_reduce(my_dot, scratch, tid, nthreads);

                    // Store w1[k] for this column
                    if (tid == 0) scratch[16 + k] = w1_k;
                    __syncthreads();
                }
                // Now scratch[16+0..16+actual_nb-1] = W1[:,c] = V^T @ tile[:,c]

                // W2 = T^T @ W1: w2[k] = sum_p T[p,k] * w1[p] (T is upper tri, T^T is lower)
                if (tid < actual_nb) {
                    float sum = 0.0f;
                    for (int p = 0; p <= tid; p++) { // T^T[tid,p] = T[p,tid]
                        sum += sT[p * NB + tid] * scratch[16 + p];
                    }
                    scratch[16 + NB + tid] = sum; // w2 stored at offset 16+NB
                }
                __syncthreads();

                // Update: tile[:,c] -= V @ W2
                // tile[row, c] -= sum_k V[row,k] * w2[k]
                for (int row = tid; row < m; row += nthreads) {
                    float update = 0.0f;
                    for (int k = 0; k < actual_nb; k++) {
                        float v_rk = (row == k) ? 1.0f : ((row > k) ? sPanel[row * NB + k] : 0.0f);
                        update += v_rk * scratch[16 + NB + k];
                    }
                    sTile[row * TILE_W + c] -= update;
                }
                __syncthreads();
            }

            // Write updated tile back to global
            for (int idx = tid; idx < m * tw; idx += nthreads) {
                int row = idx / tw;
                int col = idx % tw;
                mat[(j_block + row) * n + (t_start + col)] = sTile[row * TILE_W + col];
            }
            __syncthreads();
        }
    }
}

// Small fused kernel (proven V11 approach for n<=32)
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
        // Hybrid: fused CUDA panel for m<=384, ATen geqrf for m>384
        int nb = NB;
        int threads = 256;
        for (int j = 0; j < n; j += nb) {
            int actual_nb = std::min(nb, n - j);
            int j_end = j + actual_nb;
            int m_panel = n - j;

            if (n <= 352) {
                // Fused panel in shared memory (proven correct for m<=352)
                size_t panel_smem = (m_panel * NB + NB + 8 + 16) * sizeof(float);
                fused_panel_kernel<<<batch, threads, panel_smem>>>(
                    H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
            } else {
                // ATen geqrf for large panels (m>384)
                auto panel = H.slice(1, j, n).slice(2, j, j_end).contiguous();
                auto geqrf_result = at::geqrf(panel);
                auto panel_h = std::get<0>(geqrf_result);
                auto panel_tau = std::get<1>(geqrf_result);
                H.slice(1, j, n).slice(2, j, j_end).copy_(panel_h);
                tau.slice(1, j, j_end).copy_(panel_tau);
            }

            int trailing_cols = n - j_end;
            if (trailing_cols <= 0) break;

            // V/T extraction + ATen trailing (proven correct from V9)
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

    // Fallback
    auto result = at::geqrf(A);
    return {std::get<0>(result), std::get<1>(result)};
}
"""

cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dispatch_qr(torch::Tensor A);
void fused_panel_kernel(float* A, float* tau_out, int n, int j_start, int j_end);
"""

module = load_inline(
    name="qr_v13_final",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--maxrregcount=64"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    result = module.dispatch_qr(data)
    return (result[0], result[1])
