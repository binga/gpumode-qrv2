"""V25: 2-level blocked Householder QR.

Competitor hint: `submission_2level.py` at 3.4ms.

Architecture:
- Outer block NB=128, inner block ib=32
- Inner loop: panel(ib=32) + small trailing update within outer block only
- After inner loop: accumulate outer V(m x NB) and T(NB x NB), then ONE big trailing GEMM
- Result: 4 large outer trailing GEMMs instead of 16 small ones → much better tensor core utilization

For n=512: 4 outer steps × (4 inner panels + 1 big trailing GEMM) = 20 operations
vs V24: 16 steps × (panel + V/T + 3 bmm) = 80 operations

Tier 1 (n=32): CuTe DSL QR32
Tier 2 (n<=512, batch>=16): 2-level blocked Householder
Tier 3: torch.geqrf fallback
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_cute_qr32_status = "untried"
_cute_qr32_error = ""
_cute_qr32_compiled = None
_cute_qr32_defs_ready = False


def _ensure_cute_qr32(data, H, tau):
    global _cute_qr32_status, _cute_qr32_error, _cute_qr32_compiled, _cute_qr32_defs_ready
    if _cute_qr32_status in ("ready", "unavailable"):
        return _cute_qr32_status == "ready"
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception:
        _cute_qr32_status = "unavailable"; return False
    if not _cute_qr32_defs_ready:
        try:
            g = globals()
            @cute.kernel
            def _qr32_kernel(a: cute.Tensor, h: cute.Tensor, tau_out: cute.Tensor):
                bid, _, _ = cute.arch.block_idx(); tid, _, _ = cute.arch.thread_idx(); n = 32
                alloc = cutlass.utils.SmemAllocator()
                s_h = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((32,32)), byte_alignment=16, swizzle=None)
                s_tau = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), byte_alignment=16, swizzle=None)
                for idx in range(tid, 1024, 32):
                    row = idx // n; col = idx - row * n; s_h[(row,col)] = a[(bid,row,col)]
                if tid < n: s_tau[tid] = 0.0
                cute.arch.sync_threads()
                for j in range(32):
                    my_sq = 0.0
                    if tid > j: v = s_h[(tid,j)]; my_sq = v*v
                    sigma_sq = cute.arch.warp_reduction_sum(my_sq, threads_in_group=32)
                    x0 = s_h[(j,j)]; tau_j = 0.0
                    if sigma_sq > 0.0 or (sigma_sq == 0.0 and x0 < 0.0):
                        norm_x = 1.0 / cute.math.rsqrt(x0*x0+sigma_sq)
                        diag = -norm_x
                        if x0 < 0.0: diag = norm_x
                        v0 = x0-diag; tau_j = (diag-x0)/diag; inv_v0 = 1.0/v0
                        if tid > j: s_h[(tid,j)] = s_h[(tid,j)]*inv_v0
                        if tid == 0: s_h[(j,j)] = diag; s_tau[j] = tau_j
                    else:
                        if tid == 0: s_tau[j] = 0.0
                    cute.arch.sync_threads()
                    tau_j = s_tau[j]
                    if tau_j != 0.0:
                        for k in range(j+1, 32):
                            my_dot = 0.0
                            if tid == j: my_dot = s_h[(j,k)]
                            if tid > j: my_dot = s_h[(tid,j)]*s_h[(tid,k)]
                            dot = cute.arch.warp_reduction_sum(my_dot, threads_in_group=32)
                            scale = tau_j*dot
                            if tid == j: s_h[(j,k)] = s_h[(j,k)] - scale
                            if tid > j: s_h[(tid,k)] = s_h[(tid,k)] - scale*s_h[(tid,j)]
                            cute.arch.sync_threads()
                for idx in range(tid, 1024, 32):
                    row = idx // n; col = idx - row * n; h[(bid,row,col)] = s_h[(row,col)]
                if tid < n: tau_out[(bid,tid)] = s_tau[tid]
            @cute.jit
            def _qr32_launch(a: cute.Tensor, h: cute.Tensor, tau_out: cute.Tensor):
                _qr32_kernel(a, h, tau_out).launch(grid=(a.shape[0],1,1), block=(32,1,1))
            g["_cute_qr32_launch"] = _qr32_launch; g["_cute_from_dlpack"] = from_dlpack
            _cute_qr32_defs_ready = True
        except Exception: _cute_qr32_status = "unavailable"; return False
    try:
        fdl = globals()["_cute_from_dlpack"]
        ac = fdl(data).mark_layout_dynamic(); hc = fdl(H).mark_layout_dynamic(); tc = fdl(tau).mark_layout_dynamic()
        _cute_qr32_compiled = cute.compile(globals()["_cute_qr32_launch"], ac, hc, tc)
        _cute_qr32_compiled(ac, hc, tc); torch.cuda.synchronize()
        _cute_qr32_status = "ready"; return True
    except Exception: _cute_qr32_status = "unavailable"; return False


def _try_cute_qr32(data):
    if data.dim() != 3 or data.size(1) != 32 or data.size(2) != 32: return None
    H = torch.empty_like(data); tau = torch.empty((data.size(0),32), device=data.device, dtype=data.dtype)
    if not _ensure_cute_qr32(data, H, tau): return None
    fdl = globals()["_cute_from_dlpack"]
    _cute_qr32_compiled(fdl(data).mark_layout_dynamic(), fdl(H).mark_layout_dynamic(), fdl(tau).mark_layout_dynamic())
    return H, tau


cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define IB 32
#define OB 128

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
    float* sPanel = smem, *sTau = sPanel + m * IB, *scratch = sTau + IB;
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        sPanel[row * IB + col] = mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0f;
    __syncthreads();
    for (int j = 0; j < actual_nb; j++) {
        float my_sq = 0.0f;
        for (int i = j + 1 + tid; i < m; i += nthreads) { float v = sPanel[i * IB + j]; my_sq += v * v; }
        float sigma_sq = block_reduce_f(my_sq, scratch, tid, nthreads);
        float x0 = sPanel[j * IB + j], tau_j = 0.0f;
        if (sigma_sq > 0.0f || (sigma_sq == 0.0f && x0 < 0.0f)) {
            float norm_x = sqrtf(x0 * x0 + sigma_sq);
            float diag = (x0 >= 0.0f) ? -norm_x : norm_x;
            float v0 = x0 - diag; tau_j = (diag - x0) / diag;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * IB + j] *= (1.0f / v0);
            if (tid == 0) { sPanel[j * IB + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0f; }
        __syncthreads();
        tau_j = sTau[j]; if (tau_j == 0.0f) continue;
        for (int k = j + 1; k < actual_nb; k++) {
            float my_dot = 0.0f;
            if (tid == 0) my_dot = sPanel[j * IB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) my_dot += sPanel[i * IB + j] * sPanel[i * IB + k];
            float dot = block_reduce_f(my_dot, scratch, tid, nthreads);
            float scale = tau_j * dot;
            if (tid == 0) sPanel[j * IB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * IB + k] -= scale * sPanel[i * IB + j];
            __syncthreads();
        }
    }
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = sPanel[row * IB + col];
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
    double* sPanel = (double*)smem_raw, *sTau = sPanel + m * IB, *scratch = sTau + IB;
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        sPanel[row * IB + col] = (double)mat[(j_start + row) * n + (j_start + col)];
    }
    if (tid < actual_nb) sTau[tid] = 0.0;
    __syncthreads();
    for (int j = 0; j < actual_nb; j++) {
        double my_sq = 0.0;
        for (int i = j + 1 + tid; i < m; i += nthreads) { double v = sPanel[i * IB + j]; my_sq += v * v; }
        double sigma_sq = block_reduce_d(my_sq, scratch, tid, nthreads);
        double x0 = sPanel[j * IB + j], tau_j = 0.0;
        if (sigma_sq > 0.0 || (sigma_sq == 0.0 && x0 < 0.0)) {
            double norm_x = sqrt(x0 * x0 + sigma_sq);
            double diag = (x0 >= 0.0) ? -norm_x : norm_x;
            double v0 = x0 - diag; tau_j = (diag - x0) / diag;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * IB + j] *= (1.0 / v0);
            if (tid == 0) { sPanel[j * IB + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0; }
        __syncthreads();
        tau_j = sTau[j]; if (tau_j == 0.0) continue;
        for (int k = j + 1; k < actual_nb; k++) {
            double my_dot = 0.0;
            if (tid == 0) my_dot = sPanel[j * IB + k];
            for (int i = j + 1 + tid; i < m; i += nthreads) my_dot += sPanel[i * IB + j] * sPanel[i * IB + k];
            double dot = block_reduce_d(my_dot, scratch, tid, nthreads);
            double scale = tau_j * dot;
            if (tid == 0) sPanel[j * IB + k] -= scale;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * IB + k] -= scale * sPanel[i * IB + j];
            __syncthreads();
        }
    }
    for (int idx = tid; idx < m * actual_nb; idx += nthreads) {
        int row = idx / actual_nb, col = idx % actual_nb;
        mat[(j_start + row) * n + (j_start + col)] = (float)sPanel[row * IB + col];
    }
    if (tid < actual_nb) tau_base[j_start + tid] = (float)sTau[tid];
}

__global__ void build_v_kernel(
    const float* __restrict__ H, float* __restrict__ V,
    const int n, const int j_start, const int actual_nb, const int m_panel, const int v_ld
) {
    const int bid = blockIdx.y;
    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < m_panel * actual_nb; linear += (size_t)gridDim.x * blockDim.x) {
        int row = linear / actual_nb, col = linear % actual_nb;
        float value = 0.0f;
        if (row == col) value = 1.0f;
        else if (row > col) value = H[(size_t)bid * n * n + (j_start + row) * n + (j_start + col)];
        V[(size_t)bid * m_panel * v_ld + row * v_ld + col] = value;
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
    const int batch = A.size(0);
    const int n = A.size(1);

    auto H = A.clone();
    auto tau = torch::zeros({batch, n}, A.options());

    if (n <= 32) {
        auto H2 = torch::empty_like(A);
        fused_qr_small<<<batch, 32, (n*n+n)*sizeof(float)>>>(
            A.data_ptr<float>(), H2.data_ptr<float>(), tau.data_ptr<float>(), n);
        return {H2, tau};
    }

    if (n <= 512 && batch >= 16) {
        int threads = 256;
        int num_warps = threads / 32;
        bool is_medium = n <= 352;
        int ob = std::min(OB, n);  // outer block
        int ib = IB;               // inner block

        // Set up SMEM attributes
        if (is_medium) {
            size_t max_smem = ((size_t)n * IB + IB + num_warps) * sizeof(float);
            if (max_smem > 48*1024) cudaFuncSetAttribute(fused_panel_fp32, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        } else {
            size_t max_smem_d = ((size_t)n * IB + IB + num_warps) * sizeof(double);
            cudaFuncSetAttribute(fused_panel_dbl, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_d);
            size_t max_smem_f = ((size_t)n * IB + IB + num_warps) * sizeof(float);
            if (max_smem_f > 48*1024) cudaFuncSetAttribute(fused_panel_fp32, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_f);
        }

        for (int j_outer = 0; j_outer < n; j_outer += ob) {
            int actual_ob = std::min(ob, n - j_outer);
            int m_outer = n - j_outer;

            // === INNER LOOP: factorize the outer block panel (actual_ob columns) ===
            for (int j_inner = 0; j_inner < actual_ob; j_inner += ib) {
                int j = j_outer + j_inner;
                int actual_ib = std::min(ib, actual_ob - j_inner);
                int j_end = j + actual_ib;
                int m_panel = n - j;

                // Panel factorization
                bool use_dbl = !is_medium && m_panel > 256;
                if (use_dbl) {
                    size_t smem = ((size_t)m_panel * IB + IB + num_warps) * sizeof(double);
                    fused_panel_dbl<<<batch, threads, smem>>>(
                        H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
                } else {
                    size_t smem = ((size_t)m_panel * IB + IB + num_warps) * sizeof(float);
                    fused_panel_fp32<<<batch, threads, smem>>>(
                        H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
                }

                // Inner trailing: update only columns within the outer block
                int inner_trailing_end = j_outer + actual_ob;
                int inner_trailing_cols = inner_trailing_end - j_end;
                if (inner_trailing_cols > 0) {
                    // Build V for inner panel
                    auto V_inner = torch::empty({batch, m_panel, actual_ib}, A.options());
                    int v_blocks = std::min<int>((m_panel * actual_ib + threads - 1) / threads, 65535);
                    build_v_kernel<<<dim3(v_blocks, batch), threads>>>(
                        H.data_ptr<float>(), V_inner.data_ptr<float>(),
                        n, j, actual_ib, m_panel, actual_ib);

                    // Build T for inner panel (ATen, small)
                    auto panel_tau = tau.slice(1, j, j_end);
                    auto T_inner = torch::zeros({batch, actual_ib, actual_ib}, A.options());
                    for (int k = 0; k < actual_ib; k++) {
                        T_inner.select(1, k).select(1, k).copy_(panel_tau.select(1, k));
                        if (k > 0) {
                            auto Vk = V_inner.slice(2, k, k+1);
                            auto Vprev = V_inner.slice(2, 0, k);
                            auto z = at::bmm(Vprev.transpose(1,2), Vk).squeeze(2);
                            auto Tprev = T_inner.slice(1, 0, k).slice(2, 0, k);
                            z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                            auto tau_k = panel_tau.select(1, k).unsqueeze(1);
                            T_inner.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
                        }
                    }

                    // Apply to inner trailing (within outer block only)
                    auto inner_trailing = H.slice(1, j, n).slice(2, j_end, inner_trailing_end);
                    auto W1 = at::matmul(V_inner.transpose(1,2), inner_trailing);
                    auto W2 = at::matmul(T_inner.transpose(1,2), W1);
                    inner_trailing.sub_(at::matmul(V_inner, W2));
                }
            }

            // === OUTER TRAILING: apply accumulated reflectors to columns beyond outer block ===
            int outer_trailing_cols = n - (j_outer + actual_ob);
            if (outer_trailing_cols <= 0) continue;

            // Build the full outer V (m_outer × actual_ob) from the factored H
            auto V_outer = torch::empty({batch, m_outer, actual_ob}, A.options());
            int v_blocks = std::min<int>((m_outer * actual_ob + threads - 1) / threads, 65535);
            build_v_kernel<<<dim3(v_blocks, batch), threads>>>(
                H.data_ptr<float>(), V_outer.data_ptr<float>(),
                n, j_outer, actual_ob, m_outer, actual_ob);

            // Build the full outer T (actual_ob × actual_ob)
            auto outer_tau = tau.slice(1, j_outer, j_outer + actual_ob);
            auto T_outer = torch::zeros({batch, actual_ob, actual_ob}, A.options());
            for (int k = 0; k < actual_ob; k++) {
                T_outer.select(1, k).select(1, k).copy_(outer_tau.select(1, k));
                if (k > 0) {
                    auto Vk = V_outer.slice(2, k, k+1);
                    auto Vprev = V_outer.slice(2, 0, k);
                    auto z = at::bmm(Vprev.transpose(1,2), Vk).squeeze(2);
                    auto Tprev = T_outer.slice(1, 0, k).slice(2, 0, k);
                    z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                    auto tau_k = outer_tau.select(1, k).unsqueeze(1);
                    T_outer.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
                }
            }

            // ONE big trailing GEMM for the full outer trailing matrix
            auto outer_trailing = H.slice(1, j_outer, n).slice(2, j_outer + actual_ob, n);
            auto W1 = at::matmul(V_outer.transpose(1,2), outer_trailing);
            auto W2 = at::matmul(T_outer.transpose(1,2), W1);
            outer_trailing.sub_(at::matmul(V_outer, W2));
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
    name="qr_v25_2level",
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
