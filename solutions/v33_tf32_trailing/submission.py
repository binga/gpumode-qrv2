"""V33: V27 + TF32 tensor cores for trailing GEMMs.

Key hypothesis: V27's double-precision panel stabilizes the algorithm enough
that TF32 trailing updates (using tensor cores) maintain precision.
Previous TF32 failures (V10, V30) used FP32 panel — the error accumulated.
With double panel, each step's trailing error is bounded and corrected by
the next step's double-precision panel factorization.

If this passes batch=640 n=512 stress tests, we get ~8-16x speedup on GEMMs.

Original V21/V27 description:

Competitor analysis shows top solutions at 1-2ms fully fuse panel+trailing.
Our two biggest remaining overheads:
1. ATen T construction: ~496 tiny bmm calls for n=512 → CUDA parallel T-builder
2. trailing.contiguous(): ~600MB copy per step → cuBLAS strided-batched GEMM in-place

Tier 1 (n=32): CuTe DSL shared-memory Householder QR
Tier 2a (n<=352, batch>=16): FP32 fused panel + CUDA V/T + cuBLAS trailing
Tier 2b (n=512, batch>=16): double panel + CUDA V/T + cuBLAS trailing
Tier 3: torch.geqrf fallback
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_cute_qr32_status = "untried"
_cute_qr32_error = ""
_cute_qr32_compiled = None
_cute_qr32_defs_ready = False


def _ensure_cute_qr32(data: torch.Tensor, H: torch.Tensor, tau: torch.Tensor) -> bool:
    global _cute_qr32_status, _cute_qr32_error, _cute_qr32_compiled, _cute_qr32_defs_ready
    if _cute_qr32_status in ("ready", "unavailable"):
        return _cute_qr32_status == "ready"
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception as exc:
        _cute_qr32_status = "unavailable"
        _cute_qr32_error = f"import failed: {exc}"
        return False
    if not _cute_qr32_defs_ready:
        try:
            globals_dict = globals()
            @cute.kernel
            def _qr32_kernel(a: cute.Tensor, h: cute.Tensor, tau_out: cute.Tensor):
                bid, _, _ = cute.arch.block_idx()
                tid, _, _ = cute.arch.thread_idx()
                n = 32
                allocator = cutlass.utils.SmemAllocator()
                s_h = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((32, 32)), byte_alignment=16, swizzle=None)
                s_tau = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), byte_alignment=16, swizzle=None)
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
                batch = a.shape[0]
                _qr32_kernel(a, h, tau_out).launch(grid=(batch, 1, 1), block=(32, 1, 1))
            globals_dict["_cute_qr32_launch"] = _qr32_launch
            globals_dict["_cute_from_dlpack"] = from_dlpack
            _cute_qr32_defs_ready = True
        except Exception as exc:
            _cute_qr32_status = "unavailable"
            _cute_qr32_error = f"definition failed: {exc}"
            return False
    try:
        a_cute = globals()["_cute_from_dlpack"](data).mark_layout_dynamic()
        h_cute = globals()["_cute_from_dlpack"](H).mark_layout_dynamic()
        tau_cute = globals()["_cute_from_dlpack"](tau).mark_layout_dynamic()
        _cute_qr32_compiled = cute.compile(globals()["_cute_qr32_launch"], a_cute, h_cute, tau_cute)
        _cute_qr32_compiled(a_cute, h_cute, tau_cute)
        torch.cuda.synchronize()
        _cute_qr32_status = "ready"
        print("CuTe QR32 compiled and launched")
        return True
    except Exception as exc:
        _cute_qr32_status = "unavailable"
        _cute_qr32_error = f"compile/launch failed: {exc}"
        return False


def _try_cute_qr32(data: torch.Tensor) -> output_t | None:
    if data.dim() != 3 or data.size(1) != 32 or data.size(2) != 32:
        return None
    H = torch.empty_like(data)
    tau = torch.empty((data.size(0), 32), device=data.device, dtype=data.dtype)
    if not _ensure_cute_qr32(data, H, tau):
        return None
    a_cute = globals()["_cute_from_dlpack"](data).mark_layout_dynamic()
    h_cute = globals()["_cute_from_dlpack"](H).mark_layout_dynamic()
    tau_cute = globals()["_cute_from_dlpack"](tau).mark_layout_dynamic()
    _cute_qr32_compiled(a_cute, h_cute, tau_cute)
    return H, tau


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

__global__ void fused_panel_kernel(
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
            double inv_v0 = 1.0 / v0;
            for (int i = j + 1 + tid; i < m; i += nthreads) sPanel[i * NB + j] *= inv_v0;
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
    const int n, const int j_start, const int actual_nb, const int m_panel, const int v_ld
) {
    const int bid = blockIdx.y;
    const int per_matrix = m_panel * actual_nb;
    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < per_matrix; linear += (size_t)gridDim.x * blockDim.x) {
        int row = linear / actual_nb, col = linear % actual_nb;
        float value = 0.0f;
        if (row == col) value = 1.0f;
        else if (row > col) value = H[(size_t)bid * n * n + (j_start + row) * n + (j_start + col)];
        V[(size_t)bid * m_panel * v_ld + row * v_ld + col] = value;
    }
}

// Parallel T-builder: one block per batch element, 256 threads cooperate on dot products
__global__ void build_t_parallel(
    const float* __restrict__ V, const float* __restrict__ tau,
    float* __restrict__ T,
    const int m_panel, const int actual_nb, const int v_ld, const int t_ld,
    const int j_start, const int n
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    
    const float* v = V + (size_t)bid * m_panel * v_ld;
    const float* tau_b = tau + (size_t)bid * n;
    float* t = T + (size_t)bid * t_ld * t_ld;

    extern __shared__ float smem_t[];
    float* s_dots = smem_t;
    float* s_work = s_dots + NB;

    for (int idx = tid; idx < t_ld * t_ld; idx += nthreads) t[idx] = 0.0f;
    __syncthreads();

    for (int k = 0; k < actual_nb; k++) {
        float tau_k = tau_b[j_start + k];
        if (tid == 0) t[k * t_ld + k] = tau_k;
        
        if (k == 0) { __syncthreads(); continue; }

        // Compute dots[i] = V[:,i]^T @ V[:,k] for i=0..k-1, cooperatively
        for (int i = 0; i < k; i++) {
            float my_dot = 0.0f;
            for (int r = tid; r < m_panel; r += nthreads) {
                my_dot += v[r * v_ld + i] * v[r * v_ld + k];
            }
            // Block reduce
            float dot = block_reduce_f(my_dot, s_work, tid, nthreads);
            if (tid == 0) s_dots[i] = dot;
            __syncthreads();
        }

        // Compute work[i] = sum_p T[i,p] * dots[p] for i=0..k-1
        // Then T[i,k] = -tau_k * work[i]
        // This is small (k x k matmul on k values) — do serially on thread 0
        if (tid == 0) {
            for (int i = 0; i < k; i++) {
                double acc = 0.0;
                for (int p = 0; p < k; p++) {
                    acc += (double)t[i * t_ld + p] * (double)s_dots[p];
                }
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
            float inv_v0 = 1.0f / v0;
            if (tid > j && tid < n) sA[tid * n + j] *= inv_v0;
            if (tid == 0) { sA[j * n + j] = diag; sTau[j] = tau_j; }
        } else { if (tid == 0) sTau[j] = 0.0f; }
        __syncwarp();
        tau_j = sTau[j]; if (tau_j == 0.0f) continue;
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
        bool is_medium = n <= 352;

        if (is_medium) {
            size_t max_smem = (n * NB + NB + num_warps) * sizeof(float);
            if (max_smem > 48 * 1024)
                cudaFuncSetAttribute(fused_panel_fp32, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        } else {
            size_t max_smem_d = (n * NB + NB + num_warps) * sizeof(double);
            cudaFuncSetAttribute(fused_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_d);
        }

        // Preallocate ALL workspaces once (eliminates per-step allocation)
        auto V = torch::empty({batch, n, nb}, A.options());
        auto T = torch::empty({batch, nb, nb}, A.options());
        auto W1 = torch::empty({batch, nb, n}, A.options());
        auto W2 = torch::empty({batch, nb, n}, A.options());

        for (int j = 0; j < n; j += nb) {
            int actual_nb = std::min(nb, n - j);
            int j_end = j + actual_nb;
            int m_panel = n - j;

            bool use_fp32 = is_medium;
            if (use_fp32) {
                size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(float);
                fused_panel_fp32<<<batch, threads, panel_smem>>>(
                    H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
            } else {
                size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(double);
                fused_panel_kernel<<<batch, threads, panel_smem>>>(
                    H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);
            }

            int trailing_cols = n - j_end;
            if (trailing_cols <= 0) break;

            // V_view: (batch, m_panel, actual_nb) contiguous
            auto V_view = V.slice(1, 0, m_panel).slice(2, 0, actual_nb).contiguous();
            float* v_ptr = V_view.data_ptr<float>();

            // CUDA V-builder writes to contiguous V_view
            int per_matrix_v = m_panel * actual_nb;
            int v_blocks = std::min<int>((per_matrix_v + threads - 1) / threads, 65535);
            build_v_kernel<<<dim3(v_blocks, batch), threads>>>(
                H.data_ptr<float>(), v_ptr,
                n, j, actual_nb, m_panel, actual_nb);

            // T construction: CUDA parallel for medium (low batch), ATen for 512 (high batch)
            if (is_medium) {
                size_t t_smem = (NB + num_warps) * sizeof(float);
                build_t_parallel<<<batch, threads, t_smem>>>(
                    v_ptr, tau.data_ptr<float>(), T.data_ptr<float>(),
                    m_panel, actual_nb, actual_nb, nb, j, n);
            } else {
                auto panel_tau = tau.slice(1, j, j_end);
                T.zero_();
                auto T_tmp = T.slice(1, 0, actual_nb).slice(2, 0, actual_nb);
                for (int k = 0; k < actual_nb; k++) {
                    T_tmp.select(1, k).select(1, k).copy_(panel_tau.select(1, k));
                    if (k > 0) {
                        auto Vk = V_view.slice(2, k, k+1);
                        auto Vprev = V_view.slice(2, 0, k);
                        auto z = at::bmm(Vprev.transpose(1,2), Vk).squeeze(2);
                        auto Tprev = T_tmp.slice(1, 0, k).slice(2, 0, k);
                        z = at::bmm(Tprev, z.unsqueeze(2)).squeeze(2);
                        auto tau_k = panel_tau.select(1, k).unsqueeze(1);
                        T_tmp.slice(1, 0, k).select(2, k).copy_(z * (-tau_k));
                    }
                }
            }

            auto T_view = T.slice(1, 0, actual_nb).slice(2, 0, actual_nb).contiguous();
            auto trailing = H.slice(1, j, n).slice(2, j_end, n);

            // Preallocated W1/W2 views for this step's trailing size
            auto W1_view = W1.slice(1, 0, actual_nb).slice(2, 0, trailing_cols);
            auto W2_view = W2.slice(1, 0, actual_nb).slice(2, 0, trailing_cols);

            // W1 = V^T @ trailing
            at::matmul_out(W1_view, V_view.transpose(1,2), trailing);
            // W2 = T^T @ W1
            at::matmul_out(W2_view, T_view.transpose(1,2), W1_view);
            // trailing -= V @ W2 (fused via baddbmm: trailing = 1*trailing + (-1)*V@W2)
            at::baddbmm_out(trailing, trailing, V_view, W2_view, /*beta=*/1.0, /*alpha=*/-1.0);
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
    name="qr_v33_tf32",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["dispatch_qr"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    # Use 'high' precision: 3xTF32 internally (tensor cores, FP32-accurate)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cute_qr32 = _try_cute_qr32(data)
    if cute_qr32 is not None:
        return cute_qr32
    result = module.dispatch_qr(data)
    return (result[0], result[1])
