"""V17: V13 best path with CuTe DSL QR32.

Tier 1 (n=32): CuTe DSL shared-memory Householder QR, one warp per matrix.
Tier 2 (n<=512, batch>=16): V13 fused double-accumulation panel + ATen trailing.
Tier 3: torch.geqrf fallback.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_cute_qr32_status = "untried"
_cute_qr32_error = ""
_cute_qr32_compiled = None
_cute_qr32_defs_ready = False


def _ensure_cute_qr32(data: torch.Tensor, H: torch.Tensor, tau: torch.Tensor) -> bool:
    """Compile the n=32 CuTe QR kernel once, falling back to CUDA if unavailable."""
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
        print(f"CuTe QR32 unavailable: {_cute_qr32_error}")
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
                h_layout = cute.make_layout((32, 32))
                tau_layout = cute.make_layout((32,))
                s_h = allocator.allocate_tensor(
                    cutlass.Float32, h_layout, byte_alignment=16, swizzle=None
                )
                s_tau = allocator.allocate_tensor(
                    cutlass.Float32, tau_layout, byte_alignment=16, swizzle=None
                )

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
            print(f"CuTe QR32 unavailable: {_cute_qr32_error}")
            return False

    try:
        a_cute = globals()["_cute_from_dlpack"](data).mark_layout_dynamic()
        h_cute = globals()["_cute_from_dlpack"](H).mark_layout_dynamic()
        tau_cute = globals()["_cute_from_dlpack"](tau).mark_layout_dynamic()
        _cute_qr32_compiled = cute.compile(
            globals()["_cute_qr32_launch"], a_cute, h_cute, tau_cute
        )
        _cute_qr32_compiled(a_cute, h_cute, tau_cute)
        torch.cuda.synchronize()
        _cute_qr32_status = "ready"
        print("CuTe QR32 compiled and launched")
        return True
    except Exception as exc:
        _cute_qr32_status = "unavailable"
        _cute_qr32_error = f"compile/launch failed: {exc}"
        print(f"CuTe QR32 unavailable: {_cute_qr32_error}")
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

        size_t max_smem = (n * NB + NB + num_warps) * sizeof(double);
        cudaFuncSetAttribute(fused_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);

        for (int j = 0; j < n; j += nb) {
            int actual_nb = std::min(nb, n - j);
            int j_end = j + actual_nb;
            int m_panel = n - j;

            size_t panel_smem = (m_panel * NB + NB + num_warps) * sizeof(double);
            fused_panel_kernel<<<batch, threads, panel_smem>>>(
                H.data_ptr<float>(), tau.data_ptr<float>(), n, j, j_end);

            int trailing_cols = n - j_end;
            if (trailing_cols <= 0) break;

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
    name="qr_v17_cute_qr32_v13",
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
