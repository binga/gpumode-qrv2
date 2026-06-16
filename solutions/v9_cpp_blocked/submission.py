"""V9: Fully C++ blocked Householder QR — zero Python in hot path.

The entire block loop, panel factorization (via ATen's geqrf which calls
cuSOLVER), V/T construction, and trailing update (via ATen's bmm which calls
cuBLAS) all execute in compiled C++. Python only calls the entry function once.

The key advantage over V5: no Python interpreter overhead between block steps.
V5's block loop dispatches ~6 Python-level tensor operations per iteration;
V9 does the same operations but from C++ with zero interpreter round-trips.

Size dispatch: n<=512 & batch>=16 → blocked QR; else → torch.geqrf fallback
"""
import torch
from task import input_t, output_t

_kernel = None


def _build_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel

    from torch.utils.cpp_extension import load_inline

    cpp_src = r"""
#include <torch/extension.h>
#include <vector>
#include <algorithm>

std::vector<torch::Tensor> batched_qr(torch::Tensor A, int nb) {
    TORCH_CHECK(A.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(A.size(1) == A.size(2), "Expected square matrices");
    TORCH_CHECK(A.is_cuda(), "Expected CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Expected float32");

    const int batch = A.size(0);
    const int n = A.size(1);

    auto H = A.clone();
    auto tau_all = torch::zeros({batch, n}, A.options());

    for (int j = 0; j < n; j += nb) {
        int actual_nb = std::min(nb, n - j);
        int j_end = j + actual_nb;
        int m = n - j;

        // --- Panel factorization via ATen (calls cuSOLVER internally) ---
        auto panel = H.slice(1, j, n).slice(2, j, j_end).contiguous();
        auto geqrf_result = at::geqrf(panel);
        auto panel_h = std::get<0>(geqrf_result);
        auto panel_tau = std::get<1>(geqrf_result);

        // Write factored panel back
        H.slice(1, j, n).slice(2, j, j_end).copy_(panel_h);
        tau_all.slice(1, j, j_end).copy_(panel_tau);

        int trailing_cols = n - j_end;
        if (trailing_cols <= 0) break;

        // --- Extract V (unit lower triangular) from panel_h ---
        auto V = torch::zeros({batch, m, actual_nb}, A.options());
        for (int k = 0; k < actual_nb; k++) {
            V.select(1, k).select(1, k).fill_(1.0f);
            if (k + 1 < m) {
                V.slice(1, k + 1, m).select(2, k).copy_(
                    panel_h.slice(1, k + 1, m).select(2, k)
                );
            }
        }

        // --- Build T matrix (upper triangular, nb x nb) ---
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

        // --- Trailing update: A[j:, j_end:] -= V @ T^T @ V^T @ A[j:, j_end:] ---
        auto trailing = H.slice(1, j, n).slice(2, j_end, n).contiguous();
        auto W1 = at::bmm(V.transpose(1, 2), trailing);
        auto W2 = at::bmm(T.transpose(1, 2), W1);
        auto update = at::bmm(V, W2);
        H.slice(1, j, n).slice(2, j_end, n).sub_(update);
    }

    return {H, tau_all};
}
"""

    try:
        _kernel = load_inline(
            name="qr_v9_cpp_blocked",
            cpp_sources=[cpp_src],
            cuda_sources=[],
            functions=["batched_qr"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            verbose=False,
        )
    except Exception as e:
        print(f"C++ compile failed: {e}")
        import traceback
        traceback.print_exc()
        _kernel = None
    return _kernel


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    if n <= 512 and batch >= 16 and data.is_cuda:
        kernel = _build_kernel()
        if kernel is not None:
            nb = 32
            result = kernel.batched_qr(data, nb)
            return (result[0], result[1])

    return torch.geqrf(data)
