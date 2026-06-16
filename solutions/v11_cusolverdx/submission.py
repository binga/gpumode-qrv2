"""V11 probe: Test cuSOLVERDx availability on Popcorn B200.

If cuSOLVERDx headers are available, use device-side batched QR.
Falls back to torch.geqrf if compilation fails.
"""
import torch
from task import input_t, output_t

_kernel = None
_probe_failed = False


def _build_kernel():
    global _kernel, _probe_failed
    if _kernel is not None:
        return _kernel
    if _probe_failed:
        return None

    from torch.utils.cpp_extension import load_inline

    # Minimal probe: just try to include the cuSOLVERDx header
    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Try to include cuSOLVERDx
#if __has_include(<cusolverdx.hpp>)
#include <cusolverdx.hpp>
#define HAS_CUSOLVERDX 1
#else
#define HAS_CUSOLVERDX 0
#endif

#include <vector>

std::vector<torch::Tensor> probe_cusolverdx(torch::Tensor A) {
    // Just return whether cuSOLVERDx is available
    auto result = torch::zeros({1}, A.options());
    result[0] = HAS_CUSOLVERDX;
    return {result};
}
"""

    cpp_src = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> probe_cusolverdx(torch::Tensor A);
"""

    try:
        _kernel = load_inline(
            name="qr_v11_cusolverdx_probe",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["probe_cusolverdx"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=True,
        )
    except Exception as e:
        print(f"cuSOLVERDx probe compile failed: {e}")
        _probe_failed = True
        _kernel = None
    return _kernel


def custom_kernel(data: input_t) -> output_t:
    kernel = _build_kernel()
    if kernel is not None:
        result = kernel.probe_cusolverdx(data)
        has_cusolverdx = result[0].item()
        print(f"cuSOLVERDx available: {bool(has_cusolverdx)}")

    # Always fall back to torch.geqrf for correctness
    return torch.geqrf(data)
