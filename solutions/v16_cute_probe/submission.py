"""V16: CuTe DSL probe plus a first one-shape QR kernel.

This version does two things:
  - probes CuTe DSL on n=512 cases so we know the B200 runner supports it
  - attempts a real CuTe DSL Householder QR path for n=32

Both CuTe paths are guarded by fallbacks to `torch.geqrf` so correctness is not
coupled to DSL compile success while we iterate.
"""
import torch
from task import input_t, output_t


_cute_status = "untried"
_cute_error = ""
_cute_compiled = None
_cute_probe_defs_ready = False
_cute_qr32_status = "untried"
_cute_qr32_error = ""
_cute_qr32_compiled = None
_cute_qr32_defs_ready = False


def _ensure_cute_probe():
    """Compile a tiny CuTe kernel once if the runtime is available."""
    global _cute_status, _cute_error, _cute_compiled, _cute_probe_defs_ready

    if _cute_status in ("ready", "unavailable"):
        return _cute_status == "ready"

    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception as exc:
        _cute_status = "unavailable"
        _cute_error = f"import failed: {exc}"
        print(f"CuTe DSL unavailable: {_cute_error}")
        return False

    if not _cute_probe_defs_ready:
        try:
            globals_dict = globals()

            @cute.kernel
            def _probe_kernel(a: cute.Tensor, out: cute.Tensor):
                tid, _, _ = cute.arch.thread_idx()
                if tid == 0:
                    out[0] = a[0]

            @cute.jit
            def _probe_launch(a: cute.Tensor, out: cute.Tensor):
                _probe_kernel(a, out).launch(grid=(1, 1, 1), block=(32, 1, 1))

            globals_dict["_cute_probe_launch"] = _probe_launch
            globals_dict["_cute_from_dlpack"] = from_dlpack
            _cute_probe_defs_ready = True
        except Exception as exc:
            _cute_status = "unavailable"
            _cute_error = f"definition failed: {exc}"
            print(f"CuTe DSL unavailable: {_cute_error}")
            return False

    try:
        sample = torch.empty(1, device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        sample_cute = globals()["_cute_from_dlpack"](sample).mark_layout_dynamic()
        out_cute = globals()["_cute_from_dlpack"](out).mark_layout_dynamic()
        _cute_compiled = cute.compile(globals()["_cute_probe_launch"], sample_cute, out_cute)
        _cute_compiled(sample_cute, out_cute)
        torch.cuda.synchronize()
        _cute_status = "ready"
        print("CuTe DSL probe compiled and launched")
        return True
    except Exception as exc:
        _cute_status = "unavailable"
        _cute_error = f"compile/launch failed: {exc}"
        print(f"CuTe DSL unavailable: {_cute_error}")
        return False


def _run_cute_probe_for_shape(data: torch.Tensor) -> None:
    """Exercise CuTe only on the shape we would optimize next."""
    if data.dim() == 3 and data.size(1) == 512:
        if _ensure_cute_probe():
            flat = data.reshape(-1)
            out = torch.empty(1, device=data.device, dtype=data.dtype)
            flat_cute = globals()["_cute_from_dlpack"](flat).mark_layout_dynamic()
            out_cute = globals()["_cute_from_dlpack"](out).mark_layout_dynamic()
            _cute_compiled(flat_cute, out_cute)


def _ensure_cute_qr32(data: torch.Tensor, H: torch.Tensor, tau: torch.Tensor) -> bool:
    """Compile the n=32 CuTe QR kernel once."""
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


def custom_kernel(data: input_t) -> output_t:
    cute_qr32 = _try_cute_qr32(data)
    if cute_qr32 is not None:
        return cute_qr32
    _run_cute_probe_for_shape(data)
    return torch.geqrf(data)
