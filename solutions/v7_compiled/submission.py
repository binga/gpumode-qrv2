"""V7: Compiled blocked Householder QR with tuned dispatch.

Key changes from V6:
1. torch.compile on the trailing update to fuse bmm chain
2. Per-shape nb tuning: nb=32 for n<=352, nb=64 for n=512
3. Blocked WY extended to n<=1024 when batch>=16
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _extract_V_fast(panel_h: torch.Tensor, nb: int) -> torch.Tensor:
    batch, m, _ = panel_h.shape
    V = panel_h[:, :, :nb].clone()
    mask = torch.triu(torch.ones(m, nb, device=V.device, dtype=torch.bool), diagonal=1)
    V[:, mask] = 0.0
    diag_idx = torch.arange(nb, device=V.device)
    V[:, diag_idx, diag_idx] = 1.0
    return V


def _trailing_update(V, T, trailing):
    """Apply block reflector: trailing -= V @ T^T @ V^T @ trailing"""
    W = torch.bmm(V.transpose(1, 2), trailing)
    W = torch.bmm(T.transpose(1, 2), W)
    return trailing - torch.bmm(V, W)


def _build_T_fast(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    batch, m, nb = V.shape
    T = torch.zeros(batch, nb, nb, device=V.device, dtype=V.dtype)
    VtV = torch.bmm(V.transpose(1, 2), V)
    for j in range(nb):
        T[:, j, j] = tau[:, j]
        if j > 0:
            z = VtV[:, :j, j]
            Tz = torch.bmm(T[:, :j, :j], z.unsqueeze(2)).squeeze(2)
            T[:, :j, j] = -tau[:, j].unsqueeze(1) * Tz
    return T


def _blocked_qr(A: torch.Tensor, nb: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n, _ = A.shape
    H = A.clone()
    tau_all = torch.zeros(batch, n, device=A.device, dtype=A.dtype)

    for j in range(0, n, nb):
        actual_nb = min(nb, n - j)
        j_end = j + actual_nb

        panel = H[:, j:, j:j_end].contiguous()
        panel_h, panel_tau = torch.geqrf(panel)

        H[:, j:, j:j_end] = panel_h
        tau_all[:, j:j_end] = panel_tau

        trailing_cols = n - j_end
        if trailing_cols <= 0:
            break

        V = _extract_V_fast(panel_h, actual_nb)
        T = _build_T_fast(V, panel_tau)

        trailing = H[:, j:, j_end:].contiguous()
        H[:, j:, j_end:] = _trailing_update(V, T, trailing)

    return H, tau_all


# Try to compile the trailing update for kernel fusion
_compiled_trailing = None

def _get_compiled_trailing():
    global _compiled_trailing
    if _compiled_trailing is None:
        try:
            _compiled_trailing = torch.compile(_trailing_update, mode="max-autotune-no-cudagraphs")
        except Exception:
            _compiled_trailing = _trailing_update
    return _compiled_trailing


def _blocked_qr_compiled(A: torch.Tensor, nb: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n, _ = A.shape
    H = A.clone()
    tau_all = torch.zeros(batch, n, device=A.device, dtype=A.dtype)

    update_fn = _get_compiled_trailing()

    for j in range(0, n, nb):
        actual_nb = min(nb, n - j)
        j_end = j + actual_nb

        panel = H[:, j:, j:j_end].contiguous()
        panel_h, panel_tau = torch.geqrf(panel)

        H[:, j:, j:j_end] = panel_h
        tau_all[:, j:j_end] = panel_tau

        trailing_cols = n - j_end
        if trailing_cols <= 0:
            break

        V = _extract_V_fast(panel_h, actual_nb)
        T = _build_T_fast(V, panel_tau)

        trailing = H[:, j:, j_end:].contiguous()
        H[:, j:, j_end:] = update_fn(V, T, trailing)

    return H, tau_all


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    if n <= 64:
        return torch.geqrf(data)

    if n <= 352 and batch >= 16:
        return _blocked_qr_compiled(data, nb=32)

    if n <= 512 and batch >= 16:
        return _blocked_qr_compiled(data, nb=64)

    return torch.geqrf(data)
