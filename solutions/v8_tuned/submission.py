"""V8: Tuned blocked Householder QR — best per-shape nb from V5/V6 profiling.

nb=32 for n<=352 (V5 was 27ms, V6 nb=64 regressed to 34ms)
nb=64 for n=512 (V6 was 598ms, V5 nb=32 was 616ms)
geqrf for n>=1024 (blocked approach is slower at low batch)
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _extract_V(panel_h: torch.Tensor, nb: int) -> torch.Tensor:
    batch, m, _ = panel_h.shape
    V = panel_h[:, :, :nb].clone()
    mask = torch.triu(torch.ones(m, nb, device=V.device, dtype=torch.bool), diagonal=1)
    V[:, mask] = 0.0
    diag_idx = torch.arange(nb, device=V.device)
    V[:, diag_idx, diag_idx] = 1.0
    return V


def _build_T(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    batch, m, nb = V.shape
    T = torch.zeros(batch, nb, nb, device=V.device, dtype=V.dtype)
    VtV = torch.bmm(V.transpose(1, 2), V)
    for j in range(nb):
        T[:, j, j] = tau[:, j]
        if j > 0:
            Tz = torch.bmm(T[:, :j, :j], VtV[:, :j, j:j+1]).squeeze(2)
            T[:, :j, j] = -tau[:, j].unsqueeze(1) * Tz
    return T


def _blocked_qr(A: torch.Tensor, nb: int) -> tuple[torch.Tensor, torch.Tensor]:
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

        V = _extract_V(panel_h, actual_nb)
        T = _build_T(V, panel_tau)

        trailing = H[:, j:, j_end:].contiguous()
        W = torch.bmm(V.transpose(1, 2), trailing)
        W = torch.bmm(T.transpose(1, 2), W)
        H[:, j:, j_end:] = trailing - torch.bmm(V, W)

    return H, tau_all


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    if n <= 64:
        return torch.geqrf(data)

    if batch >= 16 and n <= 352:
        return _blocked_qr(data, nb=32)

    if batch >= 16 and n <= 512:
        return _blocked_qr(data, nb=64)

    return torch.geqrf(data)
