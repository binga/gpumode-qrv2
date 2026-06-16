"""V5: Size-adaptive dispatch.

Uses blocked WY for high-batch medium-n shapes (where the trailing GEMM
amortizes the V/T construction overhead), and falls back to torch.geqrf
for low-batch large-n shapes (where per-block Python overhead dominates).

Based on V3 B200 profiling:
  - Blocked WY wins at n<=512 (high batch saturates SMs)
  - torch.geqrf wins at n>=1024 (low batch, Python overhead dominates)
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _extract_V(panel_h: torch.Tensor, nb: int) -> torch.Tensor:
    batch, m, _ = panel_h.shape
    V = torch.zeros(batch, m, nb, device=panel_h.device, dtype=panel_h.dtype)
    for k in range(nb):
        V[:, k, k] = 1.0
        if k + 1 < m:
            V[:, k+1:, k] = panel_h[:, k+1:, k]
    return V


def _build_T(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    batch, m, nb = V.shape
    T = torch.zeros(batch, nb, nb, device=V.device, dtype=V.dtype)
    for j in range(nb):
        T[:, j, j] = tau[:, j]
        if j > 0:
            z = torch.bmm(V[:, :, :j].transpose(1, 2), V[:, :, j:j+1]).squeeze(2)
            Tz = torch.bmm(T[:, :j, :j], z.unsqueeze(2)).squeeze(2)
            T[:, :j, j] = -tau[:, j].unsqueeze(1) * Tz
    return T


def _blocked_qr_wy(A: torch.Tensor, nb: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
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

    # Blocked WY wins when batch is large enough to amortize Python overhead
    # Threshold: batch * n^2 > ~10M elements AND n <= 512
    if n <= 512 and batch >= 16:
        return _blocked_qr_wy(data, nb=32)

    # For large n with small batch, torch.geqrf (cuSOLVER/MAGMA) is faster
    return torch.geqrf(data)
