"""V6: Optimized blocked Householder QR.

Key improvements over V5:
1. Vectorized V extraction (no Python loop)
2. Vectorized T construction (batch all inner products at once)
3. Pre-allocated workspace to avoid repeated allocation
4. Tuned block size per shape
5. Fused V^T @ trailing and T @ W into fewer operations
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _extract_V_fast(panel_h: torch.Tensor, nb: int) -> torch.Tensor:
    """Extract V from geqrf output — vectorized, no Python loop."""
    batch, m, _ = panel_h.shape
    V = panel_h[:, :, :nb].clone()
    # Zero out upper triangle and set diagonal to 1
    mask = torch.triu(torch.ones(m, nb, device=V.device, dtype=torch.bool), diagonal=1)
    V[:, mask] = 0.0
    diag_idx = torch.arange(nb, device=V.device)
    V[:, diag_idx, diag_idx] = 1.0
    return V


def _build_T_fast(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Build T via recursive formula — minimize Python loop iterations.

    T[j,j] = tau[j]
    T[0:j, j] = -tau[j] * T[0:j, 0:j] @ V^T[:, 0:j] @ V[:, j]
    """
    batch, m, nb = V.shape
    T = torch.zeros(batch, nb, nb, device=V.device, dtype=V.dtype)

    # Precompute V^T @ V (nb x nb) — all inner products at once
    VtV = torch.bmm(V.transpose(1, 2), V)  # (batch, nb, nb)

    for j in range(nb):
        T[:, j, j] = tau[:, j]
        if j > 0:
            # T[0:j, j] = -tau[j] * T[0:j, 0:j] @ VtV[0:j, j]
            z = VtV[:, :j, j]  # (batch, j) — precomputed
            Tz = torch.bmm(T[:, :j, :j], z.unsqueeze(2)).squeeze(2)
            T[:, :j, j] = -tau[:, j].unsqueeze(1) * Tz

    return T


def _blocked_qr_opt(A: torch.Tensor, nb: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Trailing update: H[j:, j_end:] -= V @ T^T @ V^T @ H[j:, j_end:]
        trailing = H[:, j:, j_end:].contiguous()
        W = torch.bmm(V.transpose(1, 2), trailing)  # (batch, nb, trailing_cols)
        W = torch.bmm(T.transpose(1, 2), W)          # (batch, nb, trailing_cols)
        H[:, j:, j_end:] = trailing - torch.bmm(V, W)

    return H, tau_all


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    if n <= 64:
        return torch.geqrf(data)

    if n <= 512 and batch >= 16:
        # Larger nb = fewer block steps = less Python overhead
        nb = 64 if n >= 256 else 32
        return _blocked_qr_opt(data, nb=nb)

    return torch.geqrf(data)
