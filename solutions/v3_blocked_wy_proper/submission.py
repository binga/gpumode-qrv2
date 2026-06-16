"""V3: Blocked Householder QR with proper WY trailing update.

Key difference from V1: uses WY compact form (V, T matrices) and two
batched GEMMs for the trailing update instead of reconstructing Q via
householder_product. This avoids the expensive Q reconstruction.

Panel factorization still uses torch.geqrf (cuSOLVER/MAGMA).
Trailing update uses torch.bmm which dispatches to cuBLAS with TF32.
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _extract_V(panel_h: torch.Tensor, nb: int) -> torch.Tensor:
    """Extract V (unit lower trapezoidal) from geqrf output."""
    batch, m, _ = panel_h.shape
    V = torch.zeros(batch, m, nb, device=panel_h.device, dtype=panel_h.dtype)
    for k in range(nb):
        V[:, k, k] = 1.0
        if k + 1 < m:
            V[:, k+1:, k] = panel_h[:, k+1:, k]
    return V


def _build_T(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Build T factor for WY representation: block reflector = I - V T V^T.

    T is nb x nb upper triangular, built column by column.
    """
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
    n = data.shape[-1]
    if n <= 64:
        return torch.geqrf(data)
    return _blocked_qr_wy(data, nb=32)
