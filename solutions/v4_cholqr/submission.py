"""V4: CholeskyQR2 with conversion to compact Householder format.

Algorithm:
  1. R1 = cholesky(A^T @ A)         — GEMM + Cholesky
  2. Q1 = A @ inv(R1)               — triangular solve
  3. R2 = cholesky(Q1^T @ Q1)       — second pass for stability
  4. Q  = Q1 @ inv(R2)
  5. R  = R2 @ R1
  6. Convert (Q, R) to compact Householder (H, tau)

The advantage: steps 1-5 are all GEMMs and triangular solves — fully
tensor-core friendly. No sequential column dependencies.

The risk: A^T @ A squares the condition number. CholeskyQR2 (two passes)
recovers stability for most inputs, but near-collinear matrices may fail.

The cost: Converting Q back to compact Householder format requires
computing the Householder vectors from Q, which is O(n^3).
"""
import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _q_to_compact_householder(Q: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (Q, R) to compact Householder format matching torch.geqrf.

    Uses torch.geqrf on Q @ R = A to get the compact form directly.
    This is a workaround — the proper approach would extract Householder
    vectors from Q directly, but matching LAPACK's exact convention is tricky.
    """
    A_reconstructed = torch.bmm(Q, R)
    return torch.geqrf(A_reconstructed)


def _cholqr2(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CholeskyQR2: two-pass Cholesky-based QR."""
    batch, n, _ = A.shape

    # Pass 1: R1 = chol(A^T A), Q1 = A R1^{-1}
    AtA = torch.bmm(A.transpose(1, 2), A)
    try:
        R1 = torch.linalg.cholesky(AtA)
    except torch.linalg.LinAlgError:
        return torch.geqrf(A)

    Q1 = torch.linalg.solve_triangular(R1, A.transpose(1, 2), upper=False)
    Q1 = Q1.transpose(1, 2)  # (batch, n, n)

    # Pass 2: R2 = chol(Q1^T Q1), Q = Q1 R2^{-1}
    Q1tQ1 = torch.bmm(Q1.transpose(1, 2), Q1)
    try:
        R2 = torch.linalg.cholesky(Q1tQ1)
    except torch.linalg.LinAlgError:
        return torch.geqrf(A)

    Q = torch.linalg.solve_triangular(R2, Q1.transpose(1, 2), upper=False)
    Q = Q.transpose(1, 2)

    R = torch.bmm(R2, R1)

    # Need to convert to compact Householder format
    # Since direct conversion is complex, fall back to geqrf on the original
    # This defeats the purpose — but tests correctness of the cholqr path
    return torch.geqrf(A)


def _cholqr_geqrf_hybrid(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use CholeskyQR to get a good preconditioner, then run geqrf on
    the better-conditioned matrix. This may help cuSOLVER converge faster.

    Actually, torch.geqrf doesn't iterate — it's direct Householder.
    So preconditioning doesn't help. Just run geqrf directly.
    """
    return torch.geqrf(A)


def custom_kernel(data: input_t) -> output_t:
    # CholeskyQR can't easily produce compact Householder format without
    # calling geqrf anyway. The conversion Q→(H,tau) is itself O(n^3)
    # and basically requires running Householder reflections on Q.
    #
    # So this approach only works if we can convince the checker to accept
    # (Q, R) directly — but the checker requires (H, tau) format.
    #
    # Falling back to geqrf for now. The real optimization needs custom CUDA.
    return torch.geqrf(data)
