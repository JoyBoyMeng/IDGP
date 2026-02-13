import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxonRouterCos(nn.Module):
    """
    Single-selection routing (argmax / argmin):

      - metric='cosine'    : select by maximum cosine similarity
      - metric='euclidean' : select by minimum (squared) Euclidean distance

    Other characteristics:
      - No projection, temperature, softmax, or entmax
      - Weight is implicitly 1 ([B,1]); tau is fixed to 1.0 for compatibility with old interfaces
      - Requires the last dimension of h and A to be the same

    Args:
        metric: 'cosine' | 'euclidean'
        eps:    small constant for numerical stability (normalization & negative-zero clipping)
    """
    def __init__(self, metric: str = "cosine", eps: float = 1e-8):
        super().__init__()
        metric = metric.lower()
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric}")
        self.metric = metric
        self.eps = float(eps)

    @torch.no_grad()
    def _check_shapes(self, h: torch.Tensor, A: torch.Tensor):
        if h.dim() != 2 or A.dim() != 2:
            raise ValueError(f"h and A must be 2D, got {h.shape=} {A.shape=}")
        if h.size(-1) != A.size(-1):
            raise ValueError(
                f"Dimension mismatch: h D={h.size(-1)} != A D={A.size(-1)}; "
                f"this implementation requires their last dimensions to match."
            )

    def _cosine_scores(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Normalize and compute cosine similarity, range approximately [-1, 1]
        h_n = F.normalize(h, p=2, dim=-1, eps=self.eps)   # [B, D]
        A_n = F.normalize(A, p=2, dim=-1, eps=self.eps)   # [K, D]
        return h_n @ A_n.t()                              # [B, K]

    def _euclidean_sq_dists(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Use squared Euclidean distance to avoid sqrt (same ordering as Euclidean distance)
        # dist2 = ||h||^2 + ||A||^2 - 2 h A^T
        h2 = (h * h).sum(dim=-1, keepdim=True)            # [B, 1]
        A2 = (A * A).sum(dim=-1, keepdim=True).t()        # [1, K]
        dist2 = h2 + A2 - 2.0 * (h @ A.t())               # [B, K]
        return dist2.clamp_min(0.0)                       # numerical stability: remove tiny negatives

    def forward(self, h: torch.Tensor, A: torch.Tensor):
        """
        h: [B, D]   memory representations
        A: [K, D]   fluxon state matrix

        Returns:
            idx: [B, 1] selected fluxon indices
        """
        self._check_shapes(h, A)

        if self.metric == "cosine":
            scores = self._cosine_scores(h, A)            # [B, K], larger is better
            idx = scores.argmax(dim=1, keepdim=True)      # [B, 1]
        else:  # 'euclidean'
            dist2 = self._euclidean_sq_dists(h, A)        # [B, K], smaller is better
            idx = dist2.argmin(dim=1, keepdim=True)       # [B, 1]

        return idx