import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Entmax 1.5 (α=1.5) implementation: numerically stable, suitable for sparse gating ----------
def _entmax_threshold_and_support(probs, alpha: float = 1.5, dim: int = -1, n_iter: int = 50, tol: float = 1e-6):
    assert alpha > 1.0
    x = probs
    # Shift for numerical stability: max to 0
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val  # ensure max(x) = 0

    inv = 1.0 / (alpha - 1.0)

    # --- Key: construct a bracketing interval [left, right] such that f(left) > 0, f(right) < 0 ---
    right = torch.zeros_like(max_val)  # = 0
    # Initial left endpoint is negative
    left = -torch.ones_like(max_val)   # = -1

    def f(tau):
        p = torch.clamp(x - tau, min=0) ** inv
        return p.sum(dim=dim, keepdim=True) - 1.0

    # Adaptively expand the left endpoint until f(left) > 0
    val_left = f(left)
    expand = 0
    while torch.any(val_left <= 0) and expand < 20:
        left = left * 2.0  # -1, -2, -4, ...
        val_left = f(left)
        expand += 1

    # If still not satisfied, fall back to a softmax-like safe value (rare case)
    if torch.any(val_left <= 0):
        tau = right
        support = torch.clamp(x - tau, min=0) ** inv
        return tau, support

    # --- Standard bisection ---
    for _ in range(n_iter):
        mid = (left + right) / 2.0
        val_mid = f(mid)
        left = torch.where(val_mid > 0, mid, left)
        right = torch.where(val_mid <= 0, mid, right)
        if val_mid.numel() > 0 and torch.max(torch.abs(val_mid)) < tol:
            break

    tau = (left + right) / 2.0
    support = torch.clamp(x - tau, min=0) ** inv
    return tau, support


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Entmax α=1.5 variant. Output is non-negative, sums to 1 along the given dimension,
    and is naturally sparse.
    """
    assert logits.numel() > 0, "Empty tensor passed to entmax15!"
    # Translation invariance
    z = logits - logits.max(dim=dim, keepdim=True).values
    # Compute threshold and support
    _, p = _entmax_threshold_and_support(z, alpha=1.5, dim=dim)
    # Normalize (should already sum to 1 numerically, but for safety)
    p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    return p


def topk_select(probs: torch.Tensor, k: int, dim: int = -1):
    """
    Select Top-k probabilities per row and renormalize.

    Input:
      probs: [B_valid, K] non-negative distribution summing to 1
      k:     number of top-k to select (clipped to <= K)

    Return:
      idx: [B_valid, k]  selected fluxon indices (sorted by descending probability)
      w:   [B_valid, k]  renormalized weights
    """
    k = min(k, probs.size(dim))
    values, indices = torch.topk(probs, k, dim=dim, largest=True, sorted=True)
    weights = values / (values.sum(dim=dim, keepdim=True) + 1e-12)
    return indices, weights


class FluxonRouter(nn.Module):
    def __init__(self,
                 in_dim: int,       # input dimension of [h_fast || h_slow]
                 state_dim: int,    # dimension d of each row in A
                 num_fluxons,       # K
                 mode: str = "linear",  # "linear" | "cosine" | "exp"
                 k_select: int = 3,
                 tau_start: float = 2.0,
                 tau_end: float = 0.5,
                 total_steps: int = 1000,
                 device: str = "cpu"):
        super().__init__()
        self.W_Q = nn.Linear(in_dim, state_dim, bias=False).to(device)
        self.W_K = nn.Linear(state_dim, state_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

        self.num_fluxons = num_fluxons
        self.k_select = k_select

        # tau scheduler parameters
        self.mode = mode
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.step_count = 0

    def _update_tau(self):
        """Update tau according to scheduling strategy."""
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)

        if self.mode == "linear":
            tau = self.tau_start - progress * (self.tau_start - self.tau_end)
        elif self.mode == "exp":
            tau = self.tau_end + (self.tau_start - self.tau_end) * (0.95 ** self.step_count)
        elif self.mode == "cosine":
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * progress))
        else:
            tau = self.tau_start
        return tau

    def forward(self, h_concat: torch.Tensor, A_states: torch.Tensor):
        """
        h_concat: (B_valid, in_dim) —— [h_fast || h_slow]
        A_states: (K, state_dim) —— fluxon state matrix A

        Return:
            idx:    [B_valid, k]  selected fluxon indices (sorted by probability)
            weight: [B_valid, k]  normalized weights
            tau:    float         current temperature
        """
        tau = self._update_tau()

        # Queries
        q = self.W_Q(h_concat)  # [B_valid, state_dim]

        # Keys: W_K A
        A_states = A_states.to(h_concat.device)
        K = self.W_K(A_states)  # [K, state_dim]

        # Scores: inner product / tau
        scores = (q @ K.t()) / max(1e-8, tau)  # [B_valid, K]

        # Sparse distribution via entmax
        probs = entmax15(scores, dim=-1)  # [B_valid, K]
        # Alternatively:
        # probs = F.softmax(scores / 1.0, dim=-1)

        # Select top-k
        idx, weight = topk_select(probs, k=self.k_select, dim=-1)

        return idx, weight, tau