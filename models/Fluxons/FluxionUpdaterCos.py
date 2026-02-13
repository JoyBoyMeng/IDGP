import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxonUpdaterCos(nn.Module):
    """
    Fluxon updater for "single-choice routing" (idx: [B, 1], no weight).

    Inputs:
      h_fast:   [B, D]
      h_slow:   [B, D]
      idx:      [B, 1]      selected fluxion index for each sample (single choice)
      A_states: [K, 2D]     fluxion center matrix (Tensor)

    Hyperparameters:
      in_dim        = 2D (if using [h_fast || h_slow], then in_dim should be 2*D)
      state_dim     = 2D (must match the last dimension of A_states)
      ema_momentum  ∈ [0, 1]; if > 0, apply an EMA blend after GRU to improve stability
    """
    def __init__(self, in_dim: int, state_dim: int, ema_momentum: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = int(state_dim)

        # GRUCell: input_size=hidden_size=state_dim
        self.center_gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim).to(device)

        self.ema_m = float(ema_momentum)

    def _ema_blend_sel(self, new_sel: torch.Tensor, old_sel: torch.Tensor):
        """Apply EMA blending only to the selected entries; unselected entries are handled elsewhere."""
        if self.ema_m <= 0:
            return new_sel
        m = self.ema_m
        return old_sel * (1.0 - m) + new_sel * m

    def forward(self,
                h_fast: torch.Tensor,
                h_slow: torch.Tensor,
                idx: torch.Tensor,          # [B, 1]
                A_states: torch.Tensor      # [K, 2D]
                ) -> torch.Tensor:
        """
        Return: updated_states [K, 2D] (only update the fluxions hit by routing in this batch)
        """
        device = h_fast.device
        B, D = h_fast.shape
        assert 2 * D == self.state_dim, f"state_dim({self.state_dim}) must match h_fast+h_slow dim ({2*D})"
        assert h_slow.shape == h_fast.shape, "h_fast and h_slow must have the same shape"
        assert idx.dim() == 2 and idx.size(0) == B and idx.size(1) == 1, f"idx must be [B,1], got {idx.shape}"
        assert A_states.dim() == 2 and A_states.size(1) == 2 * D, f"A_states must be [K,{2*D}], got {A_states.shape}"

        # 1) Per-sample message vector: m_i = [h_fast || h_slow] -> [B, 2D]
        message = torch.cat([h_fast, h_slow], dim=-1)  # [B, 2D]

        # 2) Group samples by their selected fluxion and take the mean message per group
        flat_idx = idx.view(-1).to(device).long()                # [B]
        uniq, inv = torch.unique(flat_idx, return_inverse=True)  # uniq:[U], inv:[B] maps each sample to a group

        # Efficient segmented mean via sorting + prefix sums
        sorted_inv, perm = torch.sort(inv, stable=True)  # [B]
        msg_sorted = message[perm]                       # [B, 2D]

        # Prefix sums
        csum = torch.cumsum(msg_sorted, dim=0)           # [B, 2D]

        # Segment boundaries
        change = torch.ones(B, dtype=torch.bool, device=device)
        change[1:] = sorted_inv[1:] != sorted_inv[:-1]
        starts = torch.nonzero(change, as_tuple=False).squeeze(1)  # [U]
        ends = torch.cat([starts[1:], torch.tensor([B], device=device)])  # [U]

        # Segment sums: csum[end-1] - csum[start-1]
        end_csum = csum[ends - 1]                         # [U, 2D]
        start_csum = torch.zeros_like(end_csum)
        valid = starts > 0
        start_csum[valid] = csum[starts[valid] - 1]
        agg = end_csum - start_csum                       # [U, 2D]

        # Segment counts
        cnt = (ends - starts).unsqueeze(1).to(message.dtype)  # [U, 1]
        m_mean = agg / (cnt + 1e-9)                           # [U, 2D] group mean

        # 3) Apply GRU update only to the selected fluxions
        old_states = A_states.to(device)        # [K, 2D]
        updated = old_states.clone()

        old_sel = old_states[uniq]              # [U, 2D]
        new_sel = self.center_gru(m_mean, old_sel)  # [U, 2D]

        # 4) Optional EMA blending (selected rows only)
        new_sel = self._ema_blend_sel(new_sel, old_sel)

        # 5) Write back updated entries (slice assignment can carry gradients)
        updated[uniq] = new_sel
        # Alternatively:
        # updated = torch.index_copy(old_states, 0, uniq, new_sel)

        return updated
