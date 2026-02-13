import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os


class FluxonUpdater(nn.Module):
    """
    Aggregate batch messages according to routing results and update fluxon centers.

    Inputs:
      h_fast:  [B_valid, D]
      h_slow:  [B_valid, D]
      idx:     [B_valid, k]   indices of selected fluxons
      weight:  [B_valid, k]   corresponding gating weights (already normalized)
      fluxon:  should provide get_all_fluxon() and set_all_fluxon(tensor) interfaces;
              states shape is [K, D]

    Hyperparameters:
      state_dim = D      (by default aligned with fast/slow dimension)
      ema_momentum       if >0, apply an EMA blend after the GRU update for stability
    """
    def __init__(self, in_dim: int, state_dim: int, ema_momentum: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        # W_m: map [h_fast || h_slow] to a message vector aligned with state_dim
        self.W_m = nn.Linear(in_dim, state_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.W_m.weight)
        # GRUCell: input_size=hidden_size=state_dim
        self.center_gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim).to(device)
        self.ema_m = float(ema_momentum)

    @torch.no_grad()
    def _ema_blend(self, new_s: torch.Tensor, old_s: torch.Tensor, mask: torch.Tensor):
        """
        Apply EMA blending for updated fluxons (mask==True):
          s = (1-m)*old + m*new

        mask: [K, 1] bool/float
        """
        if self.ema_m <= 0:
            return new_s
        m = self.ema_m
        blended = old_s * (1.0 - m) + new_s * m
        # Apply EMA only to positions that are used; keep others as old_s
        return torch.where(mask, blended, old_s)

    def forward(self,
                h_fast: torch.Tensor,
                h_slow: torch.Tensor,
                idx: torch.Tensor,        # [B_valid, k]
                weight: torch.Tensor,     # [B_valid, k]
                A_states) -> torch.Tensor:
        """
        Return: updated_states [K, D]
        """
        device = h_fast.device
        B, D = h_fast.shape
        assert D == self.state_dim, f"state_dim({self.state_dim}) must match h_fast/h_slow dim ({D})"
        assert h_slow.shape == h_fast.shape
        assert idx.shape == weight.shape and idx.dim() == 2

        # 1) Per-sample message: m_i = W_m([h_fast || h_slow]) -> [B_valid, D]
        x = torch.cat([h_fast, h_slow], dim=-1)              # [B_valid, 2D]
        m_per_sample = self.W_m(x)                           # [B_valid, D]

        # 2) Aggregate messages to each fluxon with routing weights:
        #    agg[k] = Σ_i w_i[k] * m_i
        K_total = A_states.shape[0]  # K
        flat_idx = idx.reshape(-1).to(device).long()         # [B_valid*k]
        flat_w = weight.reshape(-1, 1).to(device)            # [B_valid*k, 1]

        # Align m_i with w_i[k] by repeating over k
        m_rep = m_per_sample.unsqueeze(1).expand(B, idx.size(1), D).reshape(-1, D)  # [B_valid*k, D]
        contrib = flat_w * m_rep  # [B_valid*k, D]

        # 2a) Aggregate messages (dense one-hot approach)
        N = flat_idx.numel()
        M = torch.zeros(N, K_total, device=device)
        M.scatter_(1, flat_idx.view(-1, 1), 1.0)  # build dense one-hot
        agg = M.t() @ contrib  # [K, D]
        wsum = M.t() @ flat_w  # [K, 1]

        # Weighted mean
        agg_mean = agg / (wsum + 1e-9)
        used_mask = (wsum > 0.0)  # [K, 1] bool

        # Debug checkpoint (optional)
        # stats_path = Path("./fluxon_stats/updated_mask.npy")
        # stats_path.parent.mkdir(parents=True, exist_ok=True)
        # updated_now = used_mask.squeeze(-1).detach().cpu().numpy().astype(np.bool_)
        # if stats_path.exists():
        #     try:
        #         updated_total = np.load(stats_path, allow_pickle=False).astype(np.bool_)
        #     except Exception:
        #         updated_total = np.zeros((int(K_total),), dtype=np.bool_)
        # else:
        #     updated_total = np.zeros((int(K_total),), dtype=np.bool_)
        # updated_total |= updated_now
        # np.save(stats_path, updated_total)  # overwrite if exists
        # ever_cnt = int(updated_total.sum())
        # coverage = (ever_cnt / float(K_total)) if K_total > 0 else 0.0
        # print(f"[Fluxion] ever-updated: {ever_cnt}/{int(K_total)} = {coverage * 100:.2f}%")

        # 3) Update centers with GRU: ŝ_k = GRU(m_k, s_k)
        old_states = A_states.to(device)  # [K, D]
        new_all = self.center_gru(agg_mean, old_states)  # [K, D]
        updated = torch.where(used_mask, new_all, old_states)  # keep unchanged if not used

        # 4) Optional EMA blend (only for used positions)
        updated = self._ema_blend(updated, old_states, used_mask)

        return updated
