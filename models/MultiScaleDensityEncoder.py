import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDensityEncoder(nn.Module):
    """
    Learnable multi-scale exponential kernel encoder (r only).

    Input:
      deltas_np: numpy array [B, 50], time gaps in SECONDS (>=0), already padded
                using dataset boundary time (large deltas for missing history).

    Output:
      z: torch.Tensor [B, memory_dim]
    """
    def __init__(
        self,
        memory_dim: int,
        num_scales: int = 8,
        eps: float = 1e-8,
        scale_div: float = 86400.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_mlp: bool = False,
        hidden_dim: int = None,
        log_compress: bool = True,
        init_tau_days: np.ndarray = None,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_scales = num_scales
        self.eps = eps
        self.device = torch.device(device)
        self.dtype = dtype
        self.log_compress = log_compress
        self.scale_div = scale_div

        # ---- initialize taus in DAYS (important for your "seconds but min gap ~ day" data)
        # Good default: cover 1 day .. 365 days in log space.
        if init_tau_days is None:
            init_tau_days = np.logspace(np.log10(1.0), np.log10(365.0), num_scales).astype(np.float32)
        else:
            init_tau_days = np.asarray(init_tau_days, dtype=np.float32)
            if init_tau_days.shape != (num_scales,):
                raise ValueError(f"init_tau_days must have shape ({num_scales},), got {init_tau_days.shape}")

        # softplus(alpha) ~= tau  => alpha = log(exp(tau)-1)
        init_alpha = np.log(np.exp(init_tau_days) - 1.0 + 1e-6).astype(np.float32)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, device=self.device, dtype=self.dtype))

        # ---- projection K -> memory_dim
        if use_mlp:
            if hidden_dim is None:
                hidden_dim = max(64, memory_dim)
            self.proj = nn.Sequential(
                nn.Linear(num_scales, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, memory_dim, bias=True),
            ).to(self.device, self.dtype)
        else:
            self.proj = nn.Linear(num_scales, memory_dim, bias=True).to(self.device, self.dtype)

    def forward(self, deltas_np: np.ndarray) -> torch.Tensor:
        if not isinstance(deltas_np, np.ndarray):
            raise TypeError("deltas_np must be a numpy.ndarray")
        if deltas_np.ndim != 2:
            raise ValueError(f"deltas_np must be 2D [B, N], got shape {deltas_np.shape}")
        if deltas_np.shape[1] != 20:
            # not mandatory, but matches your stated setting
            raise ValueError(f"Expected N=20, got shape {deltas_np.shape}")

        # [B, 50] seconds -> torch
        deltas_sec = torch.from_numpy(deltas_np).to(device=self.device, dtype=self.dtype)
        deltas_sec = torch.clamp(deltas_sec, min=0.0)

        # convert seconds -> days
        deltas_days = deltas_sec / 86400.0  # [B, 50]

        # tau in days, positive
        tau_days = F.softplus(self.alpha) + self.eps  # [K]

        # kernel responses per event: [B, 50, K]
        phi = torch.exp(-deltas_days.unsqueeze(-1) / tau_days.view(1, 1, -1))

        # aggregate over events: r [B, K]
        r = phi.sum(dim=1)

        # optional compression to reduce scale sensitivity
        if self.log_compress:
            r = torch.log1p(r)  # log(1+r), stable even if r ~ 0

        # project to embedding: [B, memory_dim]
        z = self.proj(r)
        print(self.get_tau_days())
        return z

    @torch.no_grad()
    def get_tau_days(self) -> torch.Tensor:
        """Current learned taus in DAYS, shape [K]."""
        return (F.softplus(self.alpha) + self.eps).detach().clone()
