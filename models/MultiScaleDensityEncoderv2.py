import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDensityEncoder(nn.Module):
    """
    Learnable multi-scale exponential kernel encoder (r only).

    Input:
      deltas_np: numpy array [B, N], time gaps in SECONDS (>=0), already padded
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
        # ---- new: stability & learnability controls
        tau_min_days: float = 0.25,    # avoid tau -> 0
        tau_max_days: float = 365.0,   # avoid tau -> inf and keep gradients sane
        clip_deltas_days: float = 400.0,  # optional: cap huge paddings to keep signal
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_scales = num_scales
        self.eps = eps
        self.device = torch.device(device)
        self.dtype = dtype
        self.log_compress = log_compress
        self.scale_div = scale_div

        self.tau_min_days = float(tau_min_days)
        self.tau_max_days = float(tau_max_days)
        self.clip_deltas_days = float(clip_deltas_days) if clip_deltas_days is not None else None

        if self.tau_min_days <= 0:
            raise ValueError("tau_min_days must be > 0")
        if self.tau_max_days <= self.tau_min_days:
            raise ValueError("tau_max_days must be > tau_min_days")

        # ---- initialize taus in DAYS
        # default: cover 1 day .. 365 days in log space.
        if init_tau_days is None:
            init_tau_days = np.logspace(np.log10(1.0), np.log10(self.tau_max_days), num_scales).astype(np.float64)
        else:
            init_tau_days = np.asarray(init_tau_days, dtype=np.float64)
            if init_tau_days.shape != (num_scales,):
                raise ValueError(f"init_tau_days must have shape ({num_scales},), got {init_tau_days.shape}")

        # clamp init taus into [tau_min, tau_max] to avoid extreme values
        init_tau_days = np.clip(init_tau_days, self.tau_min_days, self.tau_max_days)

        # ---- NEW PARAMETERIZATION: log_tau (learn log tau directly)
        # This avoids inverse-softplus(exp()) overflow and avoids softplus saturation.
        init_log_tau = np.log(init_tau_days + 1e-12).astype(np.float32)
        self.log_tau = nn.Parameter(torch.tensor(init_log_tau, device=self.device, dtype=self.dtype))

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

    def forward(self, deltas_np: np.ndarray, valid_index: np.ndarray) -> torch.Tensor:
        if not isinstance(deltas_np, np.ndarray):
            raise TypeError("deltas_np must be a numpy.ndarray")
        if deltas_np.ndim != 2:
            raise ValueError(f"deltas_np must be 2D [B, N], got shape {deltas_np.shape}")
        if deltas_np.shape[1] != 20:
            raise ValueError(f"Expected N=20, got shape {deltas_np.shape}")

        # [B, N] seconds -> torch
        deltas_sec = torch.from_numpy(deltas_np).to(device=self.device, dtype=self.dtype)
        deltas_sec = torch.clamp(deltas_sec, min=0.0)

        # seconds -> days
        deltas_days = deltas_sec / self.scale_div  # [B, N]

        # optional: cap extremely large paddings so they don't kill all gradients
        if self.clip_deltas_days is not None:
            deltas_days = torch.clamp(deltas_days, max=self.clip_deltas_days)

        # tau in days, strictly positive and bounded
        tau_days = torch.exp(self.log_tau)  # [K]
        tau_days = torch.clamp(tau_days, min=self.tau_min_days, max=self.tau_max_days)
        tau_days = tau_days + self.eps

        # kernel responses per event: [B, N, K]
        phi = torch.exp(-deltas_days.unsqueeze(-1) / tau_days.view(1, 1, -1))

        # aggregate over events: r [B, K]
        r = phi.sum(dim=1)

        # optional compression
        if self.log_compress:
            r = torch.log1p(r)

        # project to embedding: [B, memory_dim]
        z = self.proj(r)

        # debug (remove in training if you want)
        # print(self.get_tau_days())
        # print(r[valid_index])

        return z

    @torch.no_grad()
    def get_tau_days(self) -> torch.Tensor:
        """Current learned taus in DAYS, shape [K]."""
        tau_days = torch.exp(self.log_tau)
        tau_days = torch.clamp(tau_days, min=self.tau_min_days, max=self.tau_max_days)
        return tau_days.detach().clone()
