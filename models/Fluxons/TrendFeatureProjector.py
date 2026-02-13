import torch
import torch.nn as nn

class TrendFeatureProjector(nn.Module):
    def __init__(self, slow_feat_dim: int, trend_dim: int, hidden: int = None, device: str = 'cpu'):
        super().__init__()
        hidden = hidden or max(64, trend_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(slow_feat_dim),
            nn.Linear(slow_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, trend_dim),
        ).to(device)

    def forward(self, slow_feat: torch.Tensor) -> torch.Tensor:
        return self.net(slow_feat)
