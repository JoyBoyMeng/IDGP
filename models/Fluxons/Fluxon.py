import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque


class Fluxon(nn.Module):
    def __init__(self, num_fluxons: int, state_dim: int,
                 half_life_short: float = 21.0, half_life_long: float = 90.0,
                 eps: float = 1e-6, init_type: str = 'zero', device="cpu"):
        """
        A Fluxon class to store group-level embeddings and popularity statistics.

        :param num_fluxons: number of fluxons (K)
        :param state_dim: embedding dimension of each fluxon
        :param half_life_short: short-term EMA half-life (in days, can be converted to seconds)
        :param half_life_long: long-term EMA half-life
        """
        super().__init__()
        self.num_fluxons = num_fluxons
        self.state_dim = state_dim
        self.device = device
        self.init_type = init_type

        # Fluxon state vectors (K, d)
        if self.init_type == 'zero':
            self.states = nn.Parameter(
                torch.zeros((num_fluxons, state_dim), device=device),
                requires_grad=False
            )
        elif self.init_type == 'ball':
            self.states = nn.Parameter(
                torch.empty((num_fluxons, state_dim), device=device),
                requires_grad=False
            )
        else:
            print('init type missing, exit')
            exit()

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        if self.init_type == 'zero':
            self.states.data.zero_()
        elif self.init_type == 'ball':
            with torch.no_grad():
                g = torch.Generator(device=self.device)  # local RNG
                g.manual_seed(42)  # fixed seed

                # Normalize to unit sphere
                # Gaussian sampling: generate a random direction for each fluxon (break symmetry)
                w = torch.randn(self.num_fluxons, self.state_dim, device=self.device)

                # Row-normalize to unit sphere: keep only directional information, norm = 1
                # The scale of K = W_K A will be determined by W_K; A provides directional diversity
                w = w / (w.norm(dim=1, keepdim=True) + 1e-12)

                # Choose norm scale: since W_K (usually Xavier-initialized) follows,
                # keeping norm=1.0 is more stable and avoids early saturation or uniform routing
                target_norm = 1.0
                w = w * target_norm

                # Copy to parameter as initial states (K "trend centers")
                self.states.copy_(w)
        else:
            print('init type missing, exit')
            exit()

    def get_all_fluxon(self):
        """
        Get all fluxon states.
        :return: (K, state_dim)
        """
        return self.states

    def set_all_fluxon(self, idx, updated):
        dev = self.states.device
        D = self.states.size(1)

        # Full overwrite
        if idx is None:
            assert updated.shape == self.states.shape, \
                f"updated shape should be {tuple(self.states.shape)}, but got {tuple(updated.shape)}"
            self.states.copy_(updated.to(dev))
            return

        # Normalize idx: support list / numpy / torch
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx)
        idx = idx.to(dev)

        # Integer indexing
        idx = idx.long().view(-1)
        assert updated.size(0) == idx.numel() and updated.size(1) == D, \
            f"updated shape should be [{idx.numel()}, {D}], but got {tuple(updated.shape)}"

        # Boundary check (optional)
        if torch.any(idx < 0) or torch.any(idx >= self.states.size(0)):
            raise IndexError("idx contains out-of-bound fluxon indices")

        # Use index_copy_ to support duplicate indices (last write takes effect)
        self.states.index_copy_(0, idx, updated.to(dev))

    def detach_memory_bank(self):
        self.states.detach()

    def backup_memory_bank(self):
        return self.states.data.clone()

    def reload_memory_bank(self, backup_memory_bank):
        self.states.data = backup_memory_bank.clone()