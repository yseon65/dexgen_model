"""DexGen controller networks (DexterityGen, arXiv 2502.04307, Sec. III-C and App. F).

- KeypointDiffusion: noise-prediction network for future finger keypoint offsets dx in R^{T x K x 3}.
  State encoder = 6-layer MLP (1024) with GroupNorm(8); mode embedding; sinusoidal timestep encoder.
  "UNet" with fully connected layers instead of 1D convolutions: hidden 768, 3 encoder + 3 decoder
  blocks with skip connections and FiLM conditioning.
- InverseDynamics: residual MLP that outputs a normal distribution over the action given the state and
  the commanded keypoint motion.
- DDPM training (epsilon prediction, squared-cosine schedule) and DDIM sampling with optional gradient
  guidance toward an input motion (paper Eq. 3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_gn(dims: list[int], groups: int = 8) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GroupNorm(groups, b), nn.Mish()]
    return nn.Sequential(*layers)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        arg = t.float()[:, None] * freq[None]
        return torch.cat([arg.sin(), arg.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """FC residual block: x + W2(Mish(FiLM(GN(W1 x)))), FiLM(scale, shift) from the condition."""

    def __init__(self, dim_in: int, dim_out: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.gn1 = nn.GroupNorm(groups, dim_out)
        self.film = nn.Linear(cond_dim, 2 * dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)
        self.gn2 = nn.GroupNorm(groups, dim_out)
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.gn1(self.fc1(x))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = F.mish(h * (1 + scale) + shift)
        h = F.mish(self.gn2(self.fc2(h)))
        return h + self.skip(x)


class KeypointDiffusion(nn.Module):
    def __init__(self, motion_dim: int, state_dim: int, n_modes: int = 1, hidden: int = 768,
                 state_hidden: int = 1024, state_layers: int = 6, cond_dim: int = 512, n_blocks: int = 3):
        super().__init__()
        self.motion_dim = motion_dim
        self.state_enc = nn.Sequential(mlp_gn([state_dim] + [state_hidden] * state_layers),
                                       nn.Linear(state_hidden, cond_dim))
        self.mode_emb = nn.Embedding(n_modes, cond_dim)
        self.time_enc = nn.Sequential(SinusoidalEmbedding(128), nn.Linear(128, cond_dim), nn.Mish(),
                                      nn.Linear(cond_dim, cond_dim))
        self.cond_proj = nn.Sequential(nn.Mish(), nn.Linear(3 * cond_dim, cond_dim))
        self.inp = nn.Linear(motion_dim, hidden)
        self.down = nn.ModuleList([FiLMBlock(hidden, hidden, cond_dim) for _ in range(n_blocks)])
        self.mid = FiLMBlock(hidden, hidden, cond_dim)
        self.up = nn.ModuleList([FiLMBlock(2 * hidden, hidden, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, motion_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def condition(self, state: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        """State/mode part of the condition; computed once per sampling call."""
        return torch.cat([self.state_enc(state), self.mode_emb(mode)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, sm_cond: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(torch.cat([sm_cond, self.time_enc(t)], dim=-1))
        h = self.inp(x)
        skips = []
        for blk in self.down:
            h = blk(h, cond)
            skips.append(h)
        h = self.mid(h, cond)
        for blk in self.up:
            h = blk(torch.cat([h, skips.pop()], dim=-1), cond)
        return self.out(h)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class InverseDynamics(nn.Module):
    """p(a | state, dx) = N(mu, sigma^2)."""

    def __init__(self, state_dim: int, motion_dim: int, action_dim: int, hidden: int = 1024, n_blocks: int = 4):
        super().__init__()
        self.inp = nn.Linear(state_dim + motion_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * action_dim))

    def forward(self, state: torch.Tensor, motion: torch.Tensor):
        mu, log_std = self.head(self.blocks(self.inp(torch.cat([state, motion], dim=-1)))).chunk(2, dim=-1)
        return mu, log_std.clamp(-7.0, 2.0)

    @staticmethod
    def loss(mu, log_std, target):
        """MSE on the mean (paper) + Gaussian NLL for the std with the mean detached."""
        mse = F.mse_loss(mu, target)
        nll = (log_std + 0.5 * ((target - mu.detach()) / log_std.exp()) ** 2).mean()
        return mse + nll, mse


class DDPMSchedule:
    def __init__(self, n_steps: int = 100, device="cpu"):
        self.n = n_steps
        s = 0.008
        x = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64)
        ac = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        self.alphas_cumprod = torch.cumprod(1 - betas, 0).float().to(device)

    def add_noise(self, x0, noise, t):
        a = self.alphas_cumprod[t][:, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model: KeypointDiffusion, x0, state, mode):
        t = torch.randint(0, self.n, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.add_noise(x0, noise, t), t, model.condition(state, mode))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model: KeypointDiffusion, state, mode, n_inference: int = 10,
                    guide: torch.Tensor | None = None, guide_scale: float = 0.0, generator=None, clip: float = 6.0):
        """DDIM (eta=0); x0 is clipped to +-clip (normalized units). Optional guidance: x0 <- x0 - guide_scale * (1-a) * grad ||x0 - guide||^2 (paper III-D)."""
        b = state.shape[0]
        sm = model.condition(state, mode)
        x = torch.randn((b, model.motion_dim), device=state.device, generator=generator)
        ts = torch.linspace(self.n - 1, 0, n_inference, device=state.device).round().long()
        for i, t in enumerate(ts):
            a = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=state.device)
            eps = model(x, t.expand(b), sm)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-clip, clip)
            if guide is not None and guide_scale > 0:
                x0 = x0 - guide_scale * (1 - a) * 2 * (x0 - guide)
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
