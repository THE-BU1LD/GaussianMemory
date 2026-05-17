
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMemoryBank(nn.Module):
    def __init__(self, dim: int, cells: int = 256, sigma: float = 1.5, decay: float = 0.995, lr: float = 0.08, mode: str = "hybrid"):
        super().__init__()
        self.dim = dim
        self.cells = cells
        self.sigma = sigma
        self.decay = decay
        self.lr = lr
        self.mode = mode

        self.centers = nn.Parameter(torch.randn(cells, dim) * 0.5)
        self.register_buffer("values", torch.zeros(cells, dim))

    def reset(self):
        self.values.zero_()

    def _similarity(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=-1)
        c_n = F.normalize(self.centers, dim=-1)
        cosine = x_n @ c_n.t()
        if self.mode == "cosine":
            return torch.softmax(cosine, dim=-1)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        c2 = (self.centers * self.centers).sum(dim=-1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (x @ self.centers.t())
        gaussian = torch.exp(-dist2 / (2 * self.sigma ** 2))
        if self.mode == "gaussian":
            return gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)
        hybrid = gaussian + 0.5 * (cosine + 1.0)
        return torch.softmax(hybrid, dim=-1)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        w = self._similarity(x)
        values = self.values.detach().clone()
        return w @ values

    @torch.no_grad()
    def write(self, x: torch.Tensor) -> None:
        w = self._similarity(x)
        denom = w.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        update = (w.t() @ x) / denom
        self.values.mul_(self.decay).add_(self.lr * update)

    def forward(self, x: torch.Tensor, write: bool = True) -> torch.Tensor:
        mem = self.read(x)
        if write:
            self.write(x.detach())
        return mem


class DualGaussianMemory(nn.Module):
    def __init__(self, dim: int, cells: int = 256):
        super().__init__()
        self.fast = GaussianMemoryBank(dim, cells=cells, sigma=1.0, decay=0.93, lr=0.15, mode="hybrid")
        self.slow = GaussianMemoryBank(dim, cells=cells, sigma=1.8, decay=0.995, lr=0.05, mode="hybrid")

    def reset(self):
        self.fast.reset()
        self.slow.reset()

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return 0.65 * self.fast.read(x) + 0.35 * self.slow.read(x)

    def forward(self, x: torch.Tensor, write: bool = True) -> torch.Tensor:
        mem = self.read(x)
        if write:
            self.fast.write(x.detach())
            self.slow.write(x.detach())
        return mem


class NeuralFieldLayer(nn.Module):
    def __init__(self, dim: int, decay: float = 0.93):
        super().__init__()
        self.decay = decay
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.U = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state is None or self.state.shape != x.shape:
            self.state = torch.zeros_like(x)
        self.state = self.state.detach()
        synaptic = self.U(x) + x @ self.W
        self.state = self.decay * self.state + synaptic
        return torch.tanh(self.norm(self.state))


class NeuralFieldEncoder(nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([NeuralFieldLayer(dim) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(dim)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)


class Controller(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        g = self.net(torch.cat([x, mem], dim=-1))
        return torch.softmax(g, dim=-1)


class Predictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UncertaintyHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class GMFModel(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        layers: int = 2,
        memory_cells: int = 128,
        refine_steps: int = 3,
        latent_decay: float = 0.85,
    ):
        super().__init__()
        self.dim = dim
        self.encoder = NeuralFieldEncoder(dim, layers=layers)
        self.memory = DualGaussianMemory(dim, cells=memory_cells)
        self.controller = Controller(dim)
        self.predictor = Predictor(dim)
        self.uncertainty = UncertaintyHead(dim)
        self.refine_steps = refine_steps
        self.latent_decay = latent_decay
        self.post = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.final_norm = nn.LayerNorm(dim)

    def reset_state(self):
        self.encoder.reset()
        self.memory.reset()

    def _forward_single(self, seq: torch.Tensor, write_memory: bool = True) -> dict:
        self.reset_state()
        latent = None
        last_mem = None
        last_gate = None
        for t in range(seq.shape[0]):
            x = seq[t:t+1]
            z = self.encoder(x)
            mem = self.memory(z, write=write_memory)
            gates = self.controller(z, mem)
            fused = gates[:, :1] * z + gates[:, 1:2] * mem + gates[:, 2:3] * 0.5 * (z + mem)
            if latent is None:
                latent = fused
            else:
                latent = self.latent_decay * latent + (1.0 - self.latent_decay) * fused
            for _ in range(self.refine_steps):
                delta = self.post(latent)
                latent = self.final_norm(latent + 0.15 * delta)
            last_mem = mem
            last_gate = gates
        prediction = self.predictor(latent)
        uncertainty = self.uncertainty(latent)
        return {
            "latent": latent.squeeze(0),
            "memory": last_mem.squeeze(0),
            "gates": last_gate.squeeze(0),
            "prediction": prediction.squeeze(0),
            "uncertainty": uncertainty.squeeze(0),
        }

    def forward(self, seq: torch.Tensor, write_memory: bool = True) -> dict:
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        outs = [self._forward_single(seq[b], write_memory=write_memory) for b in range(seq.shape[0])]
        keys = outs[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([o[k] for o in outs], dim=0)
        return out


class MeanPoolMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, seq: torch.Tensor) -> dict:
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        x = seq.mean(dim=1)
        return {"prediction": self.net(x)}


class GRUReasoner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, seq: torch.Tensor) -> dict:
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        _, h = self.gru(seq)
        return {"prediction": self.head(h[-1])}


class TransformerReasoner(nn.Module):
    def __init__(self, dim: int, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(256, dim) * 0.01)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, seq: torch.Tensor) -> dict:
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        x = seq + self.pos[: seq.shape[1]]
        z = self.enc(x)
        return {"prediction": self.head(z[:, -1])}
