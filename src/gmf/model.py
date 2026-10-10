
from __future__ import annotations

import math
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMemoryBank(nn.Module):
    def __init__(self, dim: int, cells: int = 256, sigma: float = 1.5, decay: float = 0.995, lr: float = 0.08, mode: str = "hybrid"):
        super().__init__()
        for name, value in (("dim", dim), ("cells", cells)):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a positive integer")
            if operator.index(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if mode not in {"gaussian", "cosine", "hybrid"}:
            raise ValueError("mode must be gaussian, cosine or hybrid")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("decay must be finite and in [0, 1]")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        self.dim = operator.index(dim)
        self.cells = operator.index(cells)
        self.sigma = float(sigma)
        self.decay = float(decay)
        self.lr = float(lr)
        self.mode = mode

        self.centers = nn.Parameter(torch.randn(cells, dim) * 0.5)
        self.register_buffer("values", torch.zeros(cells, dim))

    def reset(self):
        self.values.zero_()

    def _similarity(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("memory input must be a real floating-point tensor")
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.dim:
            raise ValueError(f"memory input must have shape [batch, {self.dim}]")
        if x.device != self.centers.device or x.dtype != self.centers.dtype:
            raise ValueError("memory input must match the bank's device and dtype")
        if not torch.isfinite(x).all() or not torch.isfinite(self.centers).all():
            raise ValueError("memory input and centers must be finite")
        # CPU distance kernels require float32 or float64. Promotion also
        # prevents fp16 Gaussian exponent/normalization underflow.
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        query = x.to(dtype=work_dtype)
        centers = self.centers.to(dtype=work_dtype)
        x_n = F.normalize(query, dim=-1)
        c_n = F.normalize(centers, dim=-1)
        cosine = x_n @ c_n.t()
        if self.mode == "cosine":
            return torch.softmax(cosine, dim=-1).to(dtype=x.dtype)
        # Direct distances avoid catastrophic cancellation in
        # ||x||^2 + ||c||^2 - 2*x.c near large, almost-equal vectors.
        distance = torch.cdist(query, centers, compute_mode="donot_use_mm_for_euclid_dist")
        log_gaussian = -0.5 * (distance / self.sigma).square()
        if not torch.isfinite(log_gaussian).all():
            raise FloatingPointError("Gaussian distances exceed the numeric range")
        if self.mode == "gaussian":
            # Normalizing exponentials directly made far queries return
            # all-zero weights. Softmax performs the same normalization in
            # log space, retaining unit probability mass.
            return torch.softmax(log_gaussian, dim=-1).to(dtype=x.dtype)
        gaussian = torch.exp(log_gaussian)
        hybrid = gaussian + 0.5 * (cosine + 1.0)
        return torch.softmax(hybrid, dim=-1).to(dtype=x.dtype)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        w = self._similarity(x)
        values = self.values.detach().clone()
        return w @ values

    @torch.no_grad()
    def write(self, x: torch.Tensor) -> None:
        w = self._similarity(x)
        denom = w.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        update = (w.t() @ x) / denom
        candidate = self.values * self.decay + self.lr * update
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("memory update produced non-finite values")
        self.values.copy_(candidate)

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
        *,
        ablation: str = "full",
        execution_mode: str = "serial-v1",
    ):
        super().__init__()
        from .ablations import ablation_plan

        settings = ablation_plan()
        if not isinstance(ablation, str) or ablation not in settings:
            raise ValueError(f"unknown GMF ablation: {ablation!r}")
        if execution_mode not in {"serial-v1", "batched-v1"}:
            raise ValueError("execution_mode must be serial-v1 or batched-v1")
        self.execution_mode = execution_mode
        self.ablation = ablation
        self._ablation_settings = settings[ablation]
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
            if not self._ablation_settings["memory"]:
                mem = torch.zeros_like(z)
            elif self._ablation_settings["dual_memory"]:
                mem = self.memory(z, write=write_memory)
            else:
                mem = self.memory.fast(z, write=write_memory)
            if self._ablation_settings["controller"]:
                gates = self.controller(z, mem)
            else:
                gates = z.new_full((z.shape[0], 3), 1.0 / 3.0)
            fused = gates[:, :1] * z + gates[:, 1:2] * mem + gates[:, 2:3] * 0.5 * (z + mem)
            if latent is None:
                latent = fused
            else:
                latent = self.latent_decay * latent + (1.0 - self.latent_decay) * fused
            refinement_steps = self.refine_steps if self._ablation_settings["refine"] else 0
            for _ in range(refinement_steps):
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

    @staticmethod
    def _independent_memory(bank, x, values, write):
        """Read/write one private bank per row; never pool the batch's writes."""
        weights = bank._similarity(x)
        memory = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        if write:
            with torch.no_grad():
                # The serial bank receives one row: its cell denominator is
                # that row's weight, not a reduction across independent rows.
                w = weights.detach().unsqueeze(-1)
                update = (w * x.detach().unsqueeze(1)) / w.clamp_min(1e-6)
                candidate = values * bank.decay + bank.lr * update
                if not torch.isfinite(candidate).all():
                    raise FloatingPointError("memory update produced non-finite values")
                values = candidate
        return memory, values

    def _forward_independent_batch(self, seq, write_memory):
        """Batch deterministic recurrence, retaining serial predictor RNG order."""
        if (seq.ndim != 3 or seq.shape[0] < 1 or seq.shape[1] < 1
                or seq.shape[2] != self.dim or not seq.is_floating_point()):
            raise ValueError("batched GMF input must be nonempty [batch, time, dim] floating data")
        parameter = next(self.parameters())
        if seq.device != parameter.device or seq.dtype != parameter.dtype:
            raise ValueError("batched GMF input must match the model device and dtype")
        if not torch.isfinite(seq).all():
            raise ValueError("batched GMF input must be finite")
        self.reset_state()
        batch = seq.shape[0]
        settings = self._ablation_settings
        fast_values = seq.new_zeros((batch, self.memory.fast.cells, self.dim))
        slow_values = (seq.new_zeros((batch, self.memory.slow.cells, self.dim))
                       if settings["memory"] and settings["dual_memory"] else None)
        latent = None
        for t in range(seq.shape[1]):
            z = self.encoder(seq[:, t])
            if not settings["memory"]:
                memory = torch.zeros_like(z)
            else:
                fast, fast_values = self._independent_memory(
                    self.memory.fast, z, fast_values, write_memory)
                if settings["dual_memory"]:
                    slow, slow_values = self._independent_memory(
                        self.memory.slow, z, slow_values, write_memory)
                    memory = 0.65 * fast + 0.35 * slow
                else:
                    memory = fast
            gates = (self.controller(z, memory) if settings["controller"]
                     else z.new_full((batch, 3), 1.0 / 3.0))
            fused = (gates[:, :1] * z + gates[:, 1:2] * memory
                     + gates[:, 2:3] * 0.5 * (z + memory))
            latent = (fused if latent is None else
                      self.latent_decay * latent + (1.0 - self.latent_decay) * fused)
            for _ in range(self.refine_steps if settings["refine"] else 0):
                latent = self.final_norm(latent + 0.15 * self.post(latent))
        # No earlier maintained component draws RNG. Preserve each predictor's
        # dropout shape/order instead of replacing it with a B-row dropout call.
        predictions, uncertainties = [], []
        for b in range(batch):
            predictions.append(self.predictor(latent[b:b + 1]).squeeze(0))
            uncertainties.append(self.uncertainty(latent[b:b + 1]).squeeze(0))
        with torch.no_grad():
            self.memory.fast.values.copy_(fast_values[-1])
            if slow_values is not None:
                self.memory.slow.values.copy_(slow_values[-1])
        # Serial forwarding leaves the last sequence's state visible. Keep the
        # same convention without allowing state from any row into another.
        for layer in self.encoder.layers:
            layer.state = layer.state[-1:]
        return {"latent": latent, "memory": memory, "gates": gates,
                "prediction": torch.stack(predictions),
                "uncertainty": torch.stack(uncertainties)}

    def forward(self, seq: torch.Tensor, write_memory: bool = True) -> dict:
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if self.execution_mode == "batched-v1":
            return self._forward_independent_batch(seq, write_memory)
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
