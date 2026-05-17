
from __future__ import annotations
import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def vicreg_regularizer(x: torch.Tensor, var_coeff: float = 1.0, cov_coeff: float = 0.1) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1 - std))
    cov = (x.T @ x) / (x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return var_coeff * var + cov_coeff * off_diag.pow(2).mean()
