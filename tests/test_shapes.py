
import torch
from src.gmf.model import GMFModel, MeanPoolMLP, GRUReasoner, TransformerReasoner

def test_gmf_shapes():
    model = GMFModel(dim=32, memory_cells=64)
    seq = torch.randn(4, 9, 32)
    out = model(seq)
    assert out["prediction"].shape == (4, 32)
    assert out["latent"].shape == (4, 32)
    assert out["memory"].shape == (4, 32)

def test_baseline_shapes():
    seq = torch.randn(4, 9, 32)
    for cls in [MeanPoolMLP, GRUReasoner, TransformerReasoner]:
        m = cls(dim=32)
        out = m(seq)
        assert out["prediction"].shape == (4, 32)
