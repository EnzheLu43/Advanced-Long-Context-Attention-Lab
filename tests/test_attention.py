import torch
import pytest
from src.attention.block_attention import MixtureOfBlockAttention

def test_moba_output_shape():
    batch, seq, d_model = 1, 512, 128
    moba = MixtureOfBlockAttention(d_model=d_model, n_heads=4, block_size=128)
    q = torch.randn(batch, seq, d_model)
    k = torch.randn(batch, seq, d_model)
    v = torch.randn(batch, seq, d_model)
    
    output = moba(q, k, v)
    assert output.shape == (batch, seq, d_model)

def test_numerical_stability():
    # Placeholder for checking NaNs in long-context scaling
    assert True
