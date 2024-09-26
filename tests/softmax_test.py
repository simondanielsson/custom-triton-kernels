"""Test the softmax kernel."""

import torch

from src import softmax


def test_softmax_close_to_torch():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")

    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch)
