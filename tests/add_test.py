"""Tests for the add kernels."""

import torch

from src import add_vector


def test_add_vector():
    vector_size = 512
    x = torch.rand(vector_size, device="cuda")
    y = torch.rand(vector_size, device="cuda")

    output_triton = add_vector(x, y)
    output_torch = x + y
    if not torch.allclose(output_triton, output_torch):
        raise ValueError("Triton and Torch outputs differ.")
