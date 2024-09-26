import pytest
from kernels.dropout import dropout
import torch


@pytest.fixture
def tensor():
    torch.manual_seed(0)
    return torch.randn((10,), dtype=torch.float32, device="cuda")


def test_dropout_deterministic(tensor: torch.Tensor):
    """Test that dropout is deterministic."""
    dropout_prob = 0.5
    tensor_with_dropout_1 = dropout(tensor, p=dropout_prob, seed=123)
    tensor_with_dropout_2 = dropout(tensor, p=dropout_prob, seed=123)

    scaling_factor = 1 - dropout_prob

    assert torch.allclose(
        tensor_with_dropout_1 * scaling_factor,
        tensor_with_dropout_2 * scaling_factor,
    )


def test_dropout_aligns_with_torch(tensor: torch.Tensor):
    """Test that dropout aligns with torch.nn.functional.dropout."""
    dropout_prob = 0.5
    tensor_with_dropout = dropout(tensor, p=dropout_prob, seed=123)
    tensor_with_torch_dropout = torch.nn.functional.dropout(
        tensor, p=dropout_prob, training=False
    )
    if not torch.allclose(tensor_with_dropout, tensor_with_torch_dropout):
        raise AssertionError("Dropout does not align with torch.nn.functional.dropout.")
