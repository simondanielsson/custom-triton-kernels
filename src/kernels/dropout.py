"""Seeded dropout layer."""

import random

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    input_ptr,
    output_ptr,
    p: tl.constexpr,
    seed: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dropout kernel.

    Args:
        input_ptr (int): pointer to the input tensor.
        output_ptr (int): pointer to the output tensor.
        p (float): dropout probability.
        seed (int): random seed.
        n_elements (int): number of elements in the input tensor.
        BLOCK_SIZE (int): number of elements per SM.
    """
    pid = tl.program_id(0)
    block_start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start_idx
    input_ptrs = input_ptr + offsets

    mask = offsets < n_elements

    input = tl.load(input_ptrs, mask=mask)

    input_keep = tl.rand(seed, offsets) > p
    output = tl.where(input_keep, input / (1 - p), 0.0)

    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)


def dropout(input: torch.tensor, p: float, seed: int | None = None):
    """Dropout layer.

    Args:
        input (torch.Tensor): input tensor, must be one dimensional and contiguous.
        p (float): dropout probability.
        seed (int | None): optional random seed.
    """
    if not input.is_contiguous():
        raise ValueError("Input tensor must be contiguous.")

    seed = seed or random.randint(0, 100)

    output = torch.empty_like(input)
    n_elements = input.numel()

    # number of scalars per block
    BLOCK_SIZE = 4

    # meta parameters are retrieved at jit compilation time.
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    dropout_kernel[grid](
        input,
        output,
        p=p,
        seed=seed,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
