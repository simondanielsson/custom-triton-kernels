"""Addition kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_vector_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Vector addition kernel.

    Args:
        x_ptr (int): pointer to the first input tensor.
        y_ptr (int): pointer to the second input tensor.
        output_ptr (int): pointer to the output tensor.
        n_elements (int): number of elements in the input tensors.
        BLOCK_SIZE (int): number of elements per SM.
    """
    pid = tl.program_id(0)

    block_start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start_idx

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(output_ptr + offsets, x + y, mask=mask)


def add_vector(x: torch.Tensor, y: torch.Tensor):
    """Element-wise addition of two vectors.

    Args:
        x (torch.Tensor): first input tensor.
        y (torch.Tensor): second input tensor.

    Returns:
        torch.Tensor: output tensor
    """
    if x.shape != y.shape:
        raise ValueError("All inputs must have the same shape.")

    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("All inputs must be contiguous.")

    if not x.is_cuda and y.is_cuda:
        raise ValueError("All inputs must be on the same device.")

    output = torch.empty_like(x)

    n_elements = output.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_elements)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_vector_kernel[grid](x, y, output, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output
