"""Fused softmax layer."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride_xm: int,
    stride_xn: int,
    stride_ym: int,
    stride_yn: int,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused softmax kernel.

    This kernel processes one row in per program instance.

    Args:
        input_ptr (int): pointer to the input tensor (MxN).
        output_ptr (int): pointer to the output tensor (MxN).
        stride_xm (int): stride along the rows of the input tensor.
        stride_xn (int): stride along the columns of the input tensor.
        stride_ym (int): stride along the rows of the output tensor.
        stride_yn (int): stride along the columns of the output tensor.
        n_elements (int): number of elements in the input tensor.
        BLOCK_SIZE (int): number of elements per SM.
    """
    pid = tl.program_id(0)

    row_start_idx = pid * stride_xm
    column_offsets = tl.arange(0, BLOCK_SIZE) * stride_xn
    mask = column_offsets < n_elements

    offsets = input_ptr + row_start_idx + column_offsets

    input_ptrs = input_ptr + offsets

    # set out of bounds memory values to -inf to assign zero probability to it
    x = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    row_max = tl.max(x, axis=0)
    # subtract max for numerical stability
    exp_row = tl.exp(x - row_max)
    sum_exp = tl.sum(exp_row, axis=0)

    result = exp_row / sum_exp

    output_offsets = tl.arange(0, BLOCK_SIZE) * stride_yn + pid * stride_ym
    tl.store(output_ptr + output_offsets, result, mask=mask)


def softmax(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)

    n_rows, n_cols = input.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    softmax_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        n_elements=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
