import os

import torch
import triton
import triton.language as tl


@triton.jit
def add_f32_kernel(a_ptr, b_ptr, c_ptr, n_elements,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(a_ptr + offsets, mask=mask)
    y = tl.load(b_ptr + offsets, mask=mask)
    output = x + y
    tl.store(c_ptr+offsets,output)
    return


def elementwise_add_f32_kernel(a: torch.Tensor, b: torch.Tensor,c: torch.Tensor):
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_f32_kernel[grid](a,b, c, n_elements, BLOCK_SIZE=256)
    return c
