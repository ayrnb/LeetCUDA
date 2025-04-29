import os

import torch
import triton
import triton.language as tl


@triton.jit
def histogram_i32_kernel(a_ptr, b_ptr, n_elements,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(a_ptr + offsets, mask=mask)
    tl.atomic_add(b_ptr + x, 1, mask=mask)
    return


def histogram_i32(a:torch.Tensor):
    max=torch.max(a).item()
    y=torch.zeros([max+1],dtype=torch.int32,device=a.device)
    grid = lambda meta: (triton.cdiv(a.numel(), meta['BLOCK_SIZE']), )
    histogram_i32_kernel[grid](a, y, a.numel(), BLOCK_SIZE=256)
    return y
