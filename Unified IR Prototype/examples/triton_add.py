import triton
import triton.language as tl


@triton.jit
def vector_add(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets0 = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offsets0
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)


@triton.jit
def scaled_vector_add(x_ptr, y_ptr, z_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets0 = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + offsets0
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    scaled_x = x * alpha
    z = scaled_x + y
    tl.store(z_ptr + offsets, z, mask=mask)
