"""
Q/K/V projections of an attention layer: three matmuls sharing an operand.

Given activations x (N x d_model) and projection weights W_q, W_k, W_v
(d_model x d_q, d_model x d_k, d_model x d_v), computes

  Q = x @ W_q,  K = x @ W_k,  V = x @ W_v.

The three matmuls share the operand x. A compiler that fuses them ("sibling
fusion") reads each tile of x once and reuses it for all three products,
instead of streaming x from memory three times. The e0358 backend is
expected to perform this fusion on top of a performant CUDA Tile IR lowering
of each matmul.

The problem sizes are drawn from common attention configurations and are
listed in SIZES. For each size, the outputs of the compiled function are
validated against a reference (eager by default, or Inductor), and the
baselines (eager, Inductor, TensorRT) and the backend, both without and with
sibling fusion, are benchmarked.

Usage:
  python qkv-projection.py --backend e0358      # All variants, all sizes.
  python qkv-projection.py --backend e0358 --fusion on --baselines
  python qkv-projection.py --backend e0358 --reference inductor
  python qkv-projection.py                      # Baselines only.
  python qkv-projection.py --sizes bert-base --iters 0   # Validate only.
"""

import collections

import torch

import common

# 'N' is the number of tokens (sequence length), 'd_model' the model width,
# and 'd' the common projection width d_q = d_k = d_v.
Size = collections.namedtuple("Size", ["N", "d_model", "d"])

# Problem sizes, drawn from common attention layers. They cover per-head
# projections with tall-and-skinny weights as well as a full multi-head
# projection with a square weight, and vary N, d_model, and d independently.
SIZES = {
    # BERT-base, one of its 12 heads (12 x 64 = 768), at its maximum
    # sequence length.
    "bert-base": Size(N=512, d_model=768, d=64),
    # GPT-2 medium, all 16 heads projected at once (16 x 64 = 1024), at its
    # full context.
    "gpt2-medium": Size(N=1024, d_model=1024, d=1024),
    # LLaMA-7B, one of its 32 heads (32 x 128 = 4096), at a 4096-token
    # context.
    "llama-7b": Size(N=4096, d_model=4096, d=128),
}


def qkv_projection(x, w_q, w_k, w_v):
    """Returns (x @ W_q, x @ W_k, x @ W_v)."""
    return x @ w_q, x @ w_k, x @ w_v


def make_inputs(size, device, dtype):
    torch.manual_seed(0)
    x = torch.randn(size.N, size.d_model, dtype=dtype, device=device)
    # Scale the weights by 1/sqrt(d_model) so that the outputs are O(1).
    ws = [
        torch.randn(size.d_model, size.d, dtype=dtype, device=device)
        / size.d_model**0.5
        for _ in range(3)
    ]
    return (x, *ws)


def flops(size):
    """Number of floating-point operations in the three matmuls."""
    return 3 * 2 * size.N * size.d_model * size.d


def main():
    args = common.parse_args(__doc__.split("\n\n")[0], SIZES)
    common.run(qkv_projection, make_inputs, SIZES, args, flops=flops)


if __name__ == "__main__":
    main()
