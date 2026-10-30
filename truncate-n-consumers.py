"""
Truncation with 'n' consumers.

A 4096x4096 float32 tensor is truncated to float16 and the truncated result
is consumed by 8 separate pointwise multiplications, each with a different
4096x4096 float16 tensor.

Unlike a broadcast, the truncation is a non-trivial computation whose result
is half the size of its input. Fusing it into its consumers would replicate
the float32 -> float16 conversion in all 8 of them and make each consumer
read the float32 input (twice the bytes of the float16 result) instead of
the truncated tensor. A good compiler should therefore NOT fuse the
truncation here: it should materialize the float16 result once and have the
8 consumers read that.

The compiled result is verified against eager execution.

Usage:
  python truncate-n-consumers.py                    # Inductor (default).
  python truncate-n-consumers.py --backend e0358    # E0358 backend.
  python truncate-n-consumers.py --device cpu
"""

import os

import torch

import common

M, N = 4096, 4096
NUM_CONSUMERS = 8


def truncate_n_consumers(x, ws):
    """Truncates 'x' (MxN float32) to float16 and multiplies it with each of
    'ws' (MxN float16)."""
    t = x.to(torch.float16)
    return tuple(t * w for w in ws)


def make_inputs(device):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32, device=device)
    ws = [
        torch.randn(M, N, dtype=torch.float16, device=device)
        for _ in range(NUM_CONSUMERS)
    ]
    return x, ws


def main():
    args = common.parse_args(__doc__.split("\n\n")[0])
    common.run(truncate_n_consumers, make_inputs, NUM_CONSUMERS, args)

    if os.environ.get("E0358_FUSION_MODE") == "producer":
        from pathlib import Path

        ir = Path("artifacts/bnc.affine.mlir").read_text()
        allocs = [line for line in ir.splitlines() if "memref.alloc" in line]
        assert sum("memref<4096x4096xf16>" in line for line in allocs) == 1
        assert ir.count("arith.truncf") == NUM_CONSUMERS + 1
        print("  fusion check: truncation materialized once (not replicated).")


if __name__ == "__main__":
    main()