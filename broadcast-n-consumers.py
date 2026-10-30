"""
Broadcast with 'n' consumers.

A 4096x128 tensor is broadcast to 4096x128x32 and the broadcast result is
consumed by 8 separate pointwise multiplications, each with a different
4096x128x32 tensor. All tensors are float32.

A good compiler should fuse the broadcast into each of its 8 consumers so that
the 4096x128x32 broadcast result is never materialized in memory.

The compiled result is verified against eager execution.

Usage:
  python broadcast-n-consumers.py                    # Inductor (default).
  python broadcast-n-consumers.py --backend e0358    # E0358 backend.
  python broadcast-n-consumers.py --device cpu
"""

import torch

import common

M, N, K = 4096, 128, 32
NUM_CONSUMERS = 8


def broadcast_n_consumers(x, ws):
    """Broadcasts 'x' (MxN) to MxNxK and multiplies it with each of 'ws'."""
    b = x.unsqueeze(-1).expand(M, N, K)
    return tuple(b * w for w in ws)


def make_inputs(device):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32, device=device)
    ws = [
        torch.randn(M, N, K, dtype=torch.float32, device=device)
        for _ in range(NUM_CONSUMERS)
    ]
    return x, ws


def main():
    args = common.parse_args(__doc__.split("\n\n")[0])
    common.run(broadcast_n_consumers, make_inputs, NUM_CONSUMERS, args)


if __name__ == "__main__":
    main()
