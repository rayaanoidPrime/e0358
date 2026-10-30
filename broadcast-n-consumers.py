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

import argparse
import time

import torch

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


def get_backend(name):
    """Returns the torch.compile backend to use for 'name'."""
    if name == "e0358":
        # The E0358 backend is expected to register itself with Dynamo (e.g.,
        # via torch._dynamo.register_backend(name="e0358")) when imported.
        try:
            import e0358  # noqa: F401
        except ImportError:
            pass
        if "e0358" not in torch._dynamo.list_backends():
            raise SystemExit(
                "Backend 'e0358' is not registered with torch._dynamo; make "
                "sure the 'e0358' module is importable and registers it."
            )
    return name


def bench(fn, args, device, iters):
    """Returns the average execution time of 'fn(*args)' in ms."""
    for _ in range(3):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--backend",
        default="inductor",
        help="torch.compile backend: inductor (default) or e0358",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Timed iterations (0 to skip)"
    )
    args = parser.parse_args()

    x, ws = make_inputs(args.device)

    # Reference: eager execution.
    expected = broadcast_n_consumers(x, ws)

    compiled = torch.compile(
        broadcast_n_consumers, backend=get_backend(args.backend)
    )
    actual = compiled(x, ws)

    assert len(actual) == NUM_CONSUMERS
    for i, (a, e) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            a, e, msg=lambda m, i=i: f"Output {i} mismatch: {m}"
        )
    print(
        f"[{args.backend}/{args.device}] PASSED: {NUM_CONSUMERS} outputs of "
        f"shape {tuple(expected[0].shape)} match eager."
    )

    if args.iters > 0:
        t_eager = bench(broadcast_n_consumers, (x, ws), args.device, args.iters)
        t_comp = bench(compiled, (x, ws), args.device, args.iters)
        print(f"  eager:    {t_eager:8.3f} ms")
        print(f"  {args.backend}: {t_comp:8.3f} ms")


if __name__ == "__main__":
    main()