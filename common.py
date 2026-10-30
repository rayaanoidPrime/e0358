"""
Utilities shared by the E0358 Assignment 1 test scripts.

Each test script defines a workload (a function to compile and a routine
that builds its inputs) and defers the rest to this module: command-line
parsing, resolving the torch.compile backend, verifying the compiled
outputs against eager execution, and timing.
"""

import argparse
import time

import torch


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


def parse_args(description):
    """Parses the command-line flags common to all test scripts."""
    parser = argparse.ArgumentParser(description=description)
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
    return parser.parse_args()


def run(fn, make_inputs, num_outputs, args):
    """Compiles 'fn' with the requested backend, checks it against eager
    execution, and times both.

    'make_inputs(device)' returns the tuple of arguments to call 'fn' with;
    'fn' returns a tuple of 'num_outputs' tensors.
    """
    inputs = make_inputs(args.device)

    # Reference: eager execution.
    expected = fn(*inputs)

    compiled = torch.compile(fn, backend=get_backend(args.backend))
    actual = compiled(*inputs)

    assert len(actual) == num_outputs
    for i, (a, e) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            a, e, msg=lambda m, i=i: f"Output {i} mismatch: {m}"
        )
    print(
        f"[{args.backend}/{args.device}] PASSED: {num_outputs} outputs of "
        f"shape {tuple(expected[0].shape)} ({expected[0].dtype}) match eager."
    )

    if args.iters > 0:
        t_eager = bench(fn, inputs, args.device, args.iters)
        t_comp = bench(compiled, inputs, args.device, args.iters)
        print(f"  eager:    {t_eager:8.3f} ms")
        print(f"  {args.backend}: {t_comp:8.3f} ms")