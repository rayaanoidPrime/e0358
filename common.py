import argparse
import time

import torch


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def get_backend(name):
    if name == "e0358":
        import e0358  # noqa: F401

        if "e0358" not in torch._dynamo.list_backends():
            raise SystemExit("Backend 'e0358' is not registered")
    return name


def bench(fn, args, device, iters):
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


def run(fn, make_inputs, num_outputs, args):
    inputs = make_inputs(args.device)
    expected = fn(*inputs)
    compiled = torch.compile(fn, backend=get_backend(args.backend))
    actual = compiled(*inputs)

    assert len(actual) == num_outputs
    for i, (a, e) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            a, e, msg=lambda message, i=i: f"Output {i} mismatch: {message}"
        )
    print(
        f"[{args.backend}/{args.device}] PASSED: {num_outputs} outputs of "
        f"shape {tuple(expected[0].shape)} match eager."
    )

    if args.iters > 0:
        eager_ms = bench(fn, inputs, args.device, args.iters)
        compiled_ms = bench(compiled, inputs, args.device, args.iters)
        print(f"  eager:    {eager_ms:8.3f} ms")
        print(f"  {args.backend}: {compiled_ms:8.3f} ms")
