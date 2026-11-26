"""
Utilities shared by the E0358 Assignment 2 test scripts.

A test script defines a workload (a function to compile, a routine that
builds its inputs, and the problem sizes to run it at) and defers the rest
to this module: command-line parsing, resolving the torch.compile backends,
validating the compiled outputs against a reference (eager or Inductor),
and benchmarking with warmup iterations followed by timed steady-state
iterations.

Every run measures a set of variants: the baselines (eager execution,
Inductor, and TensorRT through Torch-TensorRT) and the backend under test,
which for the e0358 backend is run both without and with sibling fusion.
"""

import argparse
import collections
import importlib
import importlib.util
import os
import statistics
import time

import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances for validating compiled outputs against the reference. These are
# looser than torch.testing's defaults to allow for a different accumulation
# order in matmuls with large reduction dimensions.
TOLERANCES = {
    torch.float32: dict(rtol=1e-4, atol=1e-3),
    torch.float16: dict(rtol=1e-2, atol=2e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=5e-2),
}

# The baselines, in the order they are reported. 'tensorrt' is the backend
# that Torch-TensorRT (pip install torch-tensorrt) registers with Dynamo.
BASELINES = ["eager", "inductor", "tensorrt"]

# The modules that register the non-built-in backends with torch._dynamo.
BACKEND_MODULES = {"e0358": "e0358", "tensorrt": "torch_tensorrt"}

# The harness sets this environment variable to "1" or "0" right before a
# backend is invoked to compile a graph; the e0358 backend reads it to decide
# whether to perform sibling fusion.
FUSION_ENV_VAR = "E0358_SIBLING_FUSION"

# A variant to run: 'backend' is the torch.compile backend ('eager' for eager
# execution) and 'fusion' is None, or True/False for the e0358 backend with
# or without sibling fusion.
Variant = collections.namedtuple("Variant", ["label", "backend", "fusion"])


def get_backend(name):
    """Makes sure that the torch.compile backend 'name' is registered with
    Dynamo, importing the module that registers it if needed, and returns
    'name'."""
    module = BACKEND_MODULES.get(name)
    if module and name not in torch._dynamo.list_backends():
        try:
            importlib.import_module(module)
        except ImportError as err:
            raise SystemExit(
                f"Backend '{name}' needs the '{module}' module, which could "
                f"not be imported: {err}"
            ) from None
    if name not in torch._dynamo.list_backends():
        raise SystemExit(
            f"Backend '{name}' is not registered with torch._dynamo"
            + (f" by the '{module}' module." if module else ".")
        )
    return name


def tensorrt_options(dtype):
    """Options for compiling with the 'tensorrt' backend (Torch-TensorRT)."""
    return {
        # The default min_block_size of 5 would leave a graph of three
        # matmuls to PyTorch; require TensorRT to run all of it instead.
        "min_block_size": 1,
        "require_full_compilation": True,
        "enabled_precisions": {dtype},
        # Use TF32 for float32 matmuls only if PyTorch itself does (by
        # default it does not: its float32 matmul precision is "highest").
        "disable_tf32": torch.get_float32_matmul_precision() == "highest",
    }


def with_fusion(name, fusion):
    """Returns a torch.compile backend that sets FUSION_ENV_VAR to "1"
    ('fusion' True) or "0" and then invokes the backend 'name' to compile
    the graph.

    Every call returns a distinct callable, so that a function compiled with
    both the fusion and the no-fusion backend is compiled twice, rather than
    the second reusing the code Dynamo compiled for the first.
    """
    base = torch._dynamo.lookup_backend(name)

    def backend(gm, example_inputs, **kwargs):
        os.environ[FUSION_ENV_VAR] = "1" if fusion else "0"
        return base(gm, example_inputs, **kwargs)

    backend.__name__ = f"{name}_{'fusion' if fusion else 'no_fusion'}"
    return backend


def compile_variant(fn, variant, dtype):
    """Returns 'fn' compiled as 'variant' ('fn' itself for eager)."""
    if variant.backend == "eager":
        return fn
    backend = get_backend(variant.backend)
    options = None
    if variant.backend == "tensorrt":
        options = tensorrt_options(dtype)
    if variant.fusion is not None:
        backend = with_fusion(backend, variant.fusion)
    return torch.compile(fn, backend=backend, dynamic=False, options=options)


def variants(args):
    """Returns the variants to run, in the order they are reported: the
    selected baselines (always including the reference), followed by the
    backend under test under each selected sibling fusion setting."""
    result = []
    for name in BASELINES:
        if name not in args.baselines and name not in (
            args.backend,
            args.reference,
        ):
            continue
        module = BACKEND_MODULES.get(name)
        if (
            name != args.backend
            and module
            and importlib.util.find_spec(module) is None
        ):
            print(
                f"warning: skipping the {name} baseline: the '{module}' "
                f"module is not installed"
            )
            continue
        result.append(Variant(name, name, None))
    if args.backend not in BASELINES:
        settings = {"off": [False], "on": [True], "both": [False, True]}
        for fusion in settings[args.fusion]:
            label = f"{args.backend} ({'fusion' if fusion else 'no fusion'})"
            result.append(Variant(label, args.backend, fusion))
    return result


def parse_args(description, sizes):
    """Parses the command-line flags common to all test scripts. 'sizes' maps
    problem-size names to the sizes themselves."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        default="inductor",
        help="backend to validate and benchmark: inductor (default), "
        "tensorrt, or e0358",
    )
    parser.add_argument(
        "--reference",
        default="eager",
        choices=["eager", "inductor"],
        help="what to validate the compiled outputs against (default: eager)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPES),
        help="element type of all tensors (default: float32)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(sizes),
        choices=list(sizes),
        metavar="SIZE",
        help=f"problem sizes to run (default: all): {', '.join(sizes)}",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=BASELINES,
        choices=BASELINES,
        metavar="NAME",
        help="baselines to benchmark alongside the backend (default: "
        f"{' '.join(BASELINES)}; give no names to benchmark none)",
    )
    parser.add_argument(
        "--fusion",
        default="both",
        choices=["on", "off", "both"],
        help="sibling fusion setting(s) to run the e0358 backend with, i.e., "
        f"the value(s) of {FUSION_ENV_VAR} (default: both)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed warmup iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="timed steady-state iterations (default: 50; 0 to only validate)",
    )
    parser.add_argument(
        "--stat",
        default="median",
        choices=["median", "mean"],
        help="statistic of the timed iterations to report (default: median)",
    )
    return parser.parse_args()


def check(actual, expected, dtype):
    """Compares the output tuples 'actual' and 'expected'. Returns whether
    they match within tolerance, the largest absolute error, and the
    message of the first mismatch (or None)."""
    assert len(actual) == len(expected)
    ok, max_err, message = True, 0.0, None
    for i, (a, e) in enumerate(zip(actual, expected)):
        max_err = max(max_err, (a.float() - e.float()).abs().max().item())
        try:
            torch.testing.assert_close(a, e, **TOLERANCES[dtype])
        except AssertionError as err:
            if ok:
                message = f"Output {i} mismatch: {err}"
            ok = False
    return ok, max_err, message


def timeit(fn, args, device, warmup, iters):
    """Runs 'fn(*args)' for 'warmup' untimed iterations followed by 'iters'
    timed ones, and returns the per-iteration times in ms."""
    for _ in range(warmup):
        fn(*args)
    times = []
    if device == "cuda":
        # Record a pair of events around every iteration and synchronize
        # once at the end, so that timing does not perturb the steady state.
        torch.cuda.synchronize()
        events = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in events]
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - start) * 1e3)
    return times


def summarize(times, stat):
    """Returns the requested statistic ('median' or 'mean') of 'times'."""
    if stat == "median":
        return statistics.median(times)
    return statistics.fmean(times)


def print_table(plan, names, sizes, times, args, flops):
    """Prints the benchmark results: one row per variant in 'plan' and one
    column per problem size in 'names'. 'times' maps a variant label and a
    size name to the time in ms."""
    print()
    print(
        f"{args.stat} of {args.iters} iterations after {args.warmup} warmup "
        f"iterations; time in ms"
        + (" (TFLOP/s in parentheses)" if flops else "")
        + "."
    )
    dims = {name: "x".join(str(d) for d in sizes[name]) for name in names}
    fields = " x ".join(getattr(sizes[names[0]], "_fields", ["size"]))
    label_w = max(max(len(v.label) for v in plan), len(fields))
    col_w = max(max(len(s) for s in [*names, *dims.values()]), 15) + 2
    print(f"{'':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names))
    print(
        f"{fields:<{label_w}}" + "".join(f"{dims[n]:>{col_w}}" for n in names)
    )
    print("-" * (label_w + col_w * len(names)))
    for v in plan:
        line = f"{v.label:<{label_w}}"
        for name in names:
            t = times[v.label][name]
            cell = f"{t:.3f}"
            if flops:
                cell += f" ({flops(sizes[name]) / t / 1e9:.1f})"
            line += f"{cell:>{col_w}}"
        print(line)


def run(fn, make_inputs, sizes, args, flops=None):
    """For each selected problem size: builds inputs, validates 'fn' compiled
    with args.backend against the reference, and benchmarks every variant
    (the baselines and the backend, without and with sibling fusion).

    'make_inputs(size, device, dtype)' returns the tuple of arguments to call
    'fn' with, and 'fn' returns a tuple of tensors. 'flops(size)', if given,
    returns the number of floating-point operations 'fn' performs, used to
    report the achieved TFLOP/s.
    """
    dtype = DTYPES[args.dtype]
    plan = variants(args)
    for v in plan:
        if v.backend != "eager":
            get_backend(v.backend)
    print(
        f"backend={args.backend} reference={args.reference} "
        f"device={args.device} dtype={args.dtype}"
    )
    print("variants: " + ", ".join(v.label for v in plan))

    times = {v.label: {} for v in plan}
    failed = []
    for name in args.sizes:
        size = sizes[name]
        inputs = make_inputs(size, args.device, dtype)

        # Compile afresh for every size with static shapes, so that each
        # size is compiled on its own and not through a graph specialized
        # (or made dynamic) for a previous one.
        torch._dynamo.reset()
        fns = {v.label: compile_variant(fn, v, dtype) for v in plan}

        expected = fns[args.reference](*inputs)
        for v in plan:
            if v.backend != args.backend:
                continue
            actual = fns[v.label](*inputs)
            ok, max_err, message = check(actual, expected, dtype)
            print(
                f"[{v.label}/{args.device}] {name}: "
                f"{'PASSED' if ok else 'FAILED'} vs {args.reference} "
                f"({len(expected)} outputs of shape "
                f"{tuple(expected[0].shape)}, max abs err {max_err:.3e})"
            )
            if not ok:
                print(f"  {message}")
                failed.append(f"{v.label} at {name}")

        if args.iters > 0:
            for v in plan:
                times[v.label][name] = summarize(
                    timeit(
                        fns[v.label],
                        inputs,
                        args.device,
                        args.warmup,
                        args.iters,
                    ),
                    args.stat,
                )

    if args.iters > 0:
        print_table(plan, args.sizes, sizes, times, args, flops)
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")
