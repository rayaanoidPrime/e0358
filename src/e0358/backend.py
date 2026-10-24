import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

# llvm and stablehlo binary locations
STABLEHLO_BIN = Path(
    os.environ.get("E0358_STABLEHLO_BIN", "~/llvm/stablehlo/build/bin")
).expanduser()
LLVM_BIN = Path(
    os.environ.get("E0358_LLVM_BIN", "~/llvm/llvm-project/build/bin")
).expanduser()


def memref_descriptor(t):
    """Returns an MLIR memref descriptor viewing the memory of tensor 't'."""
    rank = t.dim()

    class MemRef(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRef(
        t.data_ptr(),
        t.data_ptr(),
        0,
        (ctypes.c_int64 * rank)(*t.shape),
        (ctypes.c_int64 * rank)(*t.stride()),
    )


def make_callable(lib, meta):
    """Returns a callable that invokes the compiled MLIR function."""
    func = lib._mlir_ciface_main
    func.restype = None
    # MLIR argument k is fed by user input input_locations[k].position
    order = [loc.position for loc in meta.input_locations]

    def callable_fn(*args):
        # outputs are allocated here and passed as out-params
        outs = [
            torch.empty(s.shape, dtype=getattr(torch, s.dtype))
            for s in meta.output_signature
        ]
        # identity-layout memrefs assume contiguous row-major tensors
        tensors = [args[i].contiguous() for i in order] + outs
        func(*(ctypes.byref(memref_descriptor(t)) for t in tensors))
        return tuple(outs)

    return callable_fn


@torch._dynamo.register_backend(name="e0358")
def e0358_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    sample_inputs = tuple(example_inputs)

    # export to torch.export fx graph
    exported = export(gm, sample_inputs)

    # exported to StableHLO format
    stablehlo_program = exported_program_to_stablehlo(exported)
    stable_hlo_text = stablehlo_program.get_stablehlo_text("forward")
    print(stable_hlo_text, "\n...")

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open("artifacts/bnc.stablehlo.mlir", "w") as f:
        f.write(stable_hlo_text)

    # stablehlo -> mlir -> llvm lowering pipeline to artifacts/bnc.ll; each
    # entry is one tool invocation, run in order
    pipeline = [
        [
            STABLEHLO_BIN / "stablehlo-opt",
            "--stablehlo-legalize-to-linalg",
            "artifacts/bnc.stablehlo.mlir",
            "-o",
            "artifacts/bnc.linalg.mlir",
        ],
        # lower linalg to affine dialect, fuse, vectorize (8 x f32 = AVX2) and
        # parallelize the outermost loop only (nested regions cost more)
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.linalg.mlir",
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map",
            "--buffer-results-to-out-params="
            "hoist-static-allocs=true modify-public-functions=true",
            "--convert-linalg-to-affine-loops",
            "--affine-loop-fusion=mode=greedy",
            "--affine-scalrep",
            # parallelize first: it can't see through vector.transfer ops
            "--affine-parallelize=max-nested=1",
            "--affine-super-vectorize=virtual-vector-size=8",
            "-o",
            "artifacts/bnc.affine.mlir",
        ],
        # lower to llvm dialect; parallel loops run on OpenMP threads.
        # canonicalize drops the empty memref.alloca_scope that scf-to-openmp
        # wraps around the body, which scf-to-cf would otherwise break
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.affine.mlir",
            "--llvm-request-c-wrappers",
            "--lower-affine",
            "--convert-vector-to-scf",
            "--convert-scf-to-openmp",
            "--canonicalize",
            "--convert-scf-to-cf",
            "--convert-vector-to-llvm",
            "--expand-strided-metadata",
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-openmp-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            "artifacts/bnc.llvm.mlir",
        ],
        # translate to llvm ir
        [
            LLVM_BIN / "mlir-translate",
            "--mlir-to-llvmir",
            "artifacts/bnc.llvm.mlir",
            "-o",
            "artifacts/bnc.ll",
        ],
    ]
    for cmd in pipeline:
        result = subprocess.run(
            [str(c) for c in cmd], check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)

    # a fresh path per compilation: dlopen caches libraries by path
    work = Path(tempfile.mkdtemp(prefix="e0358-"))
    subprocess.run(
        [LLVM_BIN / "opt", "-O3", "artifacts/bnc.ll", "-o", work / "m.bc"],
        check=True,
        capture_output=True,
        text=True,
    )

    # compile to object file

    subprocess.run(
        [
            LLVM_BIN / "llc",
            "-filetype=obj",
            "--relocation-model=pic",
            "-mcpu=native",
            work / "m.bc",
            "-o",
            work / "m.o",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # compile to shared library and load

    subprocess.run(
        ["clang", "-shared", "-fopenmp", work / "m.o", "-o", work / "m.so"],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(work / "m.so"))

    meta = stablehlo_program._bundle.stablehlo_funcs[0].meta
    return make_callable(lib, meta)
