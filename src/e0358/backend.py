import re

import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

# llvm and stablehlo binary locations
STABLEHLO_BIN = Path("~/llvm/stablehlo/build/bin").expanduser()
LLVM_BIN = Path("~/llvm/llvm-project/build/bin").expanduser()
# holds libmlir_cuda_runtime.so (needs -DMLIR_ENABLE_CUDA_RUNNER=ON)
LLVM_LIB = Path(str(LLVM_BIN.parent / "lib")).expanduser()

# scf-parallel-loop-tiling sizes for the gpu path, one per parallel loop
# TODO for further optimizations
GPU_TILE_SIZES = os.environ.get("E0358_GPU_TILE_SIZES", "1,128")

# nvidia target for the gpu path (e.g., sm_86)
GPU_CHIP = os.environ.get("E0358_GPU_CHIP")


def gpu_chip(device):
    """Returns the sm_XY target to compile the kernels for."""
    if GPU_CHIP:
        return GPU_CHIP
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


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


def make_callable(lib, meta, device):
    func = lib._mlir_ciface_main
    func.restype = None
    # MLIR argument k is fed by user input input_locations[k].position
    order = [loc.position for loc in meta.input_locations]

    def callable_fn(*args):
        # outputs are allocated here and passed as out-params; on cuda the
        # descriptors carry device pointers, which the kernels use directly
        outs = [
            torch.empty(s.shape, dtype=getattr(torch, s.dtype), device=device)
            for s in meta.output_signature
        ]
        # we pass identity-layout bufferization
        # so we need contiguous row-major tensors
        tensors = [args[i].contiguous() for i in order] + outs
        if device.type == "cuda":
            # the mlir cuda runtime launches on its own non-blocking stream,
            # so wait for torch's pending work
            # the launch itself synchronizes before returning
            torch.cuda.synchronize(device)
        func(*(ctypes.byref(memref_descriptor(t)) for t in tensors))
        return tuple(outs)

    return callable_fn


def check_no_host_allocs(path):
    # Fails if 'path' has heap allocations, which would be host memory
    if re.search(r"memref\.alloc\(", Path(path).read_text()):
        raise RuntimeError(
            f"{path} has memref.alloc ops; on cuda these lower to host malloc "
            "and the kernels cannot access them"
        )


@torch._dynamo.register_backend(name="e0358")
def e0358_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    sample_inputs = tuple(example_inputs)
    device = example_inputs[0].device
    # only queried on cuda: the cpu path must work without a gpu
    chip = gpu_chip(device) if device.type == "cuda" else None
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
    common_pipeline = [
        [
            STABLEHLO_BIN / "stablehlo-opt",
            "--stablehlo-legalize-to-linalg",
            "artifacts/bnc.stablehlo.mlir",
            "-o",
            "artifacts/bnc.linalg.mlir",
        ],
        # lower linalg to affine dialect and fuse
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
            "-o",
            "artifacts/bnc.affine.mlir",
        ],
    ]

    # Fused affine -> llvm ir for cpu path
    cpu_passes = [
        # vectorize (8 x f32 = AVX2) and parallelize the outermost loop only
        # (nested regions cost more)
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.affine.mlir",
            # parallelize first: it can't see through vector.transfer ops
            "--affine-parallelize=max-nested=1",
            "--affine-super-vectorize=virtual-vector-size=8",
            "-o",
            "artifacts/bnc.cpu.affine.mlir",
        ],
        # lower to llvm dialect; parallel loops run on OpenMP threads.
        # canonicalize drops the empty memref.alloca_scope that scf-to-openmp
        # wraps around the body, which scf-to-cf would otherwise break
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.cpu.affine.mlir",
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
            "artifacts/bnc.cpu.llvm.mlir",
        ],
        [
            LLVM_BIN / "mlir-translate",
            "--mlir-to-llvmir",
            "artifacts/bnc.cpu.llvm.mlir",
            "-o",
            "artifacts/bnc.cpu.ll",
        ],
    ]

    # fused affine -> llvm ir for gpu path
    gpu_passes = [
        # map parallel loops to blocks/threads and outline kernels
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.affine.mlir",
            "--affine-parallelize",
            "--lower-affine",
            f"--scf-parallel-loop-tiling=parallel-loop-tile-sizes="
            f"{GPU_TILE_SIZES}",
            "--gpu-map-parallel-loops",
            "--convert-parallel-loops-to-gpu",
            "--gpu-kernel-outlining",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--canonicalize",
            "-o",
            "artifacts/bnc.gpu.mlir",
        ],
        # kernels -> nvvm -> cubin (embedded), host -> llvm + mgpu* calls
        [
            LLVM_BIN / "mlir-opt",
            "artifacts/bnc.gpu.mlir",
            "--llvm-request-c-wrappers",
            "--gpu-lower-to-nvvm-pipeline=" f"cubin-chip={chip} opt-level=3",
            "-o",
            "artifacts/bnc.gpu.llvm.mlir",
        ],
        [
            LLVM_BIN / "mlir-translate",
            "--mlir-to-llvmir",
            "artifacts/bnc.gpu.llvm.mlir",
            "-o",
            "artifacts/bnc.gpu.ll",
        ],
    ]
    for cmd in common_pipeline:
        result = subprocess.run(
            [str(c) for c in cmd], check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)

    if device.type == "cuda":
        check_no_host_allocs("artifacts/bnc.affine.mlir")
        for cmd in gpu_passes:
            result = subprocess.run(
                [str(c) for c in cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(result.stderr)

        ll = "artifacts/bnc.gpu.ll"
        # mgpu* runtime wrappers that the host code calls into
        link_flags = [
            f"-L{LLVM_LIB}",
            "-lmlir_cuda_runtime",
            f"-Wl,-rpath,{LLVM_LIB}",
        ]
    else:
        for cmd in cpu_passes:
            result = subprocess.run(
                [str(c) for c in cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(result.stderr)

        ll = "artifacts/bnc.cpu.ll"
        link_flags = ["-fopenmp"]

    # a fresh path per compilation: dlopen caches libraries by path
    work = Path(tempfile.mkdtemp(prefix="e0358-"))
    subprocess.run(
        [LLVM_BIN / "opt", "-O3", ll, "-o", work / "m.bc"],
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

    # link to shared object file and load
    subprocess.run(
        ["clang", "-shared", work / "m.o", *link_flags, "-o", work / "m.so"],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(work / "m.so"))

    meta = stablehlo_program._bundle.stablehlo_funcs[0].meta
    return make_callable(lib, meta, device)
