import re

import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

# MLIR's python bindings read the affine loops; cuda-tile's build the kernels
from mlir import ir as mlir
import cuda_tile._mlir.dialects.cuda_tile_ops as ct
from cuda_tile._mlir import ir as tile_ir
from cuda_tile._mlir._mlir_libs._cuda_tile import register_dialect
from cuda_tile._mlir._mlir_libs._cuda_tile import writeBytecode
import ctypes
import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# llvm and stablehlo build locations
STABLEHLO_BIN = Path(os.environ["E0358_STABLEHLO_BIN"]).expanduser()
LLVM_BIN = Path(os.environ["E0358_LLVM_BIN"]).expanduser()
# holds libmlir_cuda_runtime.so (needs -DMLIR_ENABLE_CUDA_RUNNER=ON)
LLVM_LIB = Path(os.environ["E0358_LLVM_LIB"]).expanduser()
# compiler driver used to link the kernel object into a shared library; the
# cpu path needs clang (its -fopenmp provides the __kmpc_* runtime)
CC = os.environ.get("E0358_CC", "clang")

# threads per GPU block after coalescing each loop nest to one dimension
GPU_TILE_SIZES = os.environ.get("E0358_GPU_TILE_SIZES", "256")

# nvidia target for the gpu path (e.g., sm_86)
GPU_CHIP = os.environ.get("E0358_GPU_CHIP")

# prefix of the IR files written at each stage: artifacts/<script>.<stage>
ARTIFACTS = f"artifacts/{Path(sys.argv[0]).stem or 'e0358'}"

# CUDA Tile IR element types of the memref element types
TILE_TYPES = {"f32": ct.Float32, "f16": ct.Float16, "bf16": ct.BFloat16}


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


def make_callable(run, meta, device):
    """Returns the compiled function: run(tensors) computes it on the MLIR
    function's arguments, its inputs followed by its outputs."""
    # MLIR argument k is fed by user input input_locations[k].position
    order = [loc.position for loc in meta.input_locations]

    def callable_fn(*args):
        # outputs are allocated here and passed as out-params
        outs = [
            torch.empty(s.shape, dtype=getattr(torch, s.dtype), device=device)
            for s in meta.output_signature
        ]
        # we pass identity-layout bufferization
        # so we need contiguous row-major tensors
        run([args[i].contiguous() for i in order] + outs)
        return tuple(outs)

    return callable_fn


def run_library(lib, device):
    """run() for the MLIR function compiled into shared library 'lib'."""
    func = lib._mlir_ciface_main
    func.restype = None

    def run(tensors):
        if device.type == "cuda":
            # the mlir cuda runtime launches on its own non-blocking stream,
            # so wait for torch's pending work
            # the launch itself synchronizes before returning
            torch.cuda.synchronize(device)
        # on cuda the descriptors carry device pointers, which the kernels
        # use directly
        func(*(ctypes.byref(memref_descriptor(t)) for t in tensors))

    return run


@functools.cache
def driver():
    """The CUDA driver API, loaded on first use: cpu-only machines lack it."""
    return ctypes.CDLL("libcuda.so.1")


def check(error):
    if error:
        raise RuntimeError(f"CUDA driver error {error}")


class Kernels:
    """The kernels of a cubin, launched in order; calling it runs them on the
    function's arguments ('buffers'). 'launches' lists each kernel's name,
    grid, and parameters as function argument numbers."""

    def __init__(self, cubin, launches):
        self.module = ctypes.c_void_p()
        module = ctypes.byref(self.module)
        check(driver().cuModuleLoadData(module, cubin.read_bytes()))
        self.launches = []
        for name, grid, args in launches:
            kernel = ctypes.c_void_p()
            name = name.encode()
            check(
                driver().cuModuleGetFunction(
                    ctypes.byref(kernel), self.module, name
                )
            )
            self.launches.append((kernel, (*grid, 1), args))

    def __call__(self, buffers):
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for kernel, grid, args in self.launches:
            pointers = [ctypes.c_uint64(buffers[a].data_ptr()) for a in args]
            params = (ctypes.c_void_p * len(args))(
                *map(ctypes.addressof, pointers)
            )
            # cuda tile kernels take the tile grid and a (1, 1, 1) block
            check(
                driver().cuLaunchKernel(
                    kernel, *grid, 1, 1, 1, 0, stream, params, None
                )
            )


# Lowering of affine matmul nests to CUDA Tile IR:
#
#   for i, j: c[i, j] = 0
#   for i, j, k: c[i, j] += a[i, k] * b[k, j]
#
# becomes one kernel per group of matmuls, each block computing one tile of c
# and accumulating over k in f32. With sibling fusion, matmuls that read the
# same a (same shapes, none writing what another reads) form one group, whose
# kernel loads each tile of a once for all of them.


def body(op):
    """The operations in the single block of 'op'."""
    return list(op.regions[0].blocks[0].operations)


def nest(loop):
    """The loops of the perfect nest rooted at 'loop', outermost first."""
    loops = [loop]
    while len(ops := body(loops[-1])) == 2 and ops[0].name == "affine.for":
        loops.append(ops[0])
    return loops


def trip_count(loop):
    """Trip count of a loop from 0 with step 1, or 0 for any other loop."""
    end = loop.upperBoundMap.value.results[0].maybe_downcast()
    if loop.lowerBoundMap.value != mlir.AffineMap.get_constant(0):
        return 0
    from_zero = loop.step.value == 1 and isinstance(
        end, mlir.AffineConstantExpr
    )
    return end.value if from_zero else 0


def access(op, ivs):
    """Number of the function argument that load/store 'op' accesses at
    exactly 'ivs', or None."""
    if op is None or op.name not in ("affine.load", "affine.store"):
        return None
    if op.map.value != mlir.AffineMap.get_identity(len(ivs)):
        return None
    memref = op.memref.maybe_downcast()
    argument = (
        isinstance(memref, mlir.BlockArgument) and list(op.indices) == ivs
    )
    return memref.arg_number if argument else None


def producer(value):
    """The operation defining 'value', or None for a block argument."""
    value = value.maybe_downcast()
    return value.owner if isinstance(value, mlir.OpResult) else None


def match_zero_fill(loop):
    """c for `for i, j: c[i, j] = 0`, or None."""
    loops = nest(loop)
    store = body(loops[-1])[0]
    if len(loops) != 2 or not all(map(trip_count, loops)):
        return None
    zero = producer(store.value) if store.name == "affine.store" else None
    if zero is None or zero.name != "arith.constant" or zero.value.value != 0:
        return None
    return access(store, [loop.induction_variable for loop in loops])


def match_matmul(loop):
    """The matmul `for i, j, k: c[i, j] += a[i, k] * b[k, j]`, or None: a, b,
    c as function argument numbers, the sizes m, n, k, and element type t."""
    loops = nest(loop)
    ops = body(loops[-1])
    if len(loops) != 3 or len(ops) != 7 or ops[-2].name != "affine.store":
        return None
    i, j, k = (loop.induction_variable for loop in loops)
    store = ops[-2]
    add = producer(store.value)
    if add is None or add.name != "arith.addf":
        return None
    # c[i, j] + a * b, in either order
    acc, mul = map(producer, add.operands)
    if mul is None or mul.name != "arith.mulf":
        acc, mul = mul, acc
    if mul is None or mul.name != "arith.mulf":
        return None
    lhs, rhs = map(producer, mul.operands)
    mm = dict(
        a=access(lhs, [i, k]),
        b=access(rhs, [k, j]),
        c=access(store, [i, j]),
        m=trip_count(loops[0]),
        n=trip_count(loops[1]),
        k=trip_count(loops[2]),
        t=str(mlir.MemRefType(store.memref.type).element_type),
    )
    if None in (mm["a"], mm["b"], mm["c"]) or 0 in (mm["m"], mm["n"], mm["k"]):
        return None
    return mm if access(acc, [i, j]) == mm["c"] else None


def matmuls(path):
    """The matmuls of the function in the affine file 'path'; raises
    ValueError if it holds anything else."""
    found, zeroed = [], set()
    with mlir.Context():
        affine = mlir.Module.parse(Path(path).read_text())
        for op in body(affine.body.operations[0]):
            if (
                op.name == "affine.for"
                and (c := match_zero_fill(op)) is not None
            ):
                zeroed.add(c)
            elif op.name == "affine.for" and (mm := match_matmul(op)):
                if mm["c"] not in zeroed:
                    raise ValueError("matmul output is not zero-initialized")
                found.append(mm)
            elif op.name not in ("arith.constant", "func.return"):
                raise ValueError(f"not a matmul nest: {op.name}")
    return found


def fusable(x, y):
    """Whether matmuls 'x' and 'y' can share one kernel: the same a, shapes
    and type, and neither writes what the other reads."""
    same = all(x[f] == y[f] for f in "amnkt")
    return (
        same
        and x["c"] not in (y["a"], y["b"])
        and y["c"] not in (x["a"], x["b"])
    )


def tile_sizes(mm, count):
    """Tile sizes (m x n x k) for a group of 'count' matmuls of mm's shape,
    picked from measurements on an RTX 4090: 16-deep k tiles, enough blocks
    for its 128 SMs where the matrix allows it, and, in fused groups, where
    each matmul keeps its own accumulator, narrower tiles."""
    blocks = lambda tm, tn: (mm["m"] // tm) * (mm["n"] // tn)
    f32 = mm["t"] == "f32"
    tm = 32 if f32 and count == 1 else 64
    if blocks(tm, 32) < 128:
        tm = 32
    tn = 128 if count == 1 else 32 if f32 else 64
    while tn > 32 and (mm["n"] % tn or blocks(tm, tn) < 128):
        tn //= 2
    if mm["m"] % tm or mm["n"] % tn or mm["k"] % 16:
        raise ValueError(f"matmul {mm} is not made of {tm}x{tn}x16 tiles")
    return tm, tn, 16


def build_kernel(name, group, tm, tn, tk):
    """Builds the kernel computing 'group' as @name, with tm x tn tiles of c
    and tk-deep steps over k; its parameters are a, b0, c0, b1, c1, ..."""
    mm, t = group[0], TILE_TYPES[group[0]["t"]]
    pointer = ct.make_tile_type(ct.PointerType.get(t.mlir_type))
    location = tile_ir.Location.unknown()
    with ct.EntryContext(name, location, [pointer] * (1 + 2 * len(group))) as p:
        i, j = ct.get_tile_block_id()[:2]

        # the row-major matrix at 'ptr', cut into 'shape' tiles
        def tiles(ptr, rows, cols, shape):
            view = ct.make_tensor_view(ptr, t, [rows, cols], [cols, 1])
            return ct.make_partition_view(view, shape)

        a = tiles(p[0], mm["m"], mm["k"], [tm, tk])
        bs = [tiles(b, mm["k"], mm["n"], [tk, tn]) for b in p[1::2]]
        cs = [tiles(c, mm["m"], mm["n"], [tm, tn]) for c in p[2::2]]

        # for k: acc_n += a[i, k] * b_n[k, j], one f32 accumulator per matmul
        def step(k, *accs):
            x = ct.load_view_tko(a, [i, k])
            sums = [
                ct.mma(x, ct.load_view_tko(b, [k, j]), acc)
                for b, acc in zip(bs, accs)
            ]
            ct._continue(sums)

        zero = ct.constant(
            0.0, tile_type=ct.make_tile_type(ct.Float32, [tm, tn])
        )
        accs = ct.for_loop(step, 0, mm["k"] // tk, 1, [zero] * len(group))
        # one accumulator comes back as a tile, several as a tuple
        accs = [accs] if isinstance(accs, ct.Tile) else accs

        # c_n[i, j] = acc_n, in the element type
        for acc, c in zip(accs, cs):
            ct.store_view_tko(
                acc if t is ct.Float32 else ct.ftof(t, acc), c, [i, j]
            )


def lower_to_tile(path, fusion, bytecode):
    """Writes the CUDA Tile IR kernels for the matmuls of the affine file
    'path' to the file 'bytecode', siblings fused if 'fusion'; returns the
    IR and the kernels' launches (name, grid, parameters)."""
    groups = []
    for mm in matmuls(path):
        fits = lambda group: fusion and all(fusable(x, mm) for x in group)
        group = next(filter(fits, groups), None)
        group.append(mm) if group else groups.append([mm])

    launches = []
    with tile_ir.Context() as context, tile_ir.Location.unknown():
        register_dialect(context, load=True)
        module = ct.ModuleOp("kernels")
        with tile_ir.InsertionPoint(module.body):
            for n, group in enumerate(groups):
                mm = group[0]
                tm, tn, tk = tile_sizes(mm, len(group))
                build_kernel(f"k{n}", group, tm, tn, tk)
                args = [mm["a"]] + [x for g in group for x in (g["b"], g["c"])]
                launches.append((f"k{n}", (mm["m"] // tm, mm["n"] // tn), args))
        # the helpers give each op its Python source line as a name location,
        # which the bytecode writer rejects; parsed back from its text, the
        # (verified) module has plain file locations instead
        module = tile_ir.Module.parse(str(module))
        with open(bytecode, "wb") as f:
            writeBytecode(f, module.operation)
        return str(module), launches


def move_allocs_to_device(path):
    # gpu.alloc/gpu.dealloc lower to the CUDA runtime instead of host malloc.
    text = Path(path).read_text()
    allocs = []

    def repl(match):
        allocs.append((match.group(1), match.group(2)))
        return f"%{match.group(1)} = gpu.alloc() : {match.group(2)}"

    text = re.sub(
        r"%(\w+) = memref\.alloc\(\)(?: \{alignment = \d+ : i64\})? : (memref<[^>]*>)",
        repl,
        text,
    )
    deallocs = "".join(f"    gpu.dealloc %{n} : {t}\n" for n, t in allocs)
    text = text.replace("\n    return\n", "\n" + deallocs + "    return\n")
    Path(path).write_text(text)


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
    with open(f"{ARTIFACTS}.stablehlo.mlir", "w") as f:
        f.write(stable_hlo_text)

    fusion_mode = os.environ.get("E0358_FUSION_MODE", "greedy")
    fusion_pass = (
        "--canonicalize"
        if fusion_mode == "none"
        else f"--affine-loop-fusion=mode={fusion_mode}"
    )
    if fusion_mode == "producer":
        tolerance = os.environ.get("E0358_COMPUTE_TOLERANCE", "0.25")
        fusion_pass += f" compute-tolerance={tolerance}"

    # stablehlo -> mlir -> llvm lowering pipeline to <ARTIFACTS>.*.ll; each
    # entry is one tool invocation, run in order
    common_pipeline = [
        [
            STABLEHLO_BIN / "stablehlo-opt",
            "--stablehlo-legalize-to-linalg",
            f"{ARTIFACTS}.stablehlo.mlir",
            "-o",
            f"{ARTIFACTS}.linalg.mlir",
        ],
        # lower linalg to affine loops
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.linalg.mlir",
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map",
            "--buffer-results-to-out-params="
            "hoist-static-allocs=true modify-public-functions=true",
            "--convert-linalg-to-affine-loops",
            "--canonicalize",
            "-o",
            f"{ARTIFACTS}.loops.mlir",
        ],
        # fuse
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.loops.mlir",
            fusion_pass,
            "--affine-scalrep",
            "-o",
            f"{ARTIFACTS}.affine.mlir",
        ],
    ]

    # Fused affine -> llvm ir for cpu path
    cpu_passes = [
        # vectorize (8 x f32 = AVX2) and parallelize the outermost loop only
        # (nested regions cost more)
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.affine.mlir",
            # parallelize first: it can't see through vector.transfer ops
            "--affine-parallelize=max-nested=1",
            "--affine-super-vectorize=virtual-vector-size=8",
            "-o",
            f"{ARTIFACTS}.cpu.affine.mlir",
        ],
        # lower to llvm dialect; parallel loops run on OpenMP threads.
        # canonicalize drops the empty memref.alloca_scope that scf-to-openmp
        # wraps around the body, which scf-to-cf would otherwise break
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.cpu.affine.mlir",
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
            f"{ARTIFACTS}.cpu.llvm.mlir",
        ],
        [
            LLVM_BIN / "mlir-translate",
            "--mlir-to-llvmir",
            f"{ARTIFACTS}.cpu.llvm.mlir",
            "-o",
            f"{ARTIFACTS}.cpu.ll",
        ],
    ]

    # fused affine -> llvm ir for gpu path
    gpu_passes = [
        # map parallel loops to blocks/threads and outline kernels
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.affine.mlir",
            "--affine-loop-coalescing",
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
            f"{ARTIFACTS}.gpu.mlir",
        ],
        # kernels -> nvvm -> cubin (embedded), host -> llvm + mgpu* calls
        [
            LLVM_BIN / "mlir-opt",
            f"{ARTIFACTS}.gpu.mlir",
            "--llvm-request-c-wrappers",
            "--gpu-lower-to-nvvm-pipeline=" f"cubin-chip={chip} opt-level=3",
            "-o",
            f"{ARTIFACTS}.gpu.llvm.mlir",
        ],
        [
            LLVM_BIN / "mlir-translate",
            "--mlir-to-llvmir",
            f"{ARTIFACTS}.gpu.llvm.mlir",
            "-o",
            f"{ARTIFACTS}.gpu.ll",
        ],
    ]
    for cmd in common_pipeline:
        result = subprocess.run(
            [str(c) for c in cmd], check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)

    print(Path(f"{ARTIFACTS}.affine.mlir").read_text(), "\n...")
    meta = stablehlo_program._bundle.stablehlo_funcs[0].meta

    if device.type == "cuda":
        try:
            return make_callable(compile_tile(chip), meta, device)
        except ValueError as error:
            # not made of matmuls: take the affine gpu path below
            print(error)

    if device.type == "cuda":
        move_allocs_to_device(f"{ARTIFACTS}.affine.mlir")
        check_no_host_allocs(f"{ARTIFACTS}.affine.mlir")
        for cmd in gpu_passes:
            result = subprocess.run(
                [str(c) for c in cmd],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print(result.stderr)

        ll = f"{ARTIFACTS}.gpu.ll"
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

        ll = f"{ARTIFACTS}.cpu.ll"
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
        [CC, "-shared", work / "m.o", *link_flags, "-o", work / "m.so"],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(work / "m.so"))
    return make_callable(run_library(lib, device), meta, device)


def compile_tile(chip):
    """CUDA Tile IR kernels for the matmuls of <ARTIFACTS>.loops.mlir,
    siblings fused when E0358_SIBLING_FUSION=1."""
    fusion = os.environ.get("E0358_SIBLING_FUSION") == "1"
    work = Path(tempfile.mkdtemp(prefix="e0358-"))
    bytecode, cubin = work / "kernels.tilebc", work / "kernels.cubin"
    ir, launches = lower_to_tile(f"{ARTIFACTS}.loops.mlir", fusion, bytecode)
    Path(f"{ARTIFACTS}.tile.mlir").write_text(ir)
    print(ir, "\n...")
    tileiras = ["tileiras", f"--gpu-name={chip}", bytecode, "-o", cubin]
    subprocess.run(tileiras, check=True, capture_output=True)
    return Kernels(cubin, launches)
