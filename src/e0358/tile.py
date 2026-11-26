"""Compiles and launches the CUDA Tile IR kernels of a matmul function.

tools/e0358-tile builds the kernels for the function's matmul nests and
prints the line each kernel is launched from:

  // launch @k0 grid=[8, 1] args=[1, 0, 4]

where args are the function argument numbers in kernel parameter order.
"""

import ctypes
import functools
import re
import subprocess
import tempfile
from pathlib import Path

import torch

TOOL = Path(__file__).parents[2] / "tools/e0358-tile/build/e0358-tile"
LAUNCH = re.compile(r"// launch @(\w+) grid=\[(\d+), (\d+)\] args=\[(.*)\]")


@functools.cache
def driver():
    """The CUDA driver API, loaded on first use: cpu-only machines lack it."""
    return ctypes.CDLL("libcuda.so.1")


def check(error):
    if error:
        raise RuntimeError(f"CUDA driver error {error}")


def compile(affine, chip, fusion):
    """Compiles the matmuls of the affine file 'affine' for 'chip', siblings
    fused if 'fusion'; returns the kernels and their CUDA Tile IR."""
    work = Path(tempfile.mkdtemp(prefix="e0358-"))
    run = lambda *cmd: subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout
    fuse = ["--sibling-fusion"] if fusion else []
    ir = run(TOOL, *fuse, affine, "-o", work / "kernels.tilebc")
    run(
        "tileiras",
        f"--gpu-name={chip}",
        work / "kernels.tilebc",
        "-o",
        work / "kernels.cubin",
    )
    return Kernels(work / "kernels.cubin", ir), ir


class Kernels:
    """The kernels of a cubin, launched in the order of their launch lines."""

    def __init__(self, cubin, ir):
        self.module = ctypes.c_void_p()
        module = ctypes.byref(self.module)
        check(driver().cuModuleLoadData(module, cubin.read_bytes()))
        self.launches = []
        for name, x, y, args in LAUNCH.findall(ir):
            kernel = ctypes.c_void_p()
            check(
                driver().cuModuleGetFunction(
                    ctypes.byref(kernel), self.module, name.encode()
                )
            )
            args = [int(a) for a in args.split(",")]
            self.launches.append((kernel, (int(x), int(y), 1), args))

    def __call__(self, buffers):
        """Runs the kernels on 'buffers', the function's arguments in order."""
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
