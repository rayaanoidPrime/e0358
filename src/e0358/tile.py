"""Compiles and launches the CUDA Tile IR kernels of a matmul function.

mlir-translate --affine-to-cuda-tile turns the function's matmul nests into
kernels and precedes each with the line it is launched from:

  // launch @k0 grid=[8, 1] args=[1, 0, 4]

where args are the function argument numbers in kernel parameter order.
"""

import ctypes
import functools
import re
import subprocess

import torch

LAUNCH = re.compile(r"// launch @(\w+) grid=\[(\d+), (\d+)\] args=\[(.*)\]")


@functools.cache
def driver():
    """The CUDA driver API, loaded on first use: cpu-only machines lack it."""
    return ctypes.CDLL("libcuda.so.1")


def check(error):
    if error:
        raise RuntimeError(f"CUDA driver error {error}")


def compile_kernels(mlir, work, cuda_tile_bin, chip):
    """Compiles the CUDA Tile IR file 'mlir' to a cubin in 'work'."""
    # tileiras rejects bytecode newer than its CUDA toolkit (13.2 here)
    run = lambda *cmd: subprocess.run(cmd, check=True, capture_output=True)
    run(
        cuda_tile_bin / "cuda-tile-translate",
        "--mlir-to-cudatilebc",
        "--no-implicit-module",
        "--bytecode-version=13.2",
        mlir,
        "-o",
        work / "kernels.tilebc",
    )
    run(
        "tileiras",
        f"--gpu-name={chip}",
        work / "kernels.tilebc",
        "-o",
        work / "kernels.cubin",
    )
    return work / "kernels.cubin"


class Kernels:
    """The kernels of a cubin, launched in the order of their launch lines."""

    def __init__(self, cubin, mlir):
        self.module = ctypes.c_void_p()
        check(
            driver().cuModuleLoadData(
                ctypes.byref(self.module), cubin.read_bytes()
            )
        )
        self.launches = []
        for name, x, y, args in LAUNCH.findall(mlir.read_text()):
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
            params = (ctypes.c_void_p * len(pointers))(
                *map(ctypes.addressof, pointers)
            )
            # cuda tile kernels take the tile grid and a (1, 1, 1) block
            check(
                driver().cuLaunchKernel(
                    kernel, *grid, 1, 1, 1, 0, stream, params, None
                )
            )
