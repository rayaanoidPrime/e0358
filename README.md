# e0358

## Versions

| component     | version                                                   |
|---------------|-----------------------------------------------------------|
| llvm-project  | `d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde` (2026-11-03)   |
| stablehlo     | `b26a2e2a6e91b9144f8dc1aafbee74d01016d558` (2026-11-04)   |
| cuda-tile     | `v13.4.3`                                                 |
| Python        | 3.12                                                      |
| torch         | 2.9.0 (CUDA 12.8 build)                                   |
| torch-xla     | 2.9.0                                                     |
| torch-tensorrt| 2.9.0+cu128 (PyTorch's cu128 index), TensorRT 10.13.3 |
| CUDA Toolkit  | 13.2 (RTX 4090 machine; provides `nvcc` and `tileiras`)   |

The MLIR changes are in `patches/` as a single `git format-patch` on top of
the llvm-project commit above. cuda-tile needs no patches.

stablehlo needs one one-line patch,
`patches/stablehlo/0001-Use-tablegen_internal-for-mlir_builder_tblgen.patch`.
The llvm-project commit above itself ("[CMake][NFC] Add llvm_tablegen and use
it instead of tablegen(LLVM ...)") renamed LLVM's generic `tablegen()` CMake
function to `tablegen_internal()`, with the same arguments. stablehlo
b26a2e2a6e91 still calls `tablegen()` once, in
`stablehlo/integrations/cpp/builder/MlirBuilderTblgen.cmake`, so configuring
it fails with `Unknown CMake command "tablegen"`. The patch renames that one
call; no C++ changes are needed.

torch and torch-xla are pinned to 2.9.0 because torch-xla 2.8.0 cannot be
imported on a machine with a GPU: when CUDA is available it skips loading
`_XLAC_cuda_functions`, leaving `c10::cuda::set_device(int8)` unresolved
(torch 2.8 only defines a two-argument version), and it defaults
`PJRT_DEVICE` to `CUDA`, which the PyPI wheel cannot start. 2.9.0 has
neither problem on the GPU and also works on CPU-only machines.

torch-tensorrt on PyPI is built for CUDA 13.0, while torch 2.9.0 on PyPI is
a CUDA 12.8 build, so `pyproject.toml` takes torch-tensorrt `2.9.0+cu128`
from PyTorch's index and TensorRT's CUDA 12 wheels from NVIDIA's index, and
leaves out the generic `tensorrt` metapackage, which would install the
CUDA 13 TensorRT as well.

## Setup

### Prerequisites

- git, CMake >= 3.20, Ninja, a C++17 compiler (clang or gcc), Python 3
- [uv](https://docs.astral.sh/uv/) for the Python environment
- for the GPU path: an NVIDIA driver and a CUDA Toolkit 13.x with `nvcc` and
  `tileiras` (`/usr/local/cuda/bin` is added to `PATH` by `setup.sh` if
  `nvcc` is not already on it)

Without root (e.g. a shared GPU server), CMake, Ninja and uv can be installed
into `~/.local/bin` from PyPI:

```sh
python3 -m pip install --user cmake ninja uv
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH  # also in ~/.bashrc
```

### 1. Toolchain: LLVM/MLIR, stablehlo, cuda-tile

```sh
./setup.sh
```

`setup.sh` clones the three repositories under `$E0358_ROOT` (default
`~/llvm`), applies `patches/*.patch` on a new llvm-project branch
`e0358-assign-2` and `patches/stablehlo/*.patch` on a stablehlo branch of the
same name, builds LLVM/MLIR (Release with assertions, LLD, MLIR, X86 and NVPTX,
and the MLIR CUDA runner when `nvcc` is found), stablehlo, cuda-tile, and
`tools/e0358-tile` (against the MLIR and cuda-tile builds), and writes `$E0358_ROOT/e0358-env.sh` with the paths the backend reads. Rerunning
it keeps existing branches instead of reapplying the patches.

Optional environment for `setup.sh`: `E0358_ROOT`, `E0358_JOBS` (compile
jobs, default `nproc`), `E0358_LINK_JOBS` (parallel links, default 2) and
`E0358_CUDA` (`ON`/`OFF` for the CUDA runner).

If a clone or fetch from GitHub drops partway (`curl 56 GnuTLS recv error`,
`early EOF`), fetching over HTTP/1.1 usually gets through; `setup.sh` skips
the fetch once the commit is present, so rerun it afterwards:

```sh
git -C ~/llvm/llvm-project -c http.version=HTTP/1.1 fetch --depth 1 origin \
  d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde
./setup.sh
```

### 2. Python environment

```sh
uv sync   # Python 3.12, torch, torch-xla and torch-tensorrt 2.9.0 into .venv
```

The environment takes about 7 GB. On a machine with little space where the
project lives, put uv's download cache on another disk:

```sh
UV_CACHE_DIR=/tmp/uv-cache-$USER UV_LINK_MODE=copy uv sync
```

### 3. Environment variables

```sh
source ~/llvm/e0358-env.sh
```

This sets `E0358_LLVM_BIN`, `E0358_LLVM_LIB` and `E0358_STABLEHLO_BIN`.
Depending on the machine, also set:

- `LD_LIBRARY_PATH`: uv's own Python keeps `libpython3.12.so` in its install
  directory, where torch-xla's `_XLAC` extension does not find it
  (`ImportError: libpython3.12.so.1.0: cannot open shared object file`). Not
  needed with a system Python 3.12.

  ```sh
  export LD_LIBRARY_PATH=$(.venv/bin/python -c \
    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
  ```

- `E0358_CC`: the compiler driver that links the compiled kernels into a
  shared library (default `clang`). Set `E0358_CC=gcc` on machines without
  clang; this works for the GPU path, while the CPU path needs clang for the
  OpenMP runtime its `-fopenmp` provides.
- `TMPDIR`: on a shared machine, set a per-user temp directory
  (`mkdir -p /tmp/$USER && export TMPDIR=/tmp/$USER`). Torch-TensorRT keeps
  its timing cache in `$TMPDIR/torch_tensorrt_engine_cache/`, and the
  TensorRT baseline fails with `PermissionError` when another user already
  owns `/tmp/torch_tensorrt_engine_cache`.
- `E0358_GPU_CHIP`: the `sm_XY` target (e.g. `sm_89` for the RTX 4090), to
  override the GPU detected from the device.

Appending these to `~/llvm/e0358-env.sh` keeps everything in one place.

### 4. Running

```sh
uv run python qkv-projection.py --backend e0358        # Asst-2, all sizes
uv run python broadcast-n-consumers.py --backend e0358 # Asst-1 test cases
uv run python truncate-n-consumers.py --backend e0358
```

The Asst-1 scripts use `common_asst_1.py`, the Asst-1 version of the test
utilities; `common.py` is the Asst-2 harness.

### CUDA Tile bytecode version

cuda-tile v13.4.3 can write bytecode up to version 13.4, which `tileiras`
from an older CUDA Toolkit rejects with a misleading
`error: invalid GPU architecture: 89` (for every target, not just `sm_89`).
The bytecode version must not be newer than the CUDA Toolkit's (13.2 on the
RTX 4090 machine). `e0358-tile` writes version 13.1, cuda-tile's default for
its bytecode writer, which any CUDA 13 `tileiras` accepts; with
`cuda-tile-translate`, pass `--bytecode-version=13.2` or lower.

## AI-assisted parts

The following parts were written with the help of an AI coding agent
(Claude Code). The optimization component was not AI-assisted.

### Toolchain setup script and gpu chip override

Files: `setup.sh`, `GPU_CHIP` / `gpu_chip()` in `src/e0358/backend.py`.

Prompt:

> Since my laptop does not have a gpu, i will check ht ecpu path fusion on
> this laptop. I will be using a separate remote machine for my gpu builds, so
> write a setup script that build llvm llvmorg=23.1.3 release and with my patch
> on top and stablehlo commit 965c428908be23f05e80e3c8e7f0b1a806ac3751 dated
> 2026-08-04 This way i have a single setup script which will build the project
> with my commit on top as a patch. Make sure you set it up so that we checkout
> a new branch e0358-assign-1 where we apply the patch and then build llvm and
> stablehlo

### Asst-2 toolchain update

Files: `setup.sh`, `patches/`, the Versions and Building sections above.

Prompts:

> read through the llvm , stable hlo setup instructions in the asst-2 folder
> readme and make sure my current llvm , stablehlo, mlir are the same as
> mentioned in the assignment 2 instructions. If they are not then make them
> the same, rebuild etc

> yes clone and build cuda-tile too and update the old stuff for this new
> assignment. I will be building this anyway in a new virtual machine with a
> gpu so that step should be simple and hassleffree

### Assignment reading roadmap

Files: none. The agent replied in chat with a reading list (asst-2.md,
the harness, the cuda-tile README/spec/tests, the existing backend),
toolchain facts (no upstream lowering to `cuda_tile` exists; bytecode is
compiled with `tileiras`), a milestone order, and a list of open design
questions left unanswered.

Prompt:

> in the meantime read hte assignment instructions and draft a plan in the
> chat of how to solve this, which part to read etc, a roadmap baically

### GPU server setup and Python environment

Files: `pyproject.toml` and `uv.lock` (torch/torch-xla 2.9.0,
torch-tensorrt), `E0358_CC` in `src/e0358/backend.py`, `common_asst_1.py`
and the `common` import in the two Asst-1 scripts, and the Versions and Setup
sections above. The agent also built the toolchain and the Python
environment on the RTX 4090 server over SSH.

Prompts, as typed (the server's address and login details are redacted):

> ssh into this server and check if the environment is built or not ->
> Created account on RTX 4090 server [IP, username and password redacted]

> Yes, build it

> build it and why cant we just git clone the repo there? wouldnt that be
> cleaner?

> i just pushed into branch assign-2 of the upstream repo

> run the bashrc line and why do we need python 3.12 again?

> lower it

> why cant i use uv to manage my envs

> i cant use uv with 3.10? 3.10 is the better choice right? sinec the torch
> builds are already there for it

> so the server does not have any project dependencies installed yet? like
> torch or xla ?

> yeah okay use uv with python 3.12 or whatever is easier and simpler

> fix 1 is an issue because we are using uv 2. fix 2 is an issue why? i
> tested assgn 2 in an rtx 6000 pro gpu machine and it ran there i think? do
> we need to update the versions for torch_xla / torch or something? are we
> using wrong versions

> okay commit the new version pins and push for all branches and whar other
> change is left to be addressed?

> update on the server and add the tensor rt dep, update the reademe with the
> setup instructions and the bytecode thing, the versions and the ai prompts,
> for the assignment 1 common, you can create a new common file with the name
> common_asst_1 or something like that.

Short status checks during the builds ("the build still running?",
"status", "statu") and one-word confirmations ("yes") are not listed
separately.
