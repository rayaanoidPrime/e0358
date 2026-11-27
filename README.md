# e0358

## Versions

| component     | version                                                   |
|---------------|-----------------------------------------------------------|
| llvm-project  | `d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde` (2026-11-03)   |
| stablehlo     | `b26a2e2a6e91b9144f8dc1aafbee74d01016d558` (2026-11-04)   |
| cuda-tile     | `v13.4.3`                                                 |
| Python        | the system's, 3.10 or newer (3.10.12 on the RTX 4090)     |
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
  `tileiras` (`setup.sh` also looks in `/usr/local/cuda/bin`)

Without root (e.g. a shared GPU server), CMake, Ninja and uv can be installed
into `~/.local/bin` from PyPI:

```sh
python3 -m pip install --user cmake ninja uv
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH  # also in ~/.bashrc
```

### 1. Python environment

```sh
uv sync   # torch, torch-xla, torch-tensorrt 2.9.0 into .venv
```

uv uses the machine's own Python (`python-preference = "only-system"`), 3.10
or newer, with its development headers: the MLIR and cuda-tile Python
bindings are built for it, and uv's downloaded Pythons keep `libpython`
where torch-xla does not find it. The environment takes about 12 GB; on a
machine with little space where the project lives, put uv's download cache on
another disk:

```sh
UV_CACHE_DIR=/tmp/uv-cache-$USER UV_LINK_MODE=copy uv sync
```

### 2. Toolchain: LLVM/MLIR, stablehlo, cuda-tile

```sh
./setup.sh
```

`setup.sh` clones the three repositories under `$E0358_ROOT` (default
`~/llvm`), applies `patches/*.patch` on a new llvm-project branch
`e0358-assign-2` and `patches/stablehlo/*.patch` on a stablehlo branch of the
same name, builds LLVM/MLIR (Release with assertions, LLD, MLIR, X86 and NVPTX,
and the MLIR CUDA runner when `nvcc` is found), stablehlo and cuda-tile, and
writes `$E0358_ROOT/e0358-env.sh` with the paths the backend reads. MLIR and
cuda-tile are built with their Python bindings, for the Python of step 1
(`.venv/bin/python`, or `E0358_PYTHON`). Rerunning it
keeps existing branches instead of reapplying the patches.

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

### 3. Environment variables

```sh
source ~/llvm/e0358-env.sh
```

This sets `E0358_LLVM_BIN`, `E0358_LLVM_LIB`, `E0358_STABLEHLO_BIN` and
`E0358_CUDA_TILE_BIN`, puts the MLIR and cuda-tile Python bindings on
`PYTHONPATH`, and puts the CUDA Toolkit's `bin` (with `tileiras`) on `PATH`.
Depending on the machine, also set:

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
uv run python qkv-projection.py --backend e0358   # all sizes and variants
```

`qkv-projection.py` and `common.py` (the course harness) go next to
`e0358.py`, which they import. The harness sets `E0358_SIBLING_FUSION` (`0` or
`1`) before each compile; the backend reads it on every compile.

### CUDA Tile bytecode version

cuda-tile v13.4.3 can write bytecode up to version 13.4, which `tileiras`
from an older CUDA Toolkit rejects with a misleading
`error: invalid GPU architecture: 89` (for every target, not just `sm_89`).
The bytecode version must not be newer than the CUDA Toolkit's (13.2 on the
RTX 4090 machine). The bytecode writer in cuda-tile's Python bindings
always writes 13.4, so the backend prints the kernels and writes their
bytecode with `cuda-tile-translate --bytecode-version=13.2`.

## Design

### Pipeline

On CUDA, the backend in `e0358.py` compiles a graph in these steps:

1. `torch.export` and torch-xla export it to StableHLO.
2. `stablehlo-opt --stablehlo-legalize-to-linalg`, then `mlir-opt` bufferizes
   (results become out-parameters) and lowers linalg to affine loops
   (`artifacts/qkv-projection.loops.mlir`).
3. `lower_to_tile` reads the affine matmuls with MLIR's Python bindings and
   builds CUDA Tile IR kernels for them with cuda-tile's
   (`artifacts/qkv-projection.tile.mlir`).
4. `cuda-tile-translate` writes them as bytecode, and `tileiras` compiles
   that to a cubin for the GPU (`sm_89`).
5. `Kernels` loads the cubin with the CUDA driver API and launches the
   kernels on torch's current stream.

The two test scripts of the previous assignment (`broadcast-n-consumers`
and `truncate-n-consumers`) take the affine path instead: affine loop fusion
(`patches/`), then the GPU dialect or the CPU. The CUDA Tile IR path needs a
GPU.

### Lowering to CUDA Tile IR

`match_zero_fill` and `match_matmul` match the two loop nests that each
matmul lowers to, through the affine ops' Python API:

    for i, j: c[i, j] = 0
    for i, j, k: c[i, j] += a[i, k] * b[k, j]

with constant bounds and `a`, `b`, `c` function arguments accessed exactly at
those indices. Any other op in the function is an error, so nothing is
miscompiled silently.

Each matmul group becomes one `cuda_tile.entry` kernel, built by
`build_kernel` with cuda-tile's Python op builders and checked by its
verifier. The grid has one block
per `tm x tn` tile of `c`. Block `(i, j)` loops over `k` in steps of `tk`:
it loads tile `(i, k)` of `a` and tile `(k, j)` of `b` through partition
views and accumulates their `mmaf` product in an f32 tile. It then stores
the result to tile `(i, j)` of `c`, converted with `ftof` for f16 and bf16.

Tile sizes come from sweeps on the RTX 4090:

- `tk` = 16.
- `tm` = 32 for unfused f32, otherwise 64. It drops to 32 when that leaves
  fewer blocks than the 4090's 128 SMs.
- `tn` = the widest of 128, 64 and 32 that still gives every SM a block. In
  fused groups `tn` is at most 32 for f32 and 64 for 16-bit types, since each
  matmul keeps its own accumulator.
- If there are still fewer blocks than SMs, `tm` and then `tn` halve down to
  16. This gives bert-base, whose 512x64 outputs would otherwise make only 32
  blocks, 128 blocks of 16x16, and made it 1.2x to 1.8x faster.

Before that last step, the rule was within 4% of the best tile size found in
the sweeps for every size and dtype. Load latency hints (1 to 10) and
grouped ("swizzled") block order were also tried: neither helped across
sizes and dtypes, so neither is used.

### Sibling fusion

With `E0358_SIBLING_FUSION=1`, `lower_to_tile` groups matmuls into one
kernel when they:

- read the same `a` buffer,
- have the same `m`, `n` and `k`, and
- do not write anything another member reads.

The Q/K/V projections share `x` and satisfy all three, so they become one
kernel with three accumulators. Each `k` step loads the tile of `x` once and
uses it for all three `mmaf`s, instead of three kernels each loading it.

Fusion is visible in the IR (one `entry @k0` with three `mmaf`s in
`artifacts/qkv-projection.tile.mlir`) and in nsys. At bert-base, one call
launches `k0`, `k1` and `k2` without fusion, and only `k0` with fusion.

## Results

Measured on an RTX 4090 (driver 595.58, CUDA 13.2, Python 3.10.12), each run
started with no other process on the GPU, by:

```sh
python qkv-projection.py --backend e0358 --reference eager --device cuda \
  --dtype {float32,float16,bfloat16} --fusion both --warmup 10 --iters 50 \
  --stat {median,mean}
```

Each cell is `median / mean` of 50 timed iterations after 10 warmup
iterations, in ms; the fastest median and the fastest mean of each column are
in bold (ties are all bold). All e0358 outputs pass validation against eager
in every dtype; in float32 the maximum absolute error is about 1e-6 to 8e-6,
so `mmaf` does not round inputs to TF32.

**float32** (the default)

| variant           | bert-base             | gpt2-medium           | llama-7b              |
|-------------------|-----------------------|-----------------------|-----------------------|
| eager             | **0.027** / **0.027** | **0.136** / **0.137** | 0.399 / 0.395         |
| inductor          | 0.072 / 0.069         | **0.136** / **0.137** | 0.399 / 0.396         |
| tensorrt          | 0.100 / 0.099         | 0.161 / 0.161         | **0.339** / **0.340** |
| e0358 (no fusion) | 0.092 / 0.093         | 0.593 / 0.595         | 1.243 / 1.257         |
| e0358 (fusion)    | 0.042 / 0.042         | 0.495 / 0.497         | 0.902 / 0.903         |

**float16**

| variant           | bert-base             | gpt2-medium           | llama-7b              |
|-------------------|-----------------------|-----------------------|-----------------------|
| eager             | **0.026** / **0.027** | **0.063** / **0.063** | **0.097** / **0.098** |
| inductor          | 0.074 / 0.073         | 0.066 / 0.069         | **0.097** / **0.098** |
| tensorrt          | 0.114 / 0.114         | 0.122 / 0.124         | 0.127 / 0.131         |
| e0358 (no fusion) | 0.044 / 0.044         | 0.319 / 0.319         | 0.613 / 0.613         |
| e0358 (fusion)    | 0.032 / 0.034         | 0.239 / 0.240         | 0.377 / 0.377         |

**bfloat16**

| variant           | bert-base             | gpt2-medium           | llama-7b              |
|-------------------|-----------------------|-----------------------|-----------------------|
| eager             | **0.027** / **0.028** | **0.051** / **0.052** | **0.097** / **0.098** |
| inductor          | 0.072 / 0.076         | 0.066 / 0.069         | **0.097** / **0.098** |
| tensorrt          | 0.111 / 0.118         | 0.121 / 0.128         | 0.122 / 0.133         |
| e0358 (no fusion) | 0.044 / 0.045         | 0.318 / 0.320         | 0.613 / 0.614         |
| e0358 (fusion)    | 0.031 / 0.033         | 0.239 / 0.240         | 0.376 / 0.378         |

The sizes (N x d_model x d) are 512x768x64 for bert-base, 1024x1024x1024 for
gpt2-medium and 4096x4096x128 for llama-7b. Median and mean agree within 6%
for e0358 and within 9% for the baselines.

How the variants compare (medians):

- **Fusion vs no fusion:** fusion is faster at every size and dtype, by 2.2x
  (bert-base), 1.2x (gpt2-medium) and 1.4x (llama-7b) in float32. It saves
  two of every three loads of `x` and two of three kernel launches. The gain
  is largest at bert-base, whose kernels are small enough that their time is
  mostly fixed costs (launch, k-loop latency), which fusion pays once instead
  of three times. It is smallest at gpt2-medium, where the
  three 1024x1024 weight matrices, not `x`, dominate memory traffic.
- **e0358 vs eager and Inductor:** fused e0358 is 1.15x (bfloat16
  bert-base) to 4.7x (bfloat16 gpt2-medium) slower than eager. Inductor is
  slower than eager at bert-base in every dtype and at gpt2-medium in float16
  and bfloat16, and matches it at llama-7b. Eager's matmuls are cuBLAS
  kernels (nsys shows `ampere_sgemm_*` for float32), hand-tuned for this GPU.
  Each e0358 kernel is one simple k-loop that leaves scheduling to
  `tileiras`; it has no split-k, and its tile sizes come from a coarse sweep.
  In float16 it reaches 27 to 34 TFLOP/s fused at gpt2-medium and llama-7b,
  against eager's 103 to 133.
- **TensorRT:** it takes about 0.1 ms even at bert-base, which makes it the
  slowest baseline there. In float32 it beats eager only at llama-7b; in
  float16 and bfloat16, never.

## AI-assisted parts

The following parts were written with the help of an AI coding agent
(Claude Code). The affine loop fusion in `patches/` was not AI-assisted.

### Inherited setup script and gpu chip override

`setup.sh` (since rewritten, see below) and `GPU_CHIP` / `gpu_chip()` in
`e0358.py` were first written with AI help for the previous assignment; that
prompt is listed in its README.

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
torch-tensorrt), `E0358_CC` in `e0358.py`, and the Versions and Setup
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

> also whats the uv fix you did on the server>

> also the tensorrt deps and other code changes are you making on my laptop
> or the server

### CUDA Tile IR lowering, sibling fusion, and this README

Files: the CUDA Tile IR lowering (`lower_to_tile` and the functions it
calls), `Kernels`, `compile_tile`, `make_callable` and `run_library` in
`e0358.py`, the Python bindings build in `setup.sh`, and the Design and
Results sections above.
The agent reviewed the student's initial plan (first prompt below) and gave
design input. It wrote the matcher, the fusion decision and the kernel
builder, and picked the tile sizes from sweeps it ran on the RTX 4090. A
first version printed the kernels as text from an `mlir-translate`
translation in the llvm patch; a second built them with cuda-tile's C++
builders in an out-of-tree tool. Both were replaced by the Python bindings.

Prompts:

> okay now for lowering a pytroch affine loop to cuda tile ir, we should use
> pattern matching or regex to extract the sizes, dtypes and strides and
> exploit the fact that since its always going to be a matrix multiplication
> of qkv matrices with their respective weightds, then after extracting we
> use that info with c++ mlir builders to write the actual cuda tile ir ? hows
> does this plan sound

> in class ai assist was allowed

> yes they did

> yeah write this. Focus on as less number of lines as possible. I want it to
> be fully human readable and elegant. Focus on simplicity

> commit but tell me this, why are we generating text? can we do this - using
> C++ objects and MLIR APIs to programmatically create CUDA Tile operations,
> rather than generating MLIR code by concatenating strings.

> which approach is smaller and easier to understand? i hoped the builder
> approach would be smaller since the infra is built already?

> yeah builders please. write them, again, no redundant work, simplification
> and as less lines of code as possible

> Yes, push and update

> fill in the readme for now, again optimize for smaller lines of code for but
> with readability also check the instructions for all comparisions im sure we
> are missing some comparisons probably inductor or something

> why a shared gpu? no standalong gpu free?

> yeah run the checks now

> give me a reading list to understand what was done. be extremely bottoms up

> how are the measurements still going its been an hour

> how many new lines is the diff?

> change the code of the custom backend such that its not inside
> src/e0358.py but just the e0358.py file just like in the assignment 1
> submission folder e0358-submission

> for the measurements youre measuring the e0358 backend with 'both' fusion
> choice, against eager as reference and the default (all) baseline choices
> right? and iters should be default 50 and the stats median and mean i want
> both, if it helps look at the common parse args code to understand also
> default warmup (because why not, it cant be hurting). device cuda and dtype
> - all. is this what youre doing ? if not do this

> then update the results table in a clean manner in the readme

> make the best performing values in bold for each column

> why are the assignment 2 artifacts also names starting with bnc? they
> should be qkv or something. Delete all the assignment 1 artifacts please

> what are the 6 artifacts i got after running the qkv projection on cpu?
> explain in a single line what it is and at which step its generated

> are we sure these are the dependencies used? can you double check in
> server and laptop : [followed by the Versions part of asst-2.md, pasted]

> the instructions said we should need to patch stable hlo at all? so im
> confused how come we had to do it

> did we write cuTile to generate teh TILE IR?

> could we have generated the TILE IR lowering using the python bindings?
> would it have been shorter code then? The cuda-tile Python bindings
> (-DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON) are optional; they require MLIR to
> have been built with -DMLIR_ENABLE_BINDINGS_PYTHON=ON

> i think its worth a shot, first package the files needed for assignment 2
> submissiont into a separate folder outisde of e0358, in the readme remove
> everything assignment 1 related

> dont update the submission folder yet wait for my greenlight

> it seems the uv python version 3.12 is causing a lot of problems, any
> reasons we cant downgrade to 3.10 ?

> yeah do this : let each machine use its own system Python

> are these 3 fixes removed? or still needed? [followed by the three setup.sh
> fixes for the Python bindings, pasted]

> confirm and remove whatever we cna do without

Short status checks during the builds ("the build still running?",
"status", "statu", "status?", "update") and one-word confirmations ("yes")
are not listed separately.
