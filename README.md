# e0358

## Versions

| component    | version                                                    |
|--------------|------------------------------------------------------------|
| llvm-project | `d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde` (2026-11-03)    |
| stablehlo    | `b26a2e2a6e91b9144f8dc1aafbee74d01016d558` (2026-11-04)    |
| cuda-tile    | `v13.4.3`                                                  |

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

## Building

```sh
./setup.sh                 # llvm/mlir + patch, stablehlo, cuda-tile
source ~/llvm/e0358-env.sh # E0358_LLVM_BIN, E0358_LLVM_LIB, ...
```

`setup.sh` clones the three repositories under `$E0358_ROOT` (default
`~/llvm`), applies `patches/*.patch` on a new llvm branch `e0358-assign-2`,
builds everything, and writes `$E0358_ROOT/e0358-env.sh` with the environment
variables the backend reads. `E0358_GPU_CHIP` (e.g. `sm_89` for the RTX 4090)
can be set to override the detected GPU.

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
