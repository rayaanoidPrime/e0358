# e0358

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
