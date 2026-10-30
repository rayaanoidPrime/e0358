#!/usr/bin/env bash
# Builds the toolchain for the e0358 backend:
#   - llvm at llvmorg-23.1.3 with the assignment patch applied on a new
#     branch e0358-assign-1
#   - stablehlo at 965c428908be23f05e80e3c8e7f0b1a806ac3751 (2026-08-04),
#     built against the llvm above
#
# Usage: ./setup.sh [PATCH...]
#   PATCH defaults to patches/*.patch in this repo (git format-patch output).
#
# Environment:
#   E0358_ROOT   where llvm-project/ and stablehlo/ live (default ~/llvm,
#                which is where src/e0358/backend.py looks for the tools)
#   E0358_CUDA   ON/OFF for MLIR_ENABLE_CUDA_RUNNER (default: ON if nvcc is
#                found); the gpu path needs libmlir_cuda_runtime.so
#   E0358_JOBS   compile jobs (default: nproc)
set -euo pipefail

LLVM_TAG=llvmorg-23.1.3
STABLEHLO_COMMIT=965c428908be23f05e80e3c8e7f0b1a806ac3751
BRANCH=e0358-assign-1

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${E0358_ROOT:-$HOME/llvm}
JOBS=${E0358_JOBS:-$(nproc)}

if [ $# -gt 0 ]; then
  PATCHES=("$@")
else
  shopt -s nullglob
  PATCHES=("$REPO"/patches/*.patch)
  shopt -u nullglob
fi
if [ ${#PATCHES[@]} -eq 0 ]; then
  echo "error: no patch given and none in $REPO/patches/" >&2
  exit 1
fi
for p in "${PATCHES[@]}"; do
  [ -f "$p" ] || { echo "error: $p not found" >&2; exit 1; }
done
mapfile -t PATCHES < <(realpath "${PATCHES[@]}")

if [ -z "${E0358_CUDA:-}" ]; then
  if command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
    E0358_CUDA=ON
  else
    E0358_CUDA=OFF
  fi
fi

# prefer clang + lld + ccache when present, as on the dev machine
CMAKE_COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON)
if command -v clang >/dev/null && command -v clang++ >/dev/null; then
  CMAKE_COMMON+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
fi
if command -v ld.lld >/dev/null; then
  CMAKE_COMMON+=(-DLLVM_ENABLE_LLD=ON)
fi
LLVM_CCACHE=OFF
command -v ccache >/dev/null && LLVM_CCACHE=ON

missing=()
for tool in git cmake ninja; do
  command -v $tool >/dev/null || missing+=($tool)
done
if [ ${#missing[@]} -gt 0 ]; then
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    echo "== installing build tools: ${missing[*]}"
    apt-get update
    apt-get install -y git cmake ninja-build clang lld ccache
  else
    echo "error: missing build tools: ${missing[*]}" >&2
    exit 1
  fi
fi

# CUDA 13 ships CCCL headers separately; CudaRuntimeWrappers needs <nv/target>.
find_cccl() {
  python3 - <<'PY'
import glob, site
for base in site.getsitepackages() + [site.getusersitepackages()]:
    for hit in glob.glob(base + "/nvidia/*/include/cccl/nv/target"):
        print(hit[: -len("/nv/target")])
        raise SystemExit
PY
}
if [ "$E0358_CUDA" = ON ]; then
  CCCL_INC=$(find_cccl)
  if [ -z "$CCCL_INC" ]; then
    echo "== installing CUDA CCCL headers"
    python3 -m pip install --quiet nvidia-cuda-cccl
    CCCL_INC=$(find_cccl)
  fi
  [ -n "$CCCL_INC" ] && CMAKE_COMMON+=(-DCMAKE_CXX_FLAGS="-I$CCCL_INC")
fi
mkdir -p "$ROOT"

echo "== llvm $LLVM_TAG + ${#PATCHES[@]} patch(es) on $BRANCH"
cd "$ROOT"
if [ ! -d llvm-project ]; then
  git clone --branch "$LLVM_TAG" --depth 1 \
    https://github.com/llvm/llvm-project.git
fi
cd llvm-project
git fetch --depth 1 origin tag "$LLVM_TAG"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  # rerun: keep what is there rather than applying the patch twice
  echo "branch $BRANCH exists, not reapplying patches"
  git switch "$BRANCH"
else
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: $ROOT/llvm-project has local changes" >&2
    exit 1
  fi
  git switch -c "$BRANCH" "$LLVM_TAG"
  # git am needs a committer identity; a fresh remote machine may lack one
  ident=()
  git config user.email >/dev/null ||
    ident=(-c user.name=e0358 -c user.email=e0358@localhost)
  for p in "${PATCHES[@]}"; do
    # Accept both git format-patch mailboxes and plain `git diff` files.
    if grep -q '^From ' "$p"; then
      git "${ident[@]}" am --3way "$p"
    else
      git apply "$p"
    fi
  done
fi
git log --oneline "$LLVM_TAG^..HEAD"

cmake -S llvm -B build "${CMAKE_COMMON[@]}" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_CCACHE_BUILD="$LLVM_CCACHE" \
  -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_ENABLE_CUDA_RUNNER="$E0358_CUDA"
ninja -C build -j "$JOBS"

echo "== stablehlo $STABLEHLO_COMMIT"
cd "$ROOT"
if [ ! -d stablehlo ]; then
  git clone https://github.com/openxla/stablehlo.git
fi
cd stablehlo
git fetch origin "$STABLEHLO_COMMIT"
git switch --detach "$STABLEHLO_COMMIT"
cmake -S . -B build "${CMAKE_COMMON[@]}" \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_DIR="$ROOT/llvm-project/build/lib/cmake/mlir"
ninja -C build -j "$JOBS" stablehlo-opt stablehlo-translate

echo "== done"
"$ROOT/llvm-project/build/bin/mlir-opt" --version | sed -n 2p
"$ROOT/stablehlo/build/bin/stablehlo-opt" --version | sed -n 2p || true
if [ "$E0358_CUDA" = ON ]; then
  ls "$ROOT/llvm-project/build/lib/libmlir_cuda_runtime.so"
else
  echo "note: cuda runner off; only --device cpu will run"
fi
