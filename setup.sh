#!/usr/bin/env bash
# Builds the Asst-2 toolchain for the e0358 backend:
#   - llvm-project at d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde (2026-11-03)
#     with the assignment patch(es) applied on a new branch e0358-assign-2
#   - stablehlo at b26a2e2a6e91b9144f8dc1aafbee74d01016d558 (2026-11-04)
#     with patches/stablehlo/*.patch (tablegen rename fix),
#     built against the llvm above
#   - cuda-tile at tag v13.4.3, built against the llvm above
# with the MLIR and cuda-tile Python bindings for the project's Python (run
# `uv sync` first), and writes $ROOT/e0358-env.sh with the environment the backend reads.
#
# Usage: ./setup.sh [PATCH...]
#   PATCH defaults to patches/*.patch in this repo (git format-patch output).
#
# Environment:
#   E0358_ROOT   where llvm-project/, stablehlo/ and cuda-tile/ live
#                (default ~/llvm)
#   E0358_CUDA   ON/OFF for MLIR_ENABLE_CUDA_RUNNER (default: ON if nvcc is
#                found); the gpu path needs libmlir_cuda_runtime.so
#   E0358_JOBS   compile jobs (default: nproc)
#   E0358_LINK_JOBS  parallel link jobs for llvm (default: 2; each link of
#                an llvm/mlir tool can take several GB of RAM)
#   E0358_PYTHON the Python to build the bindings for (default: .venv/bin/python
#                in this repo, which `uv sync` creates)
set -euo pipefail

LLVM_COMMIT=d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde
STABLEHLO_COMMIT=b26a2e2a6e91b9144f8dc1aafbee74d01016d558
CUDA_TILE_TAG=v13.4.3
BRANCH=e0358-assign-2

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${E0358_PYTHON:-$REPO/.venv/bin/python}
[ -x "$PYTHON" ] || { echo "error: $PYTHON not found; run uv sync first" >&2; exit 1; }
ROOT=${E0358_ROOT:-$HOME/llvm}
JOBS=${E0358_JOBS:-$(nproc)}
LINK_JOBS=${E0358_LINK_JOBS:-2}

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

# run as root, or through sudo when available (fresh VMs)
SUDO=()
[ "$(id -u)" = 0 ] || { command -v sudo >/dev/null && SUDO=(sudo); }

missing=()
for tool in git cmake ninja python3; do
  command -v $tool >/dev/null || missing+=($tool)
done
if [ ${#missing[@]} -gt 0 ]; then
  if command -v apt-get >/dev/null && { [ "$(id -u)" = 0 ] || [ ${#SUDO[@]} -gt 0 ]; }; then
    echo "== installing build tools: ${missing[*]}"
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y git cmake ninja-build clang lld ccache \
      python3 python3-pip
  else
    echo "error: missing build tools: ${missing[*]}" >&2
    exit 1
  fi
fi

if [ -z "${E0358_CUDA:-}" ]; then
  if command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
    E0358_CUDA=ON
  else
    E0358_CUDA=OFF
  fi
fi
if [ "$E0358_CUDA" = ON ] && ! command -v nvcc >/dev/null; then
  export PATH=/usr/local/cuda/bin:$PATH
fi

# prefer clang + lld + ccache when present
CMAKE_COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release)
if command -v clang >/dev/null && command -v clang++ >/dev/null; then
  CMAKE_COMMON+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
fi
USE_LLD=OFF
command -v ld.lld >/dev/null && USE_LLD=ON
USE_CCACHE=OFF
command -v ccache >/dev/null && USE_CCACHE=ON

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
LLVM_CXX_FLAGS=()
if [ "$E0358_CUDA" = ON ]; then
  CCCL_INC=$(find_cccl)
  if [ -z "$CCCL_INC" ]; then
    echo "== installing CUDA CCCL headers"
    python3 -m pip install --quiet --user nvidia-cuda-cccl ||
      python3 -m pip install --quiet --user --break-system-packages \
        nvidia-cuda-cccl
    CCCL_INC=$(find_cccl)
  fi
  [ -n "$CCCL_INC" ] && LLVM_CXX_FLAGS=(-DCMAKE_CXX_FLAGS="-I$CCCL_INC")
fi
mkdir -p "$ROOT"

# git am needs a committer identity; a fresh machine may lack one
IDENT=()
git config user.email >/dev/null ||
  IDENT=(-c user.name=e0358 -c user.email=e0358@localhost)

echo "== llvm $LLVM_COMMIT + ${#PATCHES[@]} patch(es) on $BRANCH"
cd "$ROOT"
if [ ! -d llvm-project/.git ]; then
  # shallow clone of just the pinned commit
  git init -q llvm-project
  git -C llvm-project remote add origin \
    https://github.com/llvm/llvm-project.git
fi
cd llvm-project
git cat-file -e "$LLVM_COMMIT^{commit}" 2>/dev/null ||
  git fetch --depth 1 origin "$LLVM_COMMIT"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  # rerun: keep what is there rather than applying the patch twice
  echo "branch $BRANCH exists, not reapplying patches"
  git switch "$BRANCH"
else
  if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "error: $ROOT/llvm-project has local changes" >&2
    exit 1
  fi
  git switch -c "$BRANCH" "$LLVM_COMMIT"
  for p in "${PATCHES[@]}"; do
    # Accept both git format-patch mailboxes and plain `git diff` files.
    if grep -q '^From ' "$p"; then
      git "${IDENT[@]}" am --3way "$p"
    else
      git apply "$p"
    fi
  done
fi
git log --oneline "$LLVM_COMMIT..HEAD"

# Python bindings for e0358.py, for $PYTHON: nanobind takes its Python from
# Python_EXECUTABLE, MLIR from Python3_EXECUTABLE; the type stubs are only
# for editors.
cmake -S llvm -B build "${CMAKE_COMMON[@]}" "${LLVM_CXX_FLAGS[@]}" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD="$USE_LLD" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_CCACHE_BUILD="$USE_CCACHE" \
  -DLLVM_PARALLEL_LINK_JOBS="$LINK_JOBS" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_PYTHON_STUBGEN_ENABLED=OFF \
  -DPython3_EXECUTABLE="$PYTHON" -DPython_EXECUTABLE="$PYTHON" \
  -DMLIR_ENABLE_CUDA_RUNNER="$E0358_CUDA"
ninja -C build -j "$JOBS"
LLVM_BUILD=$ROOT/llvm-project/build

echo "== stablehlo $STABLEHLO_COMMIT"
cd "$ROOT"
[ -d stablehlo/.git ] || git clone https://github.com/openxla/stablehlo.git
cd stablehlo
git cat-file -e "$STABLEHLO_COMMIT^{commit}" 2>/dev/null ||
  git fetch origin "$STABLEHLO_COMMIT"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "branch $BRANCH exists, not reapplying patches"
  git switch "$BRANCH"
else
  git switch -c "$BRANCH" "$STABLEHLO_COMMIT"
  # stablehlo fix for the llvm CMake tablegen() rename (see README)
  for p in "$REPO"/patches/stablehlo/*.patch; do
    [ -f "$p" ] && git "${IDENT[@]}" am --3way "$p"
  done
fi
git log --oneline "$STABLEHLO_COMMIT..HEAD"
cmake -S . -B build "${CMAKE_COMMON[@]}" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD="$USE_LLD" \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
ninja -C build -j "$JOBS" stablehlo-opt stablehlo-translate

echo "== cuda-tile $CUDA_TILE_TAG"
cd "$ROOT"
[ -d cuda-tile/.git ] || git clone https://github.com/NVIDIA/cuda-tile.git
cd cuda-tile
git rev-parse -q --verify "refs/tags/$CUDA_TILE_TAG" >/dev/null ||
  git fetch origin tag "$CUDA_TILE_TAG"
git -c advice.detachedHead=false checkout "$CUDA_TILE_TAG"
grep -q "$LLVM_COMMIT" cmake/IncludeLLVM.cmake ||
  echo "warning: cuda-tile $CUDA_TILE_TAG pins a different llvm commit" >&2
# cuda-tile's bindings get their own nanobind domain, so that they load next
# to MLIR's in one process.
cmake -S . -B build "${CMAKE_COMMON[@]}" \
  -DCUDA_TILE_ENABLE_CCACHE="$USE_CCACHE" \
  -DCUDA_TILE_USE_LLVM_INSTALL_DIR="$LLVM_BUILD" \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON -DMLIR_PYTHON_STUBGEN_ENABLED=OFF \
  -DPython3_EXECUTABLE="$PYTHON" -DPython_EXECUTABLE="$PYTHON" \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=cuda_tile
ninja -C build -j "$JOBS"

ENV_FILE=$ROOT/e0358-env.sh
cat > "$ENV_FILE" <<ENV
# source this before running the e0358 backend
export E0358_LLVM_BIN=$LLVM_BUILD/bin
export E0358_LLVM_LIB=$LLVM_BUILD/lib
export E0358_STABLEHLO_BIN=$ROOT/stablehlo/build/bin
# MLIR's and cuda-tile's Python bindings (e0358.py imports both)
export PYTHONPATH=$LLVM_BUILD/tools/mlir/python_packages/mlir_core:$ROOT/cuda-tile/build/python_packages\${PYTHONPATH:+:\$PYTHONPATH}
ENV

echo "== done"
"$LLVM_BUILD/bin/mlir-opt" --version | sed -n 2p
"$ROOT/stablehlo/build/bin/stablehlo-opt" --version | sed -n 2p || true
"$PYTHON" -c "import sys; sys.path[:0] = '$LLVM_BUILD/tools/mlir/python_packages/mlir_core $ROOT/cuda-tile/build/python_packages'.split(); import mlir.ir, cuda_tile._mlir.dialects.cuda_tile_ops; print('python bindings: ok')"
if [ "$E0358_CUDA" = ON ]; then
  ls "$LLVM_BUILD/lib/libmlir_cuda_runtime.so"
else
  echo "note: cuda runner off; only --device cpu will run"
fi
echo "environment: source $ENV_FILE"
