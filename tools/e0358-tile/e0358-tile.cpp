//===- e0358-tile.cpp - Affine matmul nests to CUDA Tile IR ---------------===//
//
// Reads a function made of affine matmul nests,
//
//   for i, j: c[i, j] = 0
//   for i, j, k: c[i, j] += a[i, k] * b[k, j]
//
// and builds a CUDA Tile IR module with one kernel per group of matmuls,
// written as bytecode. Each kernel block computes one tile of c, accumulating
// over k in f32.
//
// With --sibling-fusion, matmuls that read the same a (same shapes, and no
// matmul writing what another reads) form one group: its kernel loads each
// tile of a once and multiplies it with the tiles of every b.
//
// Prints the line each kernel is launched from, then the module:
//
//   // launch @k0 grid=[8, 1] args=[1, 0, 4]
//
// args are the function argument numbers, in kernel parameter order.
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/InterleavedRange.h"

using namespace mlir;
using namespace mlir::affine;
namespace ct = mlir::cuda_tile;

namespace {
/// c[m x n] += a[m x k] * b[k x n], on function arguments.
struct Matmul {
  BlockArgument a, b, c;
  int64_t m, n, k;
};
} // namespace

/// Returns the loops of the perfect nest rooted at `loop`, outermost first.
static SmallVector<AffineForOp> perfectNest(AffineForOp loop) {
  SmallVector<AffineForOp> nest = {loop};
  while (llvm::hasSingleElement(nest.back().getBody()->without_terminator()))
    if (auto inner = dyn_cast<AffineForOp>(nest.back().getBody()->front()))
      nest.push_back(inner);
    else
      break;
  return nest;
}

/// Returns the trip count of a loop from 0 with step 1, or 0 for other loops.
static int64_t tripCount(AffineForOp loop) {
  if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0 ||
      loop.getStepAsInt() != 1)
    return 0;
  return loop.getConstantUpperBound();
}

/// Returns the function argument that `op` accesses at exactly `ivs`.
template <typename OpTy>
static BlockArgument accessAt(OpTy op, ArrayRef<Value> ivs) {
  if (!op || !op.getAffineMap().isIdentity() ||
      !llvm::equal(op.getMapOperands(), ivs))
    return nullptr;
  auto arg = dyn_cast<BlockArgument>(op.getMemRef());
  auto type = cast<MemRefType>(op.getMemRef().getType());
  return arg && type.getLayout().isIdentity() ? arg : nullptr;
}

/// Matches `for i, j: c[i, j] = 0` and returns c.
static BlockArgument matchZeroFill(AffineForOp loop) {
  SmallVector<AffineForOp> nest = perfectNest(loop);
  if (nest.size() != 2 || !tripCount(nest[0]) || !tripCount(nest[1]))
    return nullptr;
  auto store = dyn_cast<AffineStoreOp>(nest[1].getBody()->front());
  if (!store || !matchPattern(store.getValue(), m_AnyZeroFloat()))
    return nullptr;
  return accessAt(store,
                  {nest[0].getInductionVar(), nest[1].getInductionVar()});
}

/// Matches `for i, j, k: c[i, j] += a[i, k] * b[k, j]`.
static std::optional<Matmul> matchMatmul(AffineForOp loop) {
  SmallVector<AffineForOp> nest = perfectNest(loop);
  if (nest.size() != 3)
    return std::nullopt;
  Value i = nest[0].getInductionVar(), j = nest[1].getInductionVar(),
        k = nest[2].getInductionVar();
  Block *body = nest[2].getBody();
  auto store = dyn_cast<AffineStoreOp>(body->back().getPrevNode());
  auto add = store ? store.getValue().getDefiningOp<arith::AddFOp>() : nullptr;
  if (!add || body->getOperations().size() != 7)
    return std::nullopt;
  // c[i, j] + a * b, in either order.
  Value acc = add.getLhs(), product = add.getRhs();
  if (!product.getDefiningOp<arith::MulFOp>())
    std::swap(acc, product);
  auto mul = product.getDefiningOp<arith::MulFOp>();
  if (!mul)
    return std::nullopt;
  Matmul mm{accessAt(mul.getLhs().getDefiningOp<AffineLoadOp>(), {i, k}),
            accessAt(mul.getRhs().getDefiningOp<AffineLoadOp>(), {k, j}),
            accessAt(store, {i, j}),
            tripCount(nest[0]),
            tripCount(nest[1]),
            tripCount(nest[2])};
  if (!mm.a || !mm.b || !mm.c || !mm.m || !mm.n || !mm.k ||
      accessAt(acc.getDefiningOp<AffineLoadOp>(), {i, j}) != mm.c)
    return std::nullopt;
  return mm;
}

/// Whether `x` and `y` can share one kernel: the same a and shapes, and
/// neither writes what the other reads.
static bool fusable(const Matmul &x, const Matmul &y) {
  return x.a == y.a && x.m == y.m && x.n == y.n && x.k == y.k &&
         !llvm::is_contained({y.a, y.b}, x.c) &&
         !llvm::is_contained({x.a, x.b}, y.c);
}

/// Tile sizes (m x n x k) for a group of `count` matmuls of `mm`'s shape,
/// picked from measurements on an RTX 4090: 16-deep k tiles, enough blocks
/// for its 128 SMs where the matrix allows it, and, in fused groups, where
/// each matmul keeps its own accumulator, narrower tiles.
static std::array<int64_t, 3> tileSizes(const Matmul &mm, size_t count) {
  const int64_t numSMs = 128;
  bool f32 = cast<MemRefType>(mm.a.getType()).getElementType().isF32();
  auto blocks = [&](int64_t tm, int64_t tn) {
    return (mm.m / tm) * (mm.n / tn);
  };
  int64_t tm = f32 && count == 1 ? 32 : 64;
  if (blocks(tm, 32) < numSMs)
    tm = 32;
  int64_t tn = count == 1 ? 128 : f32 ? 32 : 64;
  while (tn > 32 && (mm.n % tn || blocks(tm, tn) < numSMs))
    tn /= 2;
  return {tm, tn, 16};
}

/// Builds the kernel computing `group` as @<name>, with tm x tn tiles of c and
/// tk-deep steps over k.
static void buildKernel(OpBuilder &b, StringRef name, ArrayRef<Matmul> group,
                        int64_t tm, int64_t tn, int64_t tk) {
  const Matmul &mm = group.front();
  Location loc = b.getUnknownLoc();
  Type t = cast<MemRefType>(mm.a.getType()).getElementType();
  auto tile = [](ArrayRef<int64_t> shape, Type type) {
    return ct::TileType::get(shape, type);
  };
  auto constant = [&](Type type, Attribute value) {
    auto attr = DenseElementsAttr::get(cast<ShapedType>(type), value);
    return ct::ConstantOp::create(b, loc, type,
                                  cast<DenseTypedElementsAttr>(attr));
  };
  auto i32 = tile({}, b.getI32Type()), acc = tile({tm, tn}, b.getF32Type());
  auto token = ct::TokenType::get(b.getContext());

  // @<name>(a, b0, c0, b1, c1, ...): pointers to the matrices.
  SmallVector<Type> params(1 + 2 * group.size(),
                           tile({}, ct::PointerType::get(t)));
  auto entry = ct::EntryOp::create(b, loc, name, b.getFunctionType(params, {}),
                                   /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
                                   /*optimization_hints=*/nullptr);
  OpBuilder::InsertionGuard guard(b);
  Block *body = entry.addEntryBlock();
  b.setInsertionPointToStart(body);
  auto ids = ct::GetTileBlockIdOp::create(b, loc, i32, i32, i32);
  Value i = ids.getBlockIdX(), j = ids.getBlockIdY();

  // The row-major matrix at `ptr` cut into `shape` tiles, and their loads.
  auto tiles = [&](Value ptr, int64_t rows, int64_t cols,
                   ArrayRef<int32_t> shape) -> Value {
    auto view =
        ct::TensorViewType::get(b.getContext(), t, {rows, cols}, {cols, 1});
    auto type = ct::PartitionViewType::get(
        b.getContext(), b.getDenseI32ArrayAttr(shape), view, {0, 1}, nullptr);
    return ct::MakePartitionViewOp::create(
        b, loc, type,
        ct::MakeTensorViewOp::create(b, loc, view, ptr, ValueRange(),
                                     ValueRange()));
  };
  auto load = [&](Value tiles, Value row, Value col, ArrayRef<int64_t> shape) {
    return ct::LoadViewTkoOp::create(
               b, loc, tile(shape, t), token, ct::MemoryOrderingSemantics::WEAK,
               nullptr, tiles, ValueRange{row, col}, nullptr, nullptr)
        .getTile();
  };
  Value aTiles =
      tiles(body->getArgument(0), mm.m, mm.k, {int32_t(tm), int32_t(tk)});
  SmallVector<Value> bTiles, cTiles;
  for (size_t n = 0; n < group.size(); ++n) {
    bTiles.push_back(tiles(body->getArgument(1 + 2 * n), mm.k, mm.n,
                           {int32_t(tk), int32_t(tn)}));
    cTiles.push_back(tiles(body->getArgument(2 + 2 * n), mm.m, mm.n,
                           {int32_t(tm), int32_t(tn)}));
  }

  // for k: acc_n += a[i, k] * b_n[k, j], one f32 accumulator per matmul.
  auto loop = ct::ForOp::create(
      b, loc, constant(i32, b.getI32IntegerAttr(0)),
      constant(i32, b.getI32IntegerAttr(mm.k / tk)),
      constant(i32, b.getI32IntegerAttr(1)),
      SmallVector<Value>(group.size(), constant(acc, b.getF32FloatAttr(0))),
      [&](OpBuilder &, Location, Value k, ValueRange accs) {
        Value x = load(aTiles, i, k, {tm, tk});
        SmallVector<Value> sums;
        for (auto [bn, accn] : llvm::zip(bTiles, accs))
          sums.push_back(ct::MmaFOp::create(b, loc, x, load(bn, k, j, {tk, tn}),
                                            accn,
                                            /*fast_acc=*/nullptr));
        ct::ContinueOp::create(b, loc, sums);
      });

  // c_n[i, j] = acc_n, in the element type.
  for (auto [result, cn] : llvm::zip(loop.getResults(), cTiles)) {
    Value value = result;
    if (t != b.getF32Type())
      value = ct::FToFOp::create(b, loc, tile({tm, tn}, t), value);
    ct::StoreViewTkoOp::create(b, loc, token, ct::MemoryOrderingSemantics::WEAK,
                               nullptr, value, cn, ValueRange{i, j}, nullptr,
                               nullptr);
  }
  ct::ReturnOp::create(b, loc);
}

/// Builds the kernels of `func` into `module` and prints their launch lines.
static LogicalResult translate(func::FuncOp func, ct::ModuleOp module,
                               bool siblingFusion) {
  // Collect the matmuls and the buffers zeroed before them; anything else in
  // the function cannot be translated.
  SmallVector<Matmul> matmuls;
  DenseSet<Value> zeroed;
  for (Operation &op : func.getBody().front()) {
    auto loop = dyn_cast<AffineForOp>(op);
    if (BlockArgument c = loop ? matchZeroFill(loop) : nullptr) {
      zeroed.insert(c);
      continue;
    }
    if (std::optional<Matmul> mm = loop ? matchMatmul(loop) : std::nullopt) {
      if (!zeroed.contains(mm->c))
        return loop.emitError("matmul output is not zero-initialized");
      matmuls.push_back(*mm);
      continue;
    }
    if (!isa<arith::ConstantOp, func::ReturnOp>(op))
      return op.emitError("not a matmul nest");
  }

  // Group the matmuls into kernels: each alone, or with its siblings.
  SmallVector<SmallVector<Matmul>> groups;
  for (const Matmul &mm : matmuls) {
    auto group = llvm::find_if(groups, [&](ArrayRef<Matmul> g) {
      return siblingFusion &&
             llvm::all_of(g, [&](const Matmul &x) { return fusable(x, mm); });
    });
    if (group == groups.end())
      groups.push_back({mm});
    else
      group->push_back(mm);
  }

  OpBuilder b = OpBuilder::atBlockEnd(&module.getBody().front());
  for (auto [id, group] : llvm::enumerate(groups)) {
    const Matmul &mm = group.front();
    auto [tm, tn, tk] = tileSizes(mm, group.size());
    if (mm.m % tm || mm.n % tn || mm.k % tk)
      return func.emitError(llvm::formatv(
          "matmul of {0}x{1}x{2} is not made of {3}x{4}x{5} tiles", mm.m, mm.n,
          mm.k, tm, tn, tk));
    std::string name = "k" + std::to_string(id);
    buildKernel(b, name, group, tm, tn, tk);
    SmallVector<unsigned> args = {mm.a.getArgNumber()};
    for (const Matmul &mmn : group)
      args.append({mmn.b.getArgNumber(), mmn.c.getArgNumber()});
    llvm::outs() << llvm::formatv("// launch @{0} grid=[{1}, {2}] args=[{3}]\n",
                                  name, mm.m / tm, mm.n / tn,
                                  llvm::interleaved(args));
  }
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                          llvm::cl::desc("<affine .mlir>"));
  static llvm::cl::opt<std::string> output(
      "o", llvm::cl::desc("CUDA Tile IR bytecode file"), llvm::cl::Required);
  static llvm::cl::opt<bool> siblingFusion(
      "sibling-fusion", llvm::cl::desc("Fuse matmuls sharing their a"));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "affine matmul nests to CUDA Tile IR\n");

  MLIRContext context;
  context.loadDialect<AffineDialect, arith::ArithDialect, func::FuncDialect,
                      memref::MemRefDialect, ct::CudaTileDialect>();
  OwningOpRef<ModuleOp> affine = parseSourceFile<ModuleOp>(input, &context);
  if (!affine)
    return 1;
  auto funcs = affine->getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return affine->emitError("expected a single function"), 1;

  OpBuilder b(&context);
  OwningOpRef<ct::ModuleOp> module =
      ct::ModuleOp::create(b, b.getUnknownLoc(), "kernels");
  std::error_code error;
  llvm::raw_fd_ostream os(output, error);
  if (failed(translate(*funcs.begin(), *module, siblingFusion)) ||
      failed(verify(*module)) || error ||
      failed(ct::writeBytecode(os, *module)))
    return 1;
  module->print(llvm::outs());
  return 0;
}
