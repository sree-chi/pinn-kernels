// LLVM code generator for stencil-residual kernels.
//
// From one PDESpec (graph.h) this emits, per PDE:
//   CPU (ORC LLJIT, host target):
//     pk_fwd       residual r = R(u) on the interior of an (nt, nx) grid
//     pk_adj       adjoint  gu = (dR/du)^T g, gather form over a zero-padded g
//     pk_pgrad_<p> scalar   sum(g * dR/dp) for each physical parameter p
//   GPU (NVPTX backend, PTX text):
//     pk_fwd, pk_adj  one thread per grid point
//
// Optimization is an explicit new-pass-manager pipeline (O0..O3) run before
// the module is handed to the JIT, so both pre- and post-optimization IR can
// be inspected. Pointer arguments are marked noalias/nocapture and all index
// arithmetic is 64-bit nsw, which is what lets the loop vectorizer fire.

#include "graph.h"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

thread_local std::string g_error;
thread_local std::string g_text;

void initTargets() {
  static std::once_flag once;
  std::call_once(once, [] {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

std::string moduleText(const Module &M) {
  std::string s;
  raw_string_ostream os(s);
  M.print(os, nullptr);
  return os.str();
}

void runPipeline(Module &M, TargetMachine *TM, int level) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM;
  switch (level) {
    case 0: MPM = PB.buildO0DefaultPipeline(OptimizationLevel::O0); break;
    case 1: MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O1); break;
    case 2: MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2); break;
    default: MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3); break;
  }
  MPM.run(M, MAM);
}

// Interior iteration bounds implied by the stencil offsets.
struct Bounds {
  int tLo = 0, tHi = 0, xLo = 0, xHi = 0;  // t in [tLo, nt - tHi), x in [xLo, nx - xHi)
  int halo = 0;                            // max |offset|, for padding g in the adjoint
};

Bounds boundsOf(const pk::E &residual) {
  std::set<std::pair<int, int>> loads;
  pk::collectLoads(residual, loads);
  Bounds b;
  for (auto [dt, dx] : loads) {
    b.tLo = std::max(b.tLo, -dt);
    b.tHi = std::max(b.tHi, dt);
    b.xLo = std::max(b.xLo, -dx);
    b.xHi = std::max(b.xHi, dx);
    b.halo = std::max({b.halo, std::abs(dt), std::abs(dx)});
  }
  return b;
}

// ---------------------------------------------------------------------------
// Expression emission
// ---------------------------------------------------------------------------
struct EmitCtx {
  IRBuilder<> &B;
  Value *u, *params, *nt, *nx;
  Value *t, *x;                       // i64 coordinates of the evaluation point
  int shiftT = 0, shiftX = 0;         // evaluate at (t + shiftT, x + shiftX)
  bool clamp = false;                 // clamp loads into the grid (adjoint, nonlinear)
  const std::map<std::string, int> &paramIndex;
  std::map<const pk::Expr *, Value *> memo;
};

Value *emit(const pk::E &e, EmitCtx &c) {
  auto it = c.memo.find(e.get());
  if (it != c.memo.end()) return it->second;
  IRBuilder<> &B = c.B;
  Type *f32 = B.getFloatTy(), *i64 = B.getInt64Ty();
  Value *v = nullptr;
  switch (e->kind) {
    case pk::Kind::Const:
      v = ConstantFP::get(f32, e->value);
      break;
    case pk::Kind::Param: {
      auto pi = c.paramIndex.find(e->name);
      if (pi == c.paramIndex.end()) throw std::runtime_error("unknown parameter " + e->name);
      Value *p = B.CreateInBoundsGEP(f32, c.params, ConstantInt::get(i64, pi->second));
      v = B.CreateLoad(f32, p, e->name);
      break;
    }
    case pk::Kind::Load: {
      Value *tt = B.CreateNSWAdd(c.t, ConstantInt::get(i64, e->dt + c.shiftT));
      Value *xx = B.CreateNSWAdd(c.x, ConstantInt::get(i64, e->dx + c.shiftX));
      if (c.clamp) {
        Value *zero = ConstantInt::get(i64, 0), *one = ConstantInt::get(i64, 1);
        tt = B.CreateBinaryIntrinsic(Intrinsic::smin, B.CreateBinaryIntrinsic(Intrinsic::smax, tt, zero),
                                     B.CreateNSWSub(c.nt, one));
        xx = B.CreateBinaryIntrinsic(Intrinsic::smin, B.CreateBinaryIntrinsic(Intrinsic::smax, xx, zero),
                                     B.CreateNSWSub(c.nx, one));
      }
      Value *idx = B.CreateNSWAdd(B.CreateNSWMul(tt, c.nx), xx);
      v = B.CreateLoad(f32, B.CreateInBoundsGEP(f32, c.u, idx));
      break;
    }
    case pk::Kind::Add: v = B.CreateFAdd(emit(e->a, c), emit(e->b, c)); break;
    case pk::Kind::Sub: v = B.CreateFSub(emit(e->a, c), emit(e->b, c)); break;
    case pk::Kind::Mul: v = B.CreateFMul(emit(e->a, c), emit(e->b, c)); break;
    case pk::Kind::Div: v = B.CreateFDiv(emit(e->a, c), emit(e->b, c)); break;
  }
  c.memo[e.get()] = v;
  return v;
}

FastMathFlags fmfFor(bool fast) {
  FastMathFlags f;
  f.setAllowContract(true);  // permit FMA formation; numerically benign
  if (fast) f.setFast();     // reassoc/arcp/etc. (enables vectorized reductions)
  return f;
}

// Two nested counted loops: t in [tLo, tHi), x in [xLo, xHi). The body must not
// create new basic blocks.
void forRange2D(IRBuilder<> &B, Function *F, Value *tLo, Value *tHi, Value *xLo, Value *xHi,
                const std::function<void(Value *, Value *)> &body) {
  LLVMContext &C = B.getContext();
  Type *i64 = B.getInt64Ty();
  BasicBlock *pre = B.GetInsertBlock();
  BasicBlock *tHead = BasicBlock::Create(C, "t.head", F);
  BasicBlock *xBody = BasicBlock::Create(C, "x.body", F);
  BasicBlock *tLatch = BasicBlock::Create(C, "t.latch", F);
  BasicBlock *exit = BasicBlock::Create(C, "loops.exit", F);

  B.CreateCondBr(B.CreateAnd(B.CreateICmpSLT(tLo, tHi), B.CreateICmpSLT(xLo, xHi)), tHead, exit);

  B.SetInsertPoint(tHead);
  PHINode *t = B.CreatePHI(i64, 2, "t");
  t->addIncoming(tLo, pre);
  B.CreateBr(xBody);

  B.SetInsertPoint(xBody);
  PHINode *x = B.CreatePHI(i64, 2, "x");
  x->addIncoming(xLo, tHead);
  body(t, x);
  Value *xn = B.CreateNSWAdd(x, ConstantInt::get(i64, 1), "x.next");
  x->addIncoming(xn, B.GetInsertBlock());
  B.CreateCondBr(B.CreateICmpSLT(xn, xHi), xBody, tLatch);

  B.SetInsertPoint(tLatch);
  Value *tn = B.CreateNSWAdd(t, ConstantInt::get(i64, 1), "t.next");
  t->addIncoming(tn, tLatch);
  B.CreateCondBr(B.CreateICmpSLT(tn, tHi), tHead, exit);

  B.SetInsertPoint(exit);
}

Function *declare(Module &M, const std::string &name, Type *ret, std::vector<Type *> args,
                  const std::vector<int> &noaliasArgs, const std::vector<int> &readonlyArgs) {
  auto *F = Function::Create(FunctionType::get(ret, args, false), Function::ExternalLinkage, name, M);
  F->addFnAttr(Attribute::NoUnwind);
  for (int i : noaliasArgs) {
    F->addParamAttr(i, Attribute::NoAlias);
    F->addParamAttr(i, Attribute::NoCapture);
  }
  for (int i : readonlyArgs) F->addParamAttr(i, Attribute::ReadOnly);
  return F;
}

void verifyOrThrow(Module &M) {
  std::string s;
  raw_string_ostream os(s);
  if (verifyModule(M, &os)) throw std::runtime_error("LLVM IR verification failed:\n" + os.str());
}

// Adjoint accumulation shared by CPU and GPU emitters:
//   gu[t,x] = sum_k coef_k(t - dt_k, x - dx_k) * g[t - dt_k, x - dx_k]
// g is zero-padded by `halo` on every side, so the gather never reads out of
// bounds and residual points outside the interior contribute zero.
Value *emitAdjointPoint(IRBuilder<> &B, const pk::PDESpec &spec, const Bounds &bd, Value *gpad, Value *u,
                        Value *params, Value *nt, Value *nx, Value *t, Value *x,
                        const std::map<std::string, int> &pidx) {
  Type *f32 = B.getFloatTy(), *i64 = B.getInt64Ty();
  std::set<std::pair<int, int>> loads;
  pk::collectLoads(spec.residual, loads);
  Value *W = B.CreateNSWAdd(nx, ConstantInt::get(i64, 2 * bd.halo));
  Value *acc = ConstantFP::get(f32, 0.0);
  for (auto [dt, dx] : loads) {
    pk::E coef = pk::diffLoad(spec.residual, dt, dx);
    if (pk::isConst(coef, 0.f)) continue;
    EmitCtx c{B, u, params, nt, nx, t, x, -dt, -dx, pk::hasLoad(coef), pidx, {}};
    Value *cv = emit(coef, c);
    Value *gt = B.CreateNSWAdd(t, ConstantInt::get(i64, bd.halo - dt));
    Value *gx = B.CreateNSWAdd(x, ConstantInt::get(i64, bd.halo - dx));
    Value *gv = B.CreateLoad(f32, B.CreateInBoundsGEP(f32, gpad, B.CreateNSWAdd(B.CreateNSWMul(gt, W), gx)));
    acc = B.CreateFAdd(acc, B.CreateFMul(cv, gv));
  }
  return acc;
}

std::map<std::string, int> paramIndexOf(const pk::PDESpec &spec) {
  std::map<std::string, int> m;
  for (size_t i = 0; i < spec.params.size(); ++i) m[spec.params[i]] = static_cast<int>(i);
  return m;
}

// ---------------------------------------------------------------------------
// CPU module
// ---------------------------------------------------------------------------
void buildCpuModule(Module &M, const pk::PDESpec &spec, const Bounds &bd, bool fast) {
  LLVMContext &C = M.getContext();
  IRBuilder<> B(C);
  B.setFastMathFlags(fmfFor(fast));
  Type *ptr = B.getPtrTy(), *i64 = B.getInt64Ty(), *f32 = B.getFloatTy(), *f64 = B.getDoubleTy();
  auto pidx = paramIndexOf(spec);
  auto K = [&](int v) { return ConstantInt::get(i64, v); };

  // void pk_fwd(const float* u, float* res, const float* params, i64 nt, i64 nx)
  {
    Function *F = declare(M, "pk_fwd", B.getVoidTy(), {ptr, ptr, ptr, i64, i64}, {0, 1, 2}, {0, 2});
    auto a = F->arg_begin();
    Value *u = a++, *res = a++, *params = a++, *nt = a++, *nx = a++;
    B.SetInsertPoint(BasicBlock::Create(C, "entry", F));
    forRange2D(B, F, K(bd.tLo), B.CreateNSWSub(nt, K(bd.tHi)), K(bd.xLo), B.CreateNSWSub(nx, K(bd.xHi)),
               [&](Value *t, Value *x) {
                 EmitCtx c{B, u, params, nt, nx, t, x, 0, 0, false, pidx, {}};
                 Value *r = emit(spec.residual, c);
                 Value *idx = B.CreateNSWAdd(B.CreateNSWMul(t, nx), x);
                 B.CreateStore(r, B.CreateInBoundsGEP(f32, res, idx));
               });
    B.CreateRetVoid();
  }

  // void pk_adj(const float* gpad, float* gu, const float* u, const float* params, i64 nt, i64 nx)
  {
    Function *F = declare(M, "pk_adj", B.getVoidTy(), {ptr, ptr, ptr, ptr, i64, i64}, {0, 1, 2, 3}, {0, 2, 3});
    auto a = F->arg_begin();
    Value *gpad = a++, *gu = a++, *u = a++, *params = a++, *nt = a++, *nx = a++;
    B.SetInsertPoint(BasicBlock::Create(C, "entry", F));
    forRange2D(B, F, K(0), nt, K(0), nx, [&](Value *t, Value *x) {
      Value *acc = emitAdjointPoint(B, spec, bd, gpad, u, params, nt, nx, t, x, pidx);
      B.CreateStore(acc, B.CreateInBoundsGEP(f32, gu, B.CreateNSWAdd(B.CreateNSWMul(t, nx), x)));
    });
    B.CreateRetVoid();
  }

  // double pk_pgrad_<p>(const float* u, const float* g, const float* params, i64 nt, i64 nx)
  for (const auto &p : spec.physicsParams) {
    Function *F = declare(M, "pk_pgrad_" + p, f64, {ptr, ptr, ptr, i64, i64}, {0, 1, 2}, {0, 1, 2});
    auto a = F->arg_begin();
    Value *u = a++, *g = a++, *params = a++, *nt = a++, *nx = a++;
    B.SetInsertPoint(BasicBlock::Create(C, "entry", F));
    Value *accSlot = B.CreateAlloca(f64, nullptr, "acc");
    B.CreateStore(ConstantFP::get(f64, 0.0), accSlot);
    pk::E dp = pk::diffParam(spec.residual, p);
    forRange2D(B, F, K(bd.tLo), B.CreateNSWSub(nt, K(bd.tHi)), K(bd.xLo), B.CreateNSWSub(nx, K(bd.xHi)),
               [&](Value *t, Value *x) {
                 EmitCtx c{B, u, params, nt, nx, t, x, 0, 0, false, pidx, {}};
                 Value *d = emit(dp, c);
                 Value *gv = B.CreateLoad(f32, B.CreateInBoundsGEP(f32, g, B.CreateNSWAdd(B.CreateNSWMul(t, nx), x)));
                 Value *term = B.CreateFPExt(B.CreateFMul(d, gv), f64);
                 B.CreateStore(B.CreateFAdd(B.CreateLoad(f64, accSlot), term), accSlot);
               });
    B.CreateRet(B.CreateLoad(f64, accSlot));
  }
  verifyOrThrow(M);
}

// ---------------------------------------------------------------------------
// NVPTX module: one thread per grid point (x fastest-varying for coalescing)
// ---------------------------------------------------------------------------
void markKernel(Module &M, Function *F) {
  LLVMContext &C = M.getContext();
  NamedMDNode *md = M.getOrInsertNamedMetadata("nvvm.annotations");
  md->addOperand(MDNode::get(C, {ValueAsMetadata::get(F), MDString::get(C, "kernel"),
                                 ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), 1))}));
}

std::pair<Value *, Value *> threadCoords(IRBuilder<> &B, Module &M) {
  auto rd = [&](Intrinsic::ID id) {
    return B.CreateZExt(B.CreateCall(Intrinsic::getDeclaration(&M, id)), B.getInt64Ty());
  };
  Value *x = B.CreateNUWAdd(B.CreateNUWMul(rd(Intrinsic::nvvm_read_ptx_sreg_ctaid_x), rd(Intrinsic::nvvm_read_ptx_sreg_ntid_x)),
                            rd(Intrinsic::nvvm_read_ptx_sreg_tid_x), "x");
  Value *t = B.CreateNUWAdd(B.CreateNUWMul(rd(Intrinsic::nvvm_read_ptx_sreg_ctaid_y), rd(Intrinsic::nvvm_read_ptx_sreg_ntid_y)),
                            rd(Intrinsic::nvvm_read_ptx_sreg_tid_y), "t");
  return {t, x};
}

void buildPtxModule(Module &M, const pk::PDESpec &spec, const Bounds &bd, bool fast) {
  LLVMContext &C = M.getContext();
  IRBuilder<> B(C);
  B.setFastMathFlags(fmfFor(fast));
  Type *ptr = B.getPtrTy(), *i64 = B.getInt64Ty(), *f32 = B.getFloatTy();
  auto pidx = paramIndexOf(spec);
  auto K = [&](int v) { return ConstantInt::get(i64, v); };

  {
    Function *F = declare(M, "pk_fwd", B.getVoidTy(), {ptr, ptr, ptr, i64, i64}, {0, 1, 2}, {0, 2});
    markKernel(M, F);
    auto a = F->arg_begin();
    Value *u = a++, *res = a++, *params = a++, *nt = a++, *nx = a++;
    BasicBlock *entry = BasicBlock::Create(C, "entry", F);
    BasicBlock *body = BasicBlock::Create(C, "body", F);
    BasicBlock *done = BasicBlock::Create(C, "done", F);
    B.SetInsertPoint(entry);
    auto [t, x] = threadCoords(B, M);
    Value *in = B.CreateAnd(B.CreateAnd(B.CreateICmpSGE(t, K(bd.tLo)), B.CreateICmpSLT(t, B.CreateNSWSub(nt, K(bd.tHi)))),
                            B.CreateAnd(B.CreateICmpSGE(x, K(bd.xLo)), B.CreateICmpSLT(x, B.CreateNSWSub(nx, K(bd.xHi)))));
    B.CreateCondBr(in, body, done);
    B.SetInsertPoint(body);
    EmitCtx c{B, u, params, nt, nx, t, x, 0, 0, false, pidx, {}};
    Value *r = emit(spec.residual, c);
    B.CreateStore(r, B.CreateInBoundsGEP(f32, res, B.CreateNSWAdd(B.CreateNSWMul(t, nx), x)));
    B.CreateBr(done);
    B.SetInsertPoint(done);
    B.CreateRetVoid();
  }
  {
    Function *F = declare(M, "pk_adj", B.getVoidTy(), {ptr, ptr, ptr, ptr, i64, i64}, {0, 1, 2, 3}, {0, 2, 3});
    markKernel(M, F);
    auto a = F->arg_begin();
    Value *gpad = a++, *gu = a++, *u = a++, *params = a++, *nt = a++, *nx = a++;
    BasicBlock *entry = BasicBlock::Create(C, "entry", F);
    BasicBlock *body = BasicBlock::Create(C, "body", F);
    BasicBlock *done = BasicBlock::Create(C, "done", F);
    B.SetInsertPoint(entry);
    auto [t, x] = threadCoords(B, M);
    B.CreateCondBr(B.CreateAnd(B.CreateICmpSLT(t, nt), B.CreateICmpSLT(x, nx)), body, done);
    B.SetInsertPoint(body);
    Value *acc = emitAdjointPoint(B, spec, bd, gpad, u, params, nt, nx, t, x, pidx);
    B.CreateStore(acc, B.CreateInBoundsGEP(f32, gu, B.CreateNSWAdd(B.CreateNSWMul(t, nx), x)));
    B.CreateBr(done);
    B.SetInsertPoint(done);
    B.CreateRetVoid();
  }
  verifyOrThrow(M);
}

// ---------------------------------------------------------------------------
// Compiled CPU handle
// ---------------------------------------------------------------------------
using FwdFn = void (*)(const float *, float *, const float *, int64_t, int64_t);
using AdjFn = void (*)(const float *, float *, const float *, const float *, int64_t, int64_t);
using PGradFn = double (*)(const float *, const float *, const float *, int64_t, int64_t);

struct Handle {
  pk::PDESpec spec;
  Bounds bounds;
  int opt = 3;
  bool fast = false;
  std::unique_ptr<LLJIT> jit;
  FwdFn fwd = nullptr;
  AdjFn adj = nullptr;
  std::vector<PGradFn> pgrad;
  std::string irBefore, irAfter, exprText;
};

Handle *compileCpu(const std::string &pde, int opt, bool fast) {
  initTargets();
  auto h = std::make_unique<Handle>();
  h->spec = pk::makePDE(pde);
  h->bounds = boundsOf(h->spec.residual);
  h->opt = opt;
  h->fast = fast;
  h->exprText = pk::toString(h->spec.residual);

  auto JTMB = cantFail(JITTargetMachineBuilder::detectHost());
  JTMB.setCodeGenOptLevel(opt == 0 ? CodeGenOptLevel::None : CodeGenOptLevel::Aggressive);
  auto TM = cantFail(JTMB.createTargetMachine());

  auto ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("pk_" + pde + "_cpu", *ctx);
  M->setDataLayout(TM->createDataLayout());
  M->setTargetTriple(TM->getTargetTriple().str());
  buildCpuModule(*M, h->spec, h->bounds, fast);
  // Tag functions with the host CPU so the vectorizer's cost model sees AVX etc.
  for (Function &F : *M) {
    if (F.isDeclaration()) continue;
    F.addFnAttr("target-cpu", TM->getTargetCPU());
    F.addFnAttr("target-features", TM->getTargetFeatureString());
  }
  h->irBefore = moduleText(*M);
  runPipeline(*M, TM.get(), opt);
  verifyOrThrow(*M);
  h->irAfter = moduleText(*M);

  auto jit = LLJITBuilder().setJITTargetMachineBuilder(std::move(JTMB)).create();
  if (!jit) throw std::runtime_error(toString(jit.takeError()));
  h->jit = std::move(*jit);
  if (auto err = h->jit->addIRModule(ThreadSafeModule(std::move(M), std::move(ctx))))
    throw std::runtime_error(toString(std::move(err)));
  auto look = [&](const std::string &n) {
    auto s = h->jit->lookup(n);
    if (!s) throw std::runtime_error(toString(s.takeError()));
    return *s;
  };
  h->fwd = look("pk_fwd").toPtr<FwdFn>();
  h->adj = look("pk_adj").toPtr<AdjFn>();
  for (const auto &p : h->spec.physicsParams) h->pgrad.push_back(look("pk_pgrad_" + p).toPtr<PGradFn>());
  return h.release();
}

std::string emitPtx(const std::string &pde, const std::string &sm, bool fast, int opt) {
  initTargets();
  const std::string triple = "nvptx64-nvidia-cuda";
  std::string err;
  const Target *T = TargetRegistry::lookupTarget(triple, err);
  if (!T) throw std::runtime_error("NVPTX target unavailable: " + err);
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(triple, sm, "", TargetOptions(), std::nullopt,
                                                           std::nullopt, CodeGenOptLevel::Aggressive));
  if (!TM) throw std::runtime_error("could not create NVPTX target machine for " + sm);
  pk::PDESpec spec = pk::makePDE(pde);
  LLVMContext C;
  Module M("pk_" + pde + "_ptx", C);
  M.setDataLayout(TM->createDataLayout());
  M.setTargetTriple(triple);
  buildPtxModule(M, spec, boundsOf(spec.residual), fast);
  runPipeline(M, TM.get(), opt);
  verifyOrThrow(M);
  SmallString<0> buf;
  raw_svector_ostream os(buf);
  legacy::PassManager PM;
  if (TM->addPassesToEmitFile(PM, os, nullptr, CodeGenFileType::AssemblyFile))
    throw std::runtime_error("NVPTX backend cannot emit assembly");
  PM.run(M);
  return std::string(buf.str());
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI (consumed from Python via ctypes)
// ---------------------------------------------------------------------------
extern "C" {

const char *pk_last_error() { return g_error.c_str(); }

void *pk_compile(const char *pde, int opt, int fast) {
  try {
    return compileCpu(pde, opt, fast != 0);
  } catch (const std::exception &e) {
    g_error = e.what();
    return nullptr;
  }
}

void pk_free(void *h) { delete static_cast<Handle *>(h); }

int pk_num_params(void *h) { return static_cast<int>(static_cast<Handle *>(h)->spec.params.size()); }
const char *pk_param_name(void *h, int i) { return static_cast<Handle *>(h)->spec.params.at(i).c_str(); }
int pk_num_physics_params(void *h) { return static_cast<int>(static_cast<Handle *>(h)->spec.physicsParams.size()); }
const char *pk_physics_param_name(void *h, int i) { return static_cast<Handle *>(h)->spec.physicsParams.at(i).c_str(); }

void pk_bounds(void *h, int64_t *out) {
  const Bounds &b = static_cast<Handle *>(h)->bounds;
  out[0] = b.tLo; out[1] = b.tHi; out[2] = b.xLo; out[3] = b.xHi; out[4] = b.halo;
}

void pk_forward(void *h, const float *u, float *res, const float *params, int64_t nt, int64_t nx) {
  static_cast<Handle *>(h)->fwd(u, res, params, nt, nx);
}

void pk_adjoint(void *h, const float *gpad, float *gu, const float *u, const float *params, int64_t nt, int64_t nx) {
  static_cast<Handle *>(h)->adj(gpad, gu, u, params, nt, nx);
}

double pk_param_grad(void *h, int k, const float *u, const float *g, const float *params, int64_t nt, int64_t nx) {
  return static_cast<Handle *>(h)->pgrad.at(k)(u, g, params, nt, nx);
}

const char *pk_ir(void *h, int optimized) {
  auto *H = static_cast<Handle *>(h);
  return optimized ? H->irAfter.c_str() : H->irBefore.c_str();
}

const char *pk_residual_expr(void *h) { return static_cast<Handle *>(h)->exprText.c_str(); }

const char *pk_emit_ptx(const char *pde, const char *sm, int fast, int opt) {
  try {
    g_text = emitPtx(pde, sm, fast != 0, opt);
    return g_text.c_str();
  } catch (const std::exception &e) {
    g_error = e.what();
    return nullptr;
  }
}

}  // extern "C"
