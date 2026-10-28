// Stencil expression graph (DAG) for discretized PDE residuals.
//
// A residual is written once as an expression over stencil loads
// u[t + dt][x + dx], named scalar parameters, and constants. From that single
// description the code generator emits:
//   * the forward residual kernel,
//   * the adjoint (vector-Jacobian product) kernel, derived here by symbolic
//     differentiation with respect to every stencil load, and
//   * one parameter-gradient reduction kernel per physical parameter
//     (e.g. dL/dalpha for the inverse problem).
#pragma once

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pk {

enum class Kind { Load, Param, Const, Add, Sub, Mul, Div };

struct Expr;
using E = std::shared_ptr<const Expr>;

struct Expr {
  Kind kind = Kind::Const;
  int dt = 0, dx = 0;   // Load: stencil offset in time / space
  std::string name;     // Param: parameter name
  float value = 0.f;    // Const: value
  E a, b;               // binary operands
};

inline E load(int dt, int dx) {
  auto e = std::make_shared<Expr>();
  e->kind = Kind::Load; e->dt = dt; e->dx = dx;
  return e;
}
inline E param(const std::string &n) {
  auto e = std::make_shared<Expr>();
  e->kind = Kind::Param; e->name = n;
  return e;
}
inline E cst(float v) {
  auto e = std::make_shared<Expr>();
  e->kind = Kind::Const; e->value = v;
  return e;
}

inline bool isConst(const E &e, float v) { return e->kind == Kind::Const && e->value == v; }
inline bool isConst(const E &e) { return e->kind == Kind::Const; }

// Build a binary node with light algebraic simplification. Simplification
// matters for the adjoint: without it, d/du of a stencil expression is a large
// tree of "0 * x + 1 * y" terms.
inline E bin(Kind k, E a, E b) {
  if (isConst(a) && isConst(b)) {
    float x = a->value, y = b->value;
    switch (k) {
      case Kind::Add: return cst(x + y);
      case Kind::Sub: return cst(x - y);
      case Kind::Mul: return cst(x * y);
      case Kind::Div: return cst(x / y);
      default: break;
    }
  }
  switch (k) {
    case Kind::Add:
      if (isConst(a, 0.f)) return b;
      if (isConst(b, 0.f)) return a;
      break;
    case Kind::Sub:
      if (isConst(b, 0.f)) return a;
      break;
    case Kind::Mul:
      if (isConst(a, 0.f) || isConst(b, 0.f)) return cst(0.f);
      if (isConst(a, 1.f)) return b;
      if (isConst(b, 1.f)) return a;
      break;
    case Kind::Div:
      if (isConst(a, 0.f)) return cst(0.f);
      if (isConst(b, 1.f)) return a;
      break;
    default: break;
  }
  auto e = std::make_shared<Expr>();
  e->kind = k; e->a = std::move(a); e->b = std::move(b);
  return e;
}

inline E operator+(E a, E b) { return bin(Kind::Add, std::move(a), std::move(b)); }
inline E operator-(E a, E b) { return bin(Kind::Sub, std::move(a), std::move(b)); }
inline E operator*(E a, E b) { return bin(Kind::Mul, std::move(a), std::move(b)); }
inline E operator/(E a, E b) { return bin(Kind::Div, std::move(a), std::move(b)); }
inline E operator*(float s, E b) { return cst(s) * std::move(b); }

// Generic reverse-mode-style symbolic derivative with respect to a leaf
// selected by `isTarget`.
template <class Pred>
E diff(const E &e, const Pred &isTarget) {
  switch (e->kind) {
    case Kind::Load:
    case Kind::Param:
      return cst(isTarget(*e) ? 1.f : 0.f);
    case Kind::Const:
      return cst(0.f);
    case Kind::Add: return diff(e->a, isTarget) + diff(e->b, isTarget);
    case Kind::Sub: return diff(e->a, isTarget) - diff(e->b, isTarget);
    case Kind::Mul:
      return diff(e->a, isTarget) * e->b + e->a * diff(e->b, isTarget);
    case Kind::Div: {
      E da = diff(e->a, isTarget), db = diff(e->b, isTarget);
      if (isConst(db, 0.f)) return da / e->b;
      return (da * e->b - e->a * db) / (e->b * e->b);
    }
  }
  throw std::logic_error("diff: unknown node kind");
}

inline E diffLoad(const E &e, int dt, int dx) {
  return diff(e, [&](const Expr &n) { return n.kind == Kind::Load && n.dt == dt && n.dx == dx; });
}
inline E diffParam(const E &e, const std::string &name) {
  return diff(e, [&](const Expr &n) { return n.kind == Kind::Param && n.name == name; });
}

inline void collectLoads(const E &e, std::set<std::pair<int, int>> &out) {
  if (!e) return;
  if (e->kind == Kind::Load) out.insert({e->dt, e->dx});
  collectLoads(e->a, out);
  collectLoads(e->b, out);
}
inline bool hasLoad(const E &e) {
  std::set<std::pair<int, int>> s;
  collectLoads(e, s);
  return !s.empty();
}

inline std::string toString(const E &e) {
  std::ostringstream os;
  switch (e->kind) {
    case Kind::Load: os << "u[t" << (e->dt >= 0 ? "+" : "") << e->dt << ",x" << (e->dx >= 0 ? "+" : "") << e->dx << "]"; break;
    case Kind::Param: os << e->name; break;
    case Kind::Const: os << e->value; break;
    default: {
      const char *op = e->kind == Kind::Add ? " + " : e->kind == Kind::Sub ? " - " : e->kind == Kind::Mul ? " * " : " / ";
      os << "(" << toString(e->a) << op << toString(e->b) << ")";
    }
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// PDE library. Parameter vector layout passed to every kernel: [dt, dx, phys...]
// ---------------------------------------------------------------------------
struct PDESpec {
  std::string name;
  std::vector<std::string> params;         // full layout, dt and dx first
  std::vector<std::string> physicsParams;  // learnable / differentiable ones
  E residual;
};

inline PDESpec makeHeat() {
  // u_t - alpha * u_xx  (forward Euler in time, central in space)
  E dt = param("dt"), dx = param("dx"), alpha = param("alpha");
  E uc = load(0, 0), ur = load(0, 1), ul = load(0, -1), uf = load(1, 0);
  E u_t = (uf - uc) / dt;
  E u_xx = (ur - 2.f * uc + ul) / (dx * dx);
  return {"heat", {"dt", "dx", "alpha"}, {"alpha"}, u_t - alpha * u_xx};
}

inline PDESpec makeBurgers() {
  // u_t + u * u_x - nu * u_xx  (nonlinear: exercises the clamped adjoint path)
  E dt = param("dt"), dx = param("dx"), nu = param("nu");
  E uc = load(0, 0), ur = load(0, 1), ul = load(0, -1), uf = load(1, 0);
  E u_t = (uf - uc) / dt;
  E u_x = (ur - ul) / (2.f * dx);
  E u_xx = (ur - 2.f * uc + ul) / (dx * dx);
  return {"burgers", {"dt", "dx", "nu"}, {"nu"}, u_t + uc * u_x - nu * u_xx};
}

inline PDESpec makePDE(const std::string &name) {
  if (name == "heat") return makeHeat();
  if (name == "burgers") return makeBurgers();
  throw std::invalid_argument("unknown PDE '" + name + "' (expected heat or burgers)");
}

}  // namespace pk
