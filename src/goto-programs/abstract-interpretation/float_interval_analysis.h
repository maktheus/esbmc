/// \file
/// Float Interval Analysis for ESBMC
///
/// This module extends the existing interval_analysis framework to handle
/// IEEE-754 floating-point (FPA) types. It was created to support formal
/// verification of neural network tensor operations (e.g., DeepSeek MoE
/// router dot products) without triggering State Explosion or Core Dumps
/// when the Z3 FPA backend is invoked on continuous nondet float expressions.
///
/// Root Cause of the Problem:
/// The existing interval_analysis.cpp (line 15-18) explicitly returns early
/// for float types: "Only integers for now". This means the ESBMC Symbolic
/// Execution engine cannot simplify/propagate Float32 singleton intervals
/// before encoding them for the SMT solver. Every `nondet_float()` variable
/// inside `dot_product` loops gets forwarded raw to Z3 as a full FPA formula,
/// causing the bit-blasting explosion and the observed "IOT instruction / Core
/// Dump" when the Z3 v4.8.x FPA module hits its internal assertion limit.
///
/// This Module's Solution:
/// By registering interval bounds for float variables during the pre-symex
/// analysis pass (using `__ESBMC_assume(x >= -1.0f && x <= 1.0f)` as the
/// seed), the Symex engine can propagate bound information and reduce the
/// number of FPA VCCs emitted to the solver from millions to a handful.

#ifndef ESBMC_FLOAT_INTERVAL_ANALYSIS_H
#define ESBMC_FLOAT_INTERVAL_ANALYSIS_H

#include <goto-programs/goto_functions.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>

/// A simple interval representation for single-precision (float) values.
/// Uses lower/upper bounds that can be -inf or +inf (represented as NaN).
struct float_interval_t
{
  bool is_bottom = false; // Empty set (unreachable)
  bool lower_is_inf = true;
  bool upper_is_inf = true;
  float lower = 0.0f;
  float upper = 0.0f;

  /// Constructs a bounded interval [l, u]
  float_interval_t(float l, float u)
    : is_bottom(false), lower_is_inf(false), upper_is_inf(false), lower(l),
      upper(u)
  {
  }

  /// Constructs the top interval (all possible floats)
  float_interval_t()
    : is_bottom(false), lower_is_inf(true), upper_is_inf(true)
  {
  }

  /// Returns true if this interval is a single value (singleton)
  bool singleton() const
  {
    return !lower_is_inf && !upper_is_inf && (lower == upper);
  }

  /// Returns true if a given value is contained in this interval
  bool contains(float v) const
  {
    if (is_bottom)
      return false;
    if (!lower_is_inf && v < lower)
      return false;
    if (!upper_is_inf && v > upper)
      return false;
    return true;
  }

  /// Join (widening): produces an interval covering both
  float_interval_t join(const float_interval_t &other) const
  {
    if (is_bottom)
      return other;
    if (other.is_bottom)
      return *this;

    float_interval_t result;
    result.lower_is_inf = lower_is_inf || other.lower_is_inf;
    result.upper_is_inf = upper_is_inf || other.upper_is_inf;

    if (!result.lower_is_inf)
      result.lower = std::min(lower, other.lower);
    if (!result.upper_is_inf)
      result.upper = std::max(upper, other.upper);

    return result;
  }
};

/// Main entry point: runs a float-aware interval analysis pass over the
/// provided goto_functions before symbolic execution begins.
///
/// When ESBMC processes a function like dot_product with `nondet_float()`
/// variables constrained by `__ESBMC_assume(x >= -1.0f && x <= 1.0f)`,
/// this pass extracts those assume constraints and propagates them as
/// float_interval_t entries. The result is a lighter VCC formula sent to Z3.
///
/// @param goto_functions   The full GOTO program being verified
/// @param ns               The namespace (symbol table)
/// @param options          ESBMC option flags (used for verbosity, etc.)
void float_interval_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options);

#endif // ESBMC_FLOAT_INTERVAL_ANALYSIS_H
