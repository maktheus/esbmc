/// \file
/// Float Interval Analysis Implementation for ESBMC
///
/// See float_interval_analysis.h for the full design rationale.

#include "float_interval_analysis.h"

#include <goto-programs/goto_loops.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>
#include <util/time_stopping.h>

#include <algorithm>
#include <unordered_map>

// ===========================================================================
// Internal: extract float bounds from ASSUME instructions
// ===========================================================================
// We look for patterns of the form:
//   __ESBMC_assume(sym >= LB && sym <= UB);
// which the C harness generates from:
//   __ESBMC_assume(base_token[i] >= -1.0f && base_token[i] <= 1.0f);
//
// This pattern is what the DeepSeek MoE harness emits for its nondet inputs.

/// @brief Returns true if expr is a float-type symbol expression
static bool is_float_symbol(const expr2tc &expr)
{
  return is_symbol2t(expr) && is_floatbv_type(expr->type);
}

/// @brief Tries to extract [lower, upper] bounds from a conjunction of
///   (sym >= L) AND (sym <= U) style float comparisons.
///   Returns true if a clean bound was extracted.
static bool try_extract_float_bound(
  const expr2tc &guard,
  std::string &out_symbol_name,
  float &out_lower,
  float &out_upper)
{
  // We expect: AND( GTE(sym, L), LTE(sym, U) )
  if (!is_and2t(guard))
    return false;

  const and2t &conjunction = to_and2t(guard);
  const expr2tc &left = conjunction.side_1;
  const expr2tc &right = conjunction.side_2;

  // Left side: sym >= lower
  if (!is_greaterthanequal2t(left))
    return false;
  const greaterthanequal2t &gte = to_greaterthanequal2t(left);
  if (!is_float_symbol(gte.side_1) || !is_constant_floatbv2t(gte.side_2))
    return false;

  // Right side: sym <= upper
  if (!is_lessthanequal2t(right))
    return false;
  const lessthanequal2t &lte = to_lessthanequal2t(right);
  if (!is_float_symbol(lte.side_1) || !is_constant_floatbv2t(lte.side_2))
    return false;

  // Confirm both sides refer to the same symbol
  const symbol2t &sym_gte = to_symbol2t(gte.side_1);
  const symbol2t &sym_lte = to_symbol2t(lte.side_1);
  if (sym_gte.thename != sym_lte.thename)
    return false;

  out_symbol_name = sym_gte.thename.as_string();
  out_lower =
    (float)to_constant_floatbv2t(gte.side_2).value.to_double();
  out_upper =
    (float)to_constant_floatbv2t(lte.side_2).value.to_double();
  return true;
}

// ===========================================================================
// Internal: inject simplified ASSUME intervals back into the GOTO program
// ===========================================================================
// After collecting intervals from ASSUME guards, we inject concrete "assume"
// statements that narrow the nondet_float range. This reduces the number of
// FPA VCCs sent down to the SMT solver (from millions to O(N) intervals).

using float_interval_map_t = std::unordered_map<std::string, float_interval_t>;

/// @brief Scans a single goto_function for ASSUME instructions that constrain
///   float symbols. Populates or updates the given interval map.
static void collect_float_intervals(
  const goto_functiont &goto_function,
  float_interval_map_t &interval_map)
{
  forall_goto_program_instructions(i_it, goto_function.body)
  {
    if (!i_it->is_assume())
      continue;

    std::string sym_name;
    float lb, ub;
    if (try_extract_float_bound(i_it->guard, sym_name, lb, ub))
    {
      auto it = interval_map.find(sym_name);
      if (it == interval_map.end())
        interval_map.emplace(sym_name, float_interval_t(lb, ub));
      else
        it->second = it->second.join(float_interval_t(lb, ub));
    }
  }
}

/// @brief For each nondet_float that has a known interval from the map,
///   insert a paired ASSUME constraint right after its assignment.
///   This makes the SMT encoding context-aware, reducing blind nondet ranges.
static void instrument_float_intervals(
  goto_functiont &goto_function,
  const float_interval_map_t &interval_map,
  const namespacet &ns)
{
  if (interval_map.empty())
    return;

  Forall_goto_program_instructions(i_it, goto_function.body)
  {
    if (!i_it->is_assign())
      continue;

    const code_assign2t &assign = to_code_assign2t(i_it->code);
    if (!is_float_symbol(assign.target))
      continue;

    const std::string sym_name = to_symbol2t(assign.target).thename.as_string();
    auto it = interval_map.find(sym_name);
    if (it == interval_map.end())
      continue;

    const float_interval_t &interval = it->second;
    if (interval.lower_is_inf || interval.upper_is_inf || interval.is_bottom)
      continue;

    // Build: ASSUME (sym >= lower && sym <= upper)
    // ieee_floatt must be constructed with ieee_float_spect to match the
    // internal ESBMC type spec (see gen_one() pattern in irep2_utils.h:465).
    const expr2tc &sym = assign.target;
    ieee_float_spect fspec(to_floatbv_type(sym->type));

    ieee_floatt lower_ieee(fspec);
    lower_ieee.from_double((double)interval.lower);
    ieee_floatt upper_ieee(fspec);
    upper_ieee.from_double((double)interval.upper);

    expr2tc lower_const = constant_floatbv2tc(lower_ieee);
    expr2tc upper_const = constant_floatbv2tc(upper_ieee);

    expr2tc gte = greaterthanequal2tc(sym, lower_const);
    expr2tc lte = lessthanequal2tc(sym, upper_const);
    expr2tc bounds = and2tc(gte, lte);

    goto_programt::instructiont assume_instr;
    assume_instr.make_assumption(bounds);
    assume_instr.location = i_it->location;
    assume_instr.function = i_it->function;

    // Insert the assumption right AFTER the assignment
    auto insert_pos = i_it;
    ++insert_pos;
    goto_function.body.insert_swap(insert_pos, assume_instr);
  }
}

// ===========================================================================
// Public API
// ===========================================================================

void float_interval_analysis(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
  fine_timet t_start = current_time();
  log_status("Float Interval Analysis: starting pass for FPA tensor support.");

  // Phase 1: Collect float bounds from all ASSUME guards
  float_interval_map_t global_intervals;
  Forall_goto_functions(f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;
    collect_float_intervals(f_it->second, global_intervals);
  }

  log_status(
    "Float Interval Analysis: found {} float symbol bounds.",
    global_intervals.size());

  // Phase 2: Instrument assignments with concrete interval assumptions
  Forall_goto_functions(f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;
    instrument_float_intervals(f_it->second, global_intervals, ns);
  }

  goto_functions.update();

  fine_timet t_stop = current_time();
  log_status(
    "Float Interval Analysis: done in {}s.",
    time2string(t_stop - t_start));
}
