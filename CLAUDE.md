# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

ESBMC (Efficient SMT-based Context-Bounded Model Checker) — a mature C++
software model checker (the codebase itself is built with `CMAKE_CXX_STANDARD 23`,
set in `scripts/cmake/Options.cmake`) that automatically detects runtime errors, or proves
their absence, in single- and multi-threaded programs. Version 8.0.0
(`CMakeLists.txt`).

Supported input languages: C, C++, CUDA, CHERI-C, Java/Kotlin (via Jimple),
Python, Solidity, and Rust (via GOTO-Transcoder).

This checkout is the **`maktheus/esbmc` fork** of upstream `esbmc/esbmc`. It
carries the full upstream tree plus a fork-specific research directory,
`pibic/` (see [Fork-specific content](#fork-specific-content)). `origin` points
at the fork; there is no `upstream` remote configured by default.

## Verification pipeline

Understanding this pipeline is the key to navigating the source tree. Each
stage maps to one or more directories under `src/`:

```
source code → AST → GOTO program → symbolic execution → SSA → SMT formula → solver
   (frontends)      (goto-programs)   (goto-symex)              (solvers)
```

1. **Frontend** parses source into an AST and converts it into ESBMC's symbol
   table / `irep` representation.
2. **GOTO conversion** (`goto-programs`) normalizes control flow into a state
   transition system, then runs analysis/instrumentation passes (`goto_check`,
   loop unrolling, k-induction transforms, slicing, coverage instrumentation).
3. **Symbolic execution** (`goto-symex`) unrolls loops/recursion up to bounds,
   explores thread interleavings, and emits SSA (single static assignment)
   steps into a `symex_target_equation`.
4. **SMT encoding** (`solvers`) converts the SSA equation plus the negated
   properties into an SMT formula. SAT ⇒ a property violation exists (a
   counterexample trace is reconstructed); UNSAT ⇒ no violation within bounds.

`ARCHITECTURE.md` has the prose version with a diagram.

## Repository layout

```
src/
  esbmc/              CLI entry point: main.cpp, esbmc_parseoptions.cpp,
                      options.cpp (the full CLI option table), bmc.cpp
                      (BMC/k-induction/incremental driver loop)
  util/               Core infrastructure (~164 files): irep, symbol table,
                      types, expressions, config, cmdline, message/logging,
                      simplification, migration between irep and irep2
  irep2/              The modern, typed IR (expr2tc / type2tc). Expression and
                      type kinds are enumerated in ESBMC_LIST_OF_EXPRS /
                      ESBMC_LIST_OF_TYPES boost-preprocessor lists in irep2.h
  clang-c-frontend/   C / CUDA / CHERI-C frontend (uses libclang AST)
  clang-cpp-frontend/ C++ frontend
  python-frontend/    Python frontend (CPython ast → JSON → type annotation →
                      irep). See its README.md; parser.py / preprocessor.py are
                      shipped Python helpers
  solidity-frontend/  Solidity frontend (optional: ENABLE_SOLIDITY_FRONTEND)
  jimple-frontend/    Java/Kotlin via Soot Jimple (optional)
  goto2c/             GOTO → C back-translation
  goto-programs/      GOTO IR, conversion, checks, contracts, abstract
                      interpretation, loop handling, coverage, serialization
  goto-symex/         Symbolic execution engine, reachability tree
                      (interleavings), SSA equation, slicing, goto traces,
                      witness/HTML/JSON/XML output
  pointer-analysis/   Value-set / dereference analysis
  solvers/            SMT backends: z3, bitwuzla, boolector, cvc4, cvc5,
                      mathsat, yices, smtlib (generic pipe), minisat/sat/prop.
                      solve.cpp dispatches by solver name
  langapi/            Language-registration layer (mode.cpp maps file
                      extensions/modes to frontends)
  c2goto/             The c2goto compiler + operational models of libc/pthreads
                      under library/, and ESBMC-specific headers under headers/
  big-int/            Arbitrary precision integers
unit/                 Catch2 unit tests, mirroring src/ subdirectory names
regression/           End-to-end test suites (one directory per test)
scripts/              build.sh, build.ps1, cmake/ modules, benchexec, csmith
website/              Hugo site; user + developer docs in content/docs/
docs/                 LaTeX developer manual (manual.tex) and examples
pibic/                Fork-specific AI-verification research (see below)
```

## Building

The canonical build is the script used by CI:

```sh
./scripts/build.sh -b Debug        # from the top of the source tree
```

It installs OS dependencies, downloads solvers, configures CMake in `build/`
(which must not already exist), builds with Ninja, and installs into
`./release`. Useful flags (`./scripts/build.sh -h` for all):

| Flag | Meaning |
| --- | --- |
| `-b BTYPE` | CMake build type: `Debug`, `Release`, `RelWithDebInfo` (default), `DebugOpt`, `Sanitizer` |
| `-e ON\|OFF` | `-Werror` (CI PR builds use `-e On`) |
| `-s STYPE` | Enable sanitizer, compile with clang |
| `-S ON\|OFF` | Static build (default ON on Ubuntu, OFF on macOS) |
| `-c VERS` | LLVM/Clang major version for shared Ubuntu builds |
| `-C` | SV-COMP configuration |
| `-B ON\|OFF` | Bundle ESBMC's libc |

For an incremental developer build (fastest iteration, assumes deps present):

```sh
mkdir -p build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug \
  -DDOWNLOAD_DEPENDENCIES=On -DENABLE_Z3=On \
  -DBUILD_TESTING=On -DENABLE_REGRESSION=On \
  -DENABLE_PYTHON_FRONTEND=On -DENABLE_SOLIDITY_FRONTEND=On \
  -DENABLE_JIMPLE_FRONTEND=On
ninja
```

Notes:
- `CMAKE_EXPORT_COMPILE_COMMANDS` is ON, so `build/compile_commands.json`
  exists for clangd/clang-tidy.
- At least one solver must be enabled; only `smtlib` is available by default.
- `-DBUILD_TESTING=On` is required for `unit/`; `-DENABLE_REGRESSION=1` is
  required for `regression/`.
- Frontends are opt-in via `ENABLE_*_FRONTEND`; disabling one removes both the
  `src/` subdirectory and its unit tests from the build.
- The build is large. Prefer building a single target when iterating, and never
  commit the `build/` or `release/` directories (both are gitignored).

## Testing

### Regression tests (end-to-end, the primary suite)

Each test is a directory under `regression/<suite>/<test>/` containing the
input program plus a `test.desc` file with this exact line-oriented format:

```
CORE                                    # test mode
main.c                                  # input file, relative to the test dir
--no-bounds-check --unwind 5            # arguments passed to esbmc (may be empty)
^VERIFICATION SUCCESSFUL$               # one or more regexes matched against output
```

Test modes: `CORE` (fast, essential), `THOROUGH` (slow), `KNOWNBUG` (expected
to fail due to a bug), `FUTURE` (expected to fail, unimplemented), `ALL`.

Run via ctest from the build directory; every test is labelled
`regression;<suite>`:

```sh
ctest -j4 -L esbmc-cpp/cpp        # one suite
ctest -L 'esbmc-cpp/.*'           # matching suites
ctest -LE esbmc-cpp               # everything except a suite
ctest -j4 -L python --timeout 30 --progress
ctest -R regression/esbmc/math_mod01   # a single test by name
```

The runner is `regression/testing_tool.py`; it can also be invoked directly
with `--tool`, `--regression`, `--modes`, `--file`. `regression/README.md` has
more examples. The Python suite has a helper: `./scripts/check_python_tests.sh`
(run from the repo root).

**Every PR that changes behaviour should add at least two regression tests: one
that passes and one that fails** — this is an explicit project convention
(`CONTRIBUTIONS.md`).

### Unit tests

Catch2 v2.13.7 (fetched by CMake), one directory under `unit/` per `src/`
module, files named `*.test.cpp`. Tests are registered by
`catch_discover_tests`, so ctest sees individual `SCENARIO`s:

```sh
ctest -j4 -R irept          # from build/
./unit/util/<test-binary>   # run a test binary directly
```

Shared helpers live in `unit/testing-utils/` (`goto_factory` builds GOTO
programs from source strings for testing).

## Code conventions

- **Formatting is enforced by `.clang-format`** (upstream uses Clang 11 for the
  check): Allman braces, 2-space indent, 80-column limit, pointer alignment
  right, no single-line functions/ifs/loops. Run
  `clang-format -i <files>` on anything you touch, and format only the lines
  you changed — do not reformat unrelated code.
- Python in this repo follows `.style.yapf`.
- **Two IRs coexist.** Legacy `irept`/`exprt`/`typet` (in `util/`) is used by
  frontends and the symbol table; modern `irep2` (`expr2tc`, `type2tc`,
  reference-counted, typed) is used from GOTO conversion onward.
  `util/migrate.h` converts between them. New backend code should prefer
  `irep2`.
- **Logging goes through the macros in `src/util/message.h`**: `log_error`,
  `log_warning`, `log_status`, `log_result`, `log_progress`, `log_success`,
  `log_fail`, and `log_debug(module, fmt, ...)`. They are fmt-style. Do not add
  raw `std::cout`/`printf` to library code.
- Headers are included with paths relative to `src/`
  (`#include <goto-programs/goto_functions.h>`), not relative paths.
- Doxygen comments (`/// \brief`, `\param`) are used on newer APIs; keep the
  density of the surrounding file.

## Common tasks

**Add a CLI option** — add an entry to the appropriate group in the
`all_cmd_options[]` table in `src/esbmc/options.cpp`, then read it where needed
(usually `esbmc_parseoptionst::get_command_line_options` in
`src/esbmc/esbmc_parseoptions.cpp`, which copies cmdline flags into `optionst`).
Document it in `website/content/docs/` if user-facing.

**Add or fix an operational model** for a libc/pthread function — the models
live in `src/c2goto/library/*.c` with ESBMC-specific headers in
`src/c2goto/headers/`. They are compiled to GOTO binaries by the `c2goto` tool
at build time and bundled into the ESBMC binary. Some functions are instead
handled as builtins in `src/goto-programs/builtin_functions.cpp` — check there
first. See `src/c2goto/README` and
`website/content/docs/development/om/`.

**Add a new expression or type kind** — extend the boost-preprocessor lists in
`src/irep2/irep2.h` and the corresponding template instantiations; there is a
step-by-step guide at
`website/content/docs/development/Adding-new-expressions.md`. Every switch over
expression kinds across the codebase may need updating.

**Integrate a new SMT solver** — add a subdirectory under `src/solvers/`
implementing `smt_convt` (plus optional `array_iface`, `tuple_iface`,
`fp_convt`), register it in `src/solvers/CMakeLists.txt` and `solve.cpp`. See
`website/content/docs/development/smt/`.

**Add a language frontend** — register the mode/extension in
`src/langapi/mode.cpp` and provide a `languaget` implementation; guard it
behind an `ENABLE_*_FRONTEND` CMake option as the existing optional frontends do.

## CI

GitHub Actions in `.github/workflows/`:

- `pull_request.yml` — the PR gate. Runs the testing-tool self-test
  (`regression/testing_tool_test.py`), builds the LaTeX developer manual, and
  builds + tests on Ubuntu 22.04 (static, LLVM 21, DebugOpt, `-Werror`),
  Windows, ARM64 Linux, and macOS ARM.
- `build-unix.yml` / `build-windows.yml` / `build.yml` — reusable build jobs
  driven by `scripts/build.sh` flags.
- `benchexec*.yml`, `benchbringup.yml` — SV-COMP style benchmarking.
- `release.yml`, `pages.yml` (Hugo website), `esbmc-stats.yml`.
- `esbmc-verify.yml`, `deploy-cartpole.yml` — **fork-specific**, driving the
  `pibic/` research code (pytest suite and a cart-pole visualization deploy).

Expect the PR build to be slow; a failure in the `-Werror` job usually means a
new compiler warning rather than a real test failure.

## Fork-specific content

`pibic/` is undergraduate-research (PIBIC) work layered on top of ESBMC, mostly
written in Portuguese. It is a **separate Python project**, not part of the C++
build, and nothing under `src/`, `unit/`, or `regression/` depends on it.

- `pibic/core_verify/` — Python wrapper around the ESBMC binary
  (`esbmc_caller.py`, `SMT_feedback_parser.py`); packaged via
  `pibic/pyproject.toml`.
- `pibic/cartpole/` — DQN/DDPG cart-pole controllers (PyTorch/ONNX), scripts
  that extract network weights into C and verify closed-loop safety properties
  with ESBMC, plus a web app visualization.
- `pibic/tests/` — pytest suite (`pytest tests/ -v` from `pibic/`).
- `pibic/1_python_models/`, `2_inference_engine/`, `3_neuro_symbolic/`,
  `4_control_system/` — the four case studies; `roadmap.md`, `qnn_report.md`,
  and `esbmc_genai_guide.md` document them.

Requirements: `pip install -r pibic/requirements.txt` (pytest, torch, onnx,
numpy), Python ≥ 3.10.

Treat changes to `pibic/` and changes to the ESBMC core as separate concerns:
core changes must keep the upstream regression suites green, while `pibic/`
changes only need the pytest suite.

## Gotchas

- The build tree must be out-of-source; `scripts/build.sh` refuses to run if
  `build/` already exists.
- The repo root contains some stray committed artifacts (`case2_output.txt`,
  `generated_code.c`, `nn_desconto_esbmc.c`, `chatbot_hibrido.py`). They are
  fork leftovers, not part of the build — don't treat them as reference code.
- Regression tests are inherently slow and solver-dependent; the same test can
  pass with Z3 and time out with another backend. Always state which solver a
  reported result came from.
- `KNOWNBUG` and `FUTURE` tests are *expected* to fail; ctest treats return
  code 10 as "skip". Don't "fix" them by changing the expected output — fix the
  underlying issue and change the mode to `CORE`.
- Documentation belongs in `website/content/docs/`, organized by the rules in
  `CONTRIBUTIONS.md` (usage at the root, development docs under
  `development/`, language-specific under `topic/`, theory under `theory/`).
