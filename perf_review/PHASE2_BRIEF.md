# Phase 2 brief (shared by all implementation streams)

Phase 1 profiled the package and wrote `perf_review/PHASE1_BOTTLENECKS.md`; read the entries assigned to your stream there. Phase 2 analyses each suggested fix (is it dangerous, does it make things faster), implements the worthwhile ones, and benchmarks them.

## Workspace
- Each stream works only in its own git worktree under `C:\Users\svens\Projects\FHS-perf-wt\<stream>`, which is on branch `perf/<stream>`. The worktree starts at commit 660c67f4.
- Never edit the main checkout `C:\Users\svens\Projects\FermionicHilbertSpaces.jl`. It stays at the base commit and is used as the **baseline**.
- NonCommutativeProducts (NCP) changes go only in `C:\Users\svens\Projects\FHS-perf-wt\ncp-symbolic`, on branch `perf/ncp-symbolic`. Never touch `C:\Users\svens\Projects\NonCommutativeProducts.jl`, which has uncommitted user work.
- Commits:
  - Commit each accepted fix separately on your stream branch.
  - Never commit to `main` or `perf-review`, never push, and never rewrite history outside your branch.
  - End every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Scratch files go in `C:\Users\svens\AppData\Local\Temp\claude\c--Users-svens-Projects-FermionicHilbertSpaces-jl\f27290bf-fcfe-4dfb-b0e3-3e1ec0e36974\scratchpad\phase2\<stream>\`.
- Run Julia through the Bash tool. Do **not** use the kaimon `ex` tool, because it is a REPL shared live with the user.
- Phase 1 scratch experiments, including working prototype code for many fixes, are under `...\scratchpad\phase1\<area>\`. Reuse them.

## Environments
- **Baseline env:** `...\scratchpad\phase1\benchenv` devs the main checkout (base code) plus the local NCP.
  - It needs `--startup-file=no`.
  - For JET, stack `JULIA_LOAD_PATH="@;...\scratchpad\phase1\toolsenv;@stdlib"`.
- **Your env:**
  - Copy `benchenv\Project.toml` to `...\scratchpad\phase2\<stream>\env\`.
  - In its `[sources]`, point `FermionicHilbertSpaces` at your worktree, and `NonCommutativeProducts` at the ncp-symbolic worktree only if your stream changes NCP.
  - Then run `julia --startup-file=no --project=<env> -e "using Pkg; Pkg.instantiate()"`.
- **Tests:** run from your worktree root, using the test env. `test/Manifest.toml` has already been copied into each worktree.
  ```
  julia --startup-file=no --project=test -e 'push!(LOAD_PATH, pwd()); using TestItemRunner; TestItemRunner.run_tests("."; filter=ti->occursin("permutation", ti.filename), verbose=false)'
  ```
  - The tests are `@testitem`s in `src/` and `test/runtests.jl`.
  - The full suite (`TestItemRunner.run_tests(".")`) takes about 9 minutes. Run it once at the end of your stream, and redirect its output to a file.
  - Do not use `@run_package_tests` with `-e`.
  - Check in the output that the worktree's code is loaded (`pathof(FermionicHilbertSpaces)`).

## Analysis requirements (before implementing)
Write, for each assigned item:
1. Is it correct for all inputs, not just the benchmark? Consider edge cases, other space types (fermions, Majoranas, spins, bosons, product, constrained, sector spaces), `BigInt` Fock numbers, generic arrays, threading (the OhMyThreads extension), and floating-point order.
2. Does it change public behaviour, return types or field types?
3. Expected gain and breadth.
4. Decision: implement, implement a safer variant, defer (needs a design decision by the maintainer), or reject.

## Implementation rules (user style: these are mandatory)
- **No new `@inbounds`.** Removing an existing unjustified one is welcome.
- **Do not over-restrict argument types.** Annotate only for dispatch.
- **Parametric constructors:** use unconstrained inner-constructor value arguments, and outer constructors that delegate in a cascade.
- **Comments and docstrings** describe the code as it is now. They contain no history ("previously", "was slow", "Phase 1", "fix B04") and no plan. Commit messages must also stand on their own for a reader who has only the repository. The `perf_review/` notes are the place for the narrative.
- Match the surrounding code style and comment density.
- Add a regression test for every correctness bug fixed, and for any non-obvious invariant a fix relies on. Use a `@testitem` in the relevant `src/` file, next to the existing ones.

## Benchmarking protocol (the machine is shared and noisy)
- Other streams run at the same time. Noise is bimodal, about ±40% on medians.
- Compare **minimum** times and **allocations/memory**. The latter are deterministic.
- For each fix, run the same standalone benchmark script against the baseline env and your env, alternating, for at least 3 rounds each (A B A B A B). Report the min of the minimums and the ratio, plus allocations.
- Do not claim a speedup under 10% as real unless allocations corroborate it.
- Benchmark the relevant `benchmark/benchmarks_v0.10.jl` cases, using the same setup code, plus at least one realistic non-benchmark case for the same code path.
- Check that outputs are equal (`==`, or `≈` where the floating-point order changes) against the baseline on several inputs. This includes the benchmark inputs, and the inputs must be serialized or regenerated deterministically.
- Reject or revert a fix that gives no measurable gain, unless it fixes a bug or removes a type instability that is cheap to remove.

## Reporting
- Write `perf_review/phase2/<stream>.md` **inside your worktree** and commit it on your branch.
- It contains the per-item analysis, decision, implementation summary, before/after numbers (min times, allocations), test results, reviewer findings and how they were resolved, and anything deferred.
- Keep the final message to the caller concise: the decisions per item, the ratios, the test status, and the commits.
