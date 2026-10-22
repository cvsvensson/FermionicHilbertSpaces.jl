# Phase 2, stream s7-bench: the benchmark suite

Branch `perf/s7-bench`, first based on `660c67f4` and now rebased onto `quality-control` (`725b5f6b`); see the last section. Only `benchmark/` was changed; `src/` and `.github/` are untouched.

## Plan

1. Infrastructure first (I1, I2): make `benchmarks.jl` runnable outside AirspeedVelocity, fix `benchmark/Project.toml`.
2. Hygiene (H03–H14, R1–R3) in one restructuring of `benchmarks_v0.10.jl`. Every section becomes a `let` block with its own RNG. Keys whose measured operation changes get new names.
3. Gap benchmarks (G2–G12) appended as new groups, with sizes tuned against the baseline env so that each leaf takes about 20 µs–20 ms. `precomputed_vs_eager.jl` is folded into SUITE (I6).
4. A separate TTFX suite (G1), plus `compare.jl` for running and comparing results.
5. Validation against the baseline env, a static review by a subagent, and fixes.

## CI and the TTFX decision

`.github/workflows/benchmark.yml` runs `MilesCranmer/AirspeedVelocity.jl@action-v1` on every PR. The action runs the PR's `benchmark/benchmarks.jl` against both the base and the PR revision, and runs the whole `SUITE`. So:
- every benchmark may use only API that also exists on `main` (goal 4);
- **G1 (TTFX) is not in `SUITE`.** It lives in `benchmark/ttfx.jl` as `TTFX_SUITE`. One sample spawns a Julia process, which takes 2 s (`using`) to 11 s (first use). With the warm-up evaluation that BenchmarkTools does, each key costs about 4 processes, roughly 1 minute in total, on every revision. In CI on shared runners, TTFX numbers are also dominated by runner variance and precompilation state. It is run on demand with `compare.jl run out.json --suite=ttfx`.

No workflow change was needed: `I1` was fixed in `benchmarks.jl` itself.

## Per-item decisions

| item | decision | what was done |
|---|---|---|
| I1 | done | `benchmarks.jl` uses `PACKAGE_VERSION` when AirspeedVelocity defines it, else `pkgversion(FermionicHilbertSpaces)`. |
| I2 | done | Removed NonCommutativeProducts from `benchmark/Project.toml`; added LinearAlgebra and Printf (both loaded by the scripts) and `[compat]` for all deps and julia. No compat entry for the package itself, because AirspeedVelocity and local `Pkg.develop` install arbitrary revisions. `benchmark/README.md` documents setup, running and comparing. Note: the locally cached General registry only has NCP ≤ 0.3.3, so with the stock registry a local setup must also `Pkg.develop` NCP 0.5 (stated in the README). |
| I6 | done; file deleted | All cases of `precomputed_vs_eager.jl` are now in SUITE: `partial_trace/precomputed/*`, `embed/precomputed/*`, `reshape/precomputed/*`, `reshape/repeat`. The file's header claimed fixed timings and an outdated pure-state caveat (the pure-state path now applies the map without forming a density matrix), so the file was removed rather than kept beside the SUITE copies. |
| H03 | done | `partial_trace/fermions/map` gets `seconds = 3` (13 samples at 181 ms instead of 8). The two `generate_states` leaves also get `seconds = 3`. |
| H04 | done | `partial_trace_algorithms/<space>/<Dense\|Sparse>/{SubAlg,FullAlg,default}`. The heuristic's choice no longer appears in keys; `default` (no `alg`) is the path users get. |
| H05 | done | Every partial trace, embedding and partial transpose leaf passes `complement=`. `tensor_product` of matrices has no such keyword, which a comment notes. |
| H06 | done | `generate_states/{hilbert_space_N60, hilbert_space_N64_bigint}` call public `hilbert_space(f, 1:N, constraint)`. N = 64 really uses BigInt Fock states (1.5 M allocations, 125 ms against 24 ms for N = 60). The old internal `generate_states(...; process_result=...)` leaves were removed. |
| H07 | done | One `Xoshiro(seed)` per section. |
| H08 | done | `DEFAULT_PARAMETERS.seconds` is set to 1 while the benchmarks are created and restored at the end of the file (`Parameters` copies the default at construction), so it no longer leaks into the session. |
| H10 | documented | `matrix_application/lazy liouvillian/mul sparse` keeps its key (the measured operation is unchanged) and gets a comment saying that it mostly measures per-call overhead. The new `lazy/mul!` vs `lazy/sparse_mul!_reference` pair covers the realistic lazy-vs-sparse comparison. |
| H14 | done | All leaves get `evals_set = true`, so `tune!` does nothing. Leaves faster than about 20 µs set `evals` explicitly (4 or 10). |
| R1 | replaced | `reshape/mixed_split_combine` (same cost as `mat_to_tens`) removed. `reshape/repeat` (a 3-tensor with `repeat=true`, 345 µs) measures a distinct path. |
| R2 | replaced | `partial_trace/product space/standard` removed. `partial_trace/product space/apply_precomputed` applies the built map (4.3 µs, evals = 10). |
| R3 | replaced | `partial_trace/fermions/standard` removed. `partial_trace/fermions/pure_state` traces a pure state with the eager API (1.3 ms). |
| G1 | done, separate suite | `ttfx/load` 2.2 s, `ttfx/first_use` 10.5 s (min of 3, baseline). |
| G2 | done | `partial_trace/precomputed/{construct, eager_matrix, apply_matrix, apply_matrix_inplace, apply_vectorized_inplace, eager_pure_state, apply_pure_state, apply_pure_state_inplace}`, `embed/precomputed/{construct, eager, apply, apply_inplace}`, `reshape/precomputed/{construct, apply, apply_inplace}`. |
| G3 | done | `spin/{hilbert_space_Sz0, symbolic_heisenberg, matrix_representation_heisenberg}`. The matrix representation uses N = 12 (dim 924), because N = 14 took 50 ms per sample. `hilbert_space_Sz0` uses N = 14 (7 ms). |
| G4 | done | `hubbard/{space, symbolic, matrix_representation, constrain_no_double_occupation, matrix_representation_projected}`, N = 10 with 3 up and 2 down (dim 5400). `constrain_no_double_occupation_N30` has 30 sites with 2 up and 1 down, and one constraint per site. The sizes follow the gap profiler, so that the cases stay above 50 µs after the expected 10–25× speedups. |
| G5 | done | `bosons/{hilbert_space, matrix_representation}` (6 sites, 6 bosons, dim 456). |
| G6 | done | `lazy/{construct, mul!, sparse_mul!_reference}` (14 modes, half filling). |
| G7 | done | `partial_transpose/{eager, construct, apply}`. |
| G8 | done | `tensor_product/{matrices, generalized_kron, spaces}` with 4 + 5 modes. |
| G9 | done | `open_system/{liouvillian_symbolic, liouvillian_sparse}` on the same 6-mode open system as the lazy Liouvillian. |
| G10 | done | `symbolic_states/{op_on_ket, parse_kets, vector_representation}`, `sectors/{construct_all, sectors, matrix_representation_sectors}`. |
| G11 | done | `state_index/{constrained, product_space}`. The proposed `basisstate` leaf took 1.5 µs for a whole sweep (a vector lookup) and was left out. |
| G12 | done | `majorana/matrix_representation`. |

### Internals used

Only names that the base suite already used: `complementary_subsystem`, `SubsystemPartialTraceAlg`/`FullPartialTraceAlg`, `open_system`, `SciMLOperators.cache_operator`, `AdditiveConstraint` (documented in docs/src/conservation.md). The internal `generate_states`, `CombineFockNumbersProcessor` and `default_partial_trace_alg` calls are gone. None of the other stream worktrees changes these names (checked against their diffs so far).

## Key changes

- **Removed:** `generate_states/{int, big_int}`, `partial_trace_algorithms/default=*/…` (all 8), `partial_trace/fermions/standard`, `partial_trace/product space/standard`, `reshape/mixed_split_combine`.
- **Added (replacements):** `generate_states/{hilbert_space_N60, hilbert_space_N64_bigint}`, `partial_trace_algorithms/{Sparse space, Full space}/{Dense, Sparse}/{SubAlg, FullAlg, default}` (12), `partial_trace/fermions/pure_state`, `partial_trace/product space/apply_precomputed`, `reshape/repeat`.
- **Added (gaps):** the groups listed under G2–G12 above, 46 leaves.
- **Unchanged keys** measure the same operation as before; only their random inputs changed (H07), and `partial_trace/fermions/map` has a longer time budget.

SUITE has 86 leaves; the base suite has 36. 23 keys are unchanged, 13 were removed, and 63 were added.

## compare.jl

```
julia --project=<env> benchmark/compare.jl run out.json [--suite=main|ttfx|all] [--filter=<regex>] [--seconds=<s>]
julia --project=<env> benchmark/compare.jl compare base1.json,base2.json new1.json,new2.json [--tol=0.1] [--memtol=0.01]
```

`run` includes the benchmark files next to the script, so a worktree's suite can be run against any env. `compare` takes the smallest minimum over several files per side (for A/B/A/B runs) and prints a markdown table with ratios. It flags time ratios outside 1 ± tol and any change in allocation count. Memory is compared with a 1% tolerance: in back-to-back runs of identical code, 17 leaves differed by 16–144 bytes, so an exact memory comparison would flag noise.

## Validation (baseline env, full suite, default settings)

Two full runs of `compare.jl run` (SUITE only, default settings) against `scratchpad\phase1\benchenv`, which devs the base checkout. The benchmark files came from this worktree. Other streams were benchmarking on the machine at the same time.

| | run 1 | run 2 |
|---|---|---|
| suite setup (include, mostly compilation) | 152 s | 149 s |
| `run(SUITE)` | 192 s | 173 s |
| process wall time | 354 s | 330 s |

- The whole suite takes about 6 minutes, against about 3 minutes for setup, `tune!` and `run` of the old suite, which has 36 leaves. The new suite has 86 leaves, and `tune!` is no longer needed. TTFX (`--suite=ttfx`) takes another 66 s.
- **Allocation counts were identical in the two runs for all 85 leaves.** Memory differed by up to 144 bytes on 17 leaves.
- **Noise:** the median over leaves of (median/min − 1) is 0.87, and the maximum is 2.3. The run2/run1 ratio of minimum times has a median of 0.89 and ranges from 0.52 to 1.55. For 60 of 85 leaves the ratio is more than 10% away from 1, so on this loaded machine even minimum times drift by ±10–40% between runs. The A/B/A/B protocol, with `compare` taking the minimum over several files, plus allocation counts, is needed to judge a change.
- Runs 1 and 2 used the code before the review-driven renames and the Hubbard/spin resizing. The renamed keys were re-run with `--filter` and work. The Hubbard and spin rows come from a separate filtered run of the final code: Hubbard has 10 sites, `spin/hilbert_space_Sz0` uses 14 spins, and `hubbard/constrain_no_double_occupation_N30` is new. These sizes follow the coordinator's gap-profiler recommendation: optimized code is expected to be 10–25× faster on the Hubbard and constraint cases, so they should stay above 50 µs. On the base code, `hubbard/space` (25 ms), `hubbard/constrain_no_double_occupation` (37 ms) and `_N30` (173 ms, `seconds = 3`) are therefore above the 20 ms target.

Per-leaf results (base code):

| benchmark | min (run 1) | min (run 2) | run2/run1 | median/min − 1 (run 1) | samples (run 1) | allocs | memory |
|---|---|---|---|---|---|---|---|
| bosons/hilbert_space | 546.600 μs | 503.700 μs | 0.92 | 1.04 | 817 | 6540 | 404.84 KiB |
| bosons/matrix_representation | 886.400 μs | 731.600 μs | 0.83 | 0.67 | 535 | 973 | 736.82 KiB |
| complement/fermions | 93.100 μs | 78.700 μs | 0.85 | 0.99 | 4611 | 589 | 69.34 KiB |
| complement/product space | 125.900 μs | 103.700 μs | 0.82 | 1.15 | 3017 | 1068 | 51.22 KiB |
| embed/fermions | 2.225 ms | 1.157 ms | 0.52 | 0.69 | 221 | 256 | 2.61 MiB |
| embed/precomputed/apply | 1.864 ms | 1.358 ms | 0.73 | 0.43 | 293 | 5 | 3.74 MiB |
| embed/precomputed/apply_inplace | 468.900 μs | 417.100 μs | 0.89 | 0.86 | 1062 | 2 | 64 bytes |
| embed/precomputed/construct | 2.938 ms | 2.128 ms | 0.72 | 0.23 | 224 | 242 | 4.74 MiB |
| embed/precomputed/eager | 800.600 μs | 778.200 μs | 0.97 | 0.79 | 582 | 253 | 1.39 MiB |
| embed/product space | 102.200 μs | 89.300 μs | 0.87 | 1.17 | 3495 | 743 | 83.50 KiB |
| generate_states/hilbert_space_N60 | 24.368 ms | 14.472 ms | 0.59 | 0.3 | 94 | 52743 | 1.86 MiB |
| generate_states/hilbert_space_N64_bigint | 125.564 ms | 112.244 ms | 0.89 | 0.37 | 14 | 1505469 | 61.64 MiB |
| hilbert_space | 278.900 μs | 162.600 μs | 0.58 | 1.03 | 1040 | 145 | 400.45 KiB |
| lazy/construct | 75.700 μs | 68.100 μs | 0.9 | 1.1 | 4167 | 899 | 278.60 KiB |
| lazy/mul! | 1.384 ms | 1.208 ms | 0.87 | 0.56 | 426 | 106 | 4.34 KiB |
| lazy/sparse_mul!_reference | 27.200 μs | 23.800 μs | 0.88 | 0.66 | 10000 | 0 | 0 bytes |
| majorana/matrix_representation | 1.800 ms | 1.240 ms | 0.69 | 0.69 | 260 | 194 | 3.59 MiB |
| matrix_application/lazy liouvillian/mul sparse | 6.996 ms | 4.052 ms | 0.58 | 0.26 | 99 | 9216 | 3.15 MiB |
| matrix_application/lazy liouvillian/mul! | 279.400 μs | 234.400 μs | 0.84 | 0.17 | 2567 | 183 | 6.83 KiB |
| matrix_representation/bdg | 985.100 μs | 792.400 μs | 0.8 | 1.16 | 364 | 228 | 2.33 MiB |
| matrix_representation/free_fermion | 470.200 μs | 381.200 μs | 0.81 | 0.71 | 1020 | 22 | 461.59 KiB |
| matrix_representation/product space | 1.045 ms | 765.800 μs | 0.73 | 0.92 | 466 | 17598 | 992.09 KiB |
| matrix_representation/product space many spaces | 2.970 ms | 4.605 ms | 1.55 | 0.77 | 148 | 9272 | 4.61 MiB |
| matrix_representation/product space many spaces constrained | 1.484 ms | 1.025 ms | 0.69 | 0.68 | 384 | 12193 | 1.01 MiB |
| matrix_representation/standard | 2.761 ms | 1.737 ms | 0.63 | 0.56 | 194 | 169 | 3.49 MiB |
| open_system/liouvillian_sparse | 4.228 ms | 4.647 ms | 1.1 | 1.27 | 94 | 3932 | 7.14 MiB |
| open_system/liouvillian_symbolic | 106.600 μs | 99.200 μs | 0.93 | 1.98 | 2874 | 3437 | 265.22 KiB |
| partial_trace/fermions/map | 181.323 ms | 114.509 ms | 0.63 | 0.31 | 13 | 242 | 257.26 MiB |
| partial_trace/fermions/pure_state | 1.284 ms | 854.700 μs | 0.67 | 0.35 | 559 | 233 | 206.60 KiB |
| partial_trace/precomputed/apply_matrix | 202.600 μs | 252.500 μs | 1.25 | 0.89 | 2463 | 5 | 4.15 KiB |
| partial_trace/precomputed/apply_matrix_inplace | 209.700 μs | 232.300 μs | 1.11 | 0.78 | 2628 | 2 | 64 bytes |
| partial_trace/precomputed/apply_pure_state | 143.100 μs | 128.100 μs | 0.9 | 1.0 | 3395 | 4 | 4.12 KiB |
| partial_trace/precomputed/apply_pure_state_inplace | 148.000 μs | 127.300 μs | 0.86 | 0.88 | 3319 | 0 | 0 bytes |
| partial_trace/precomputed/apply_vectorized_inplace | 210.500 μs | 193.900 μs | 0.92 | 0.8 | 2349 | 0 | 0 bytes |
| partial_trace/precomputed/construct | 3.723 ms | 2.115 ms | 0.57 | 0.34 | 135 | 241 | 4.74 MiB |
| partial_trace/precomputed/eager_matrix | 886.300 μs | 851.000 μs | 0.96 | 0.38 | 812 | 220 | 17.08 KiB |
| partial_trace/precomputed/eager_pure_state | 852.800 μs | 719.200 μs | 0.84 | 0.45 | 755 | 232 | 40.71 KiB |
| partial_trace/product space/apply_precomputed | 4.310 μs | 4.110 μs | 0.95 | 1.03 | 10000 | 4 | 1.17 KiB |
| partial_trace/product space/map | 377.200 μs | 362.200 μs | 0.96 | 1.04 | 1002 | 15517 | 697.86 KiB |
| partial_trace_algorithms/Full space/Dense/FullAlg | 3.089 ms | 2.525 ms | 0.82 | 0.7 | 178 | 200 | 35.91 KiB |
| partial_trace_algorithms/Full space/Dense/SubAlg | 48.700 μs | 46.600 μs | 0.96 | 0.99 | 9098 | 197 | 11.84 KiB |
| partial_trace_algorithms/Full space/Dense/default | 48.400 μs | 46.700 μs | 0.96 | 1.01 | 9451 | 197 | 11.84 KiB |
| partial_trace_algorithms/Full space/Sparse/FullAlg | 90.400 μs | 80.400 μs | 0.89 | 2.32 | 2689 | 210 | 360.37 KiB |
| partial_trace_algorithms/Full space/Sparse/SubAlg | 124.300 μs | 117.200 μs | 0.94 | 0.91 | 3907 | 198 | 11.89 KiB |
| partial_trace_algorithms/Full space/Sparse/default | 125.900 μs | 125.000 μs | 0.99 | 0.88 | 3698 | 198 | 11.89 KiB |
| partial_trace_algorithms/Sparse space/Dense/FullAlg | 976.800 μs | 897.700 μs | 0.92 | 1.06 | 485 | 457 | 85.98 KiB |
| partial_trace_algorithms/Sparse space/Dense/SubAlg | 8.022 ms | 9.181 ms | 1.14 | 0.49 | 80 | 454 | 75.66 KiB |
| partial_trace_algorithms/Sparse space/Dense/default | 967.500 μs | 893.300 μs | 0.92 | 1.01 | 479 | 457 | 85.98 KiB |
| partial_trace_algorithms/Sparse space/Sparse/FullAlg | 40.300 μs | 34.800 μs | 0.86 | 1.59 | 6794 | 467 | 146.17 KiB |
| partial_trace_algorithms/Sparse space/Sparse/SubAlg | 9.419 ms | 9.681 ms | 1.03 | 0.25 | 79 | 455 | 75.71 KiB |
| partial_trace_algorithms/Sparse space/Sparse/default | 38.300 μs | 51.500 μs | 1.34 | 1.12 | 7853 | 467 | 146.19 KiB |
| partial_transpose/eager | 2.529 ms | 2.075 ms | 0.82 | 0.42 | 247 | 177 | 1.02 MiB |
| partial_transpose/apply | 643.000 μs | 741.800 μs | 1.15 | 0.63 | 667 | 7 | 1.00 MiB |
| partial_transpose/construct | 6.790 ms | 4.758 ms | 0.7 | 0.38 | 93 | 261 | 9.22 MiB |
| permutations/few_large | 12.514 ms | 7.703 ms | 0.62 | 0.22 | 60 | 873 | 16.26 MiB |
| permutations/many_small | 969.700 μs | 1.048 ms | 1.08 | 1.97 | 316 | 9923 | 1.55 MiB |
| reshape/mat_to_tens | 48.500 μs | 43.000 μs | 0.89 | 1.02 | 8260 | 482 | 97.09 KiB |
| reshape/mat_to_vec | 105.700 μs | 95.900 μs | 0.91 | 0.95 | 3145 | 4346 | 489.31 KiB |
| reshape/precomputed/apply | 29.225 μs | 24.475 μs | 0.84 | 1.01 | 3202 | 6 | 64.22 KiB |
| reshape/precomputed/apply_inplace | 18.950 μs | 16.850 μs | 0.89 | 0.88 | 6822 | 4 | 160 bytes |
| reshape/precomputed/construct | 23.900 μs | 21.775 μs | 0.91 | 1.12 | 4331 | 496 | 34.82 KiB |
| reshape/repeat | 345.100 μs | 251.300 μs | 0.73 | 0.92 | 1286 | 530 | 545.79 KiB |
| sectors/construct_all | 193.700 μs | 116.900 μs | 0.6 | 1.18 | 1626 | 168 | 353.49 KiB |
| sectors/matrix_representation_sectors | 1.130 ms | 906.200 μs | 0.8 | 0.94 | 399 | 121 | 1.97 MiB |
| sectors/sectors | 64.600 μs | 58.100 μs | 0.9 | 1.16 | 5049 | 73 | 162.32 KiB |
| state_index/constrained | 56.900 μs | 44.300 μs | 0.78 | 0.76 | 8740 | 0 | 0 bytes |
| state_index/product_space | 66.700 μs | 59.700 μs | 0.9 | 0.88 | 6661 | 0 | 0 bytes |
| symbolic/deep_product | 843.000 μs | 855.100 μs | 1.01 | 0.73 | 603 | 2424 | 653.72 KiB |
| symbolic/sum | 223.100 μs | 200.100 μs | 0.9 | 1.32 | 1416 | 5417 | 783.20 KiB |
| symbolic/sum_square | 1.296 ms | 1.123 ms | 0.87 | 0.64 | 398 | 10393 | 1.34 MiB |
| symbolic_states/op_on_ket | 505.900 μs | 401.300 μs | 0.79 | 0.79 | 886 | 14888 | 608.34 KiB |
| symbolic_states/parse_kets | 78.300 μs | 70.100 μs | 0.9 | 0.93 | 5015 | 4003 | 203.19 KiB |
| symbolic_states/projected_op_on_state | 264.700 μs | 226.300 μs | 0.85 | 1.04 | 1497 | 7008 | 297.26 KiB |
| symbolic_states/vector_representation | 327.500 μs | 305.000 μs | 0.93 | 0.92 | 1462 | 2492 | 168.02 KiB |
| tensor_product/generalized_kron | 9.364 ms | 5.818 ms | 0.62 | 0.38 | 75 | 187 | 4.01 MiB |
| tensor_product/matrices | 6.339 ms | 4.156 ms | 0.66 | 0.2 | 94 | 1103 | 13.11 MiB |
| tensor_product/spaces | 10.150 μs | 9.030 μs | 0.89 | 1.12 | 4211 | 216 | 11.39 KiB |
| hubbard/constrain_no_double_occupation | 37.111 ms | (separate run) | | 0.43 | 18 | 134980 | 4.53 MiB |
| hubbard/constrain_no_double_occupation_N30 | 172.752 ms | (separate run) | | 0.04 | 6 | 594734 | 17.56 MiB |
| hubbard/matrix_representation | 2.549 ms | (separate run) | | 0.51 | 234 | 237 | 2.89 MiB |
| hubbard/matrix_representation_projected | 1.126 ms | (separate run) | | 0.51 | 549 | 231 | 597.09 KiB |
| hubbard/space | 25.263 ms | (separate run) | | 0.49 | 28 | 82398 | 3.63 MiB |
| hubbard/symbolic | 49.100 μs | (separate run) | | 0.95 | 8170 | 2177 | 157.56 KiB |
| spin/hilbert_space_Sz0 | 6.685 ms | (separate run) | | 0.58 | 84 | 67633 | 10.45 MiB |
| spin/matrix_representation_heisenberg | 10.891 ms | (separate run) | | 0.4 | 64 | 2812 | 5.13 MiB |
| spin/symbolic_heisenberg | 92.500 μs | (separate run) | | 1.06 | 4317 | 2630 | 171.62 KiB |

## Reviewer findings

A general-purpose subagent statically reviewed `git diff 660c67f4..HEAD`. It found no bugs: no interpolation mistakes, no leaf that measures setup instead of its operation, no mutation across evaluations, no new internals, and key stability was correct. It made these suggestions:
- Inconsistent new key names: `partial_transpose/{map_build, map_apply}` were renamed to `{construct, apply}`. `partial_trace/product space/apply_precomputed` was kept, because it pairs with the existing `map` key in that group.
- `sectors/matrix_representation_blockdiag`: there is no block-diagonal type, so it was renamed to `matrix_representation_sectors` and the heading fixed.
- The TTFX comments and README now say that the times include Julia startup.
- The comment about `complement=` now notes that `tensor_product` of matrices computes the complement.
- `compare.jl` now rejects unknown or value-less options and counts keys present on only one side.
- The claim that fast leaves set `evals` was checked in the validation run: every `evals = 1` leaf has a minimum time of at least 27 µs, so the claim holds.
- Not acted on: there is no longer an eager partial trace on the product space or a split-and-combine reshape in one call. Both duplicated other leaves in cost (R1, R2), and the eager path is covered by `partial_trace/precomputed/eager_matrix` and `partial_trace_algorithms`.

## Commits

- b6753923 Size the Hubbard and spin benchmarks for optimized code
- 0bd210b5 Name new benchmark keys consistently and tighten compare.jl options
- a9fa608a Add a script to run and compare benchmarks, and time-to-first-use benchmarks
- ca71b838 Add benchmarks for untested paths and fold in precomputed_vs_eager.jl
- 987e86d7 Make the benchmark suite reproducible and its keys stable
- 96a88882 Make the benchmark environment usable without AirspeedVelocity
- (this report)

## Rebase onto quality-control (725b5f6b)

- The suite now targets the `quality-control` API and no longer needs to run on 660c67f4.
- The only conflict was qc's own change to `benchmarks_v0.10.jl`. qc replaced the deprecated `single_particle_hilbert_space` and `bdg_hilbert_space` with `hilbert_space(f, 1:1000, SingleParticle())` and `hilbert_space(f, 1:1000, BdG())`. I applied the same change inside the restructured suite.
- `majorana_hilbert_space(γ, 1:20)`, which qc also deprecates, became `hilbert_space(γ, 1:20)`.
- Building the suite with `--depwarn=yes` on `scratchpad\phase2\baseline-qc-env` printed no warnings.
- All three edits only change how a space is built during setup. Keys and measured operations are unchanged, so the qc baseline runs `results\qc_run1.json` and `qc_run2.json` stay valid.
- I re-ran the three affected leaves on baseline-qc-env. Their allocation counts match `qc_run1` exactly: `majorana/matrix_representation` 190, `matrix_representation/bdg` 229, `matrix_representation/free_fermion` 23.
- `compare.jl compare` gained a "beyond noise" column that uses a wider band (`--band`, default 0.15). It also prints a summary grouped by allocation change (fewer, same, more allocations). Each group shows its count, how many benchmarks fall outside the band in either direction, and its geometric mean ratio. Comparing `qc_run1` with `qc_run2` (identical code) still gives 18 of 86 leaves above +15%. On this machine, timing verdicts for a single file pair should be treated as indicative only; the allocation groups are the robust signal.
