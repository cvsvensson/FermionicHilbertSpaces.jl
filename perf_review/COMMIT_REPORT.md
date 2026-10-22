# Commit report: performance branch for FermionicHilbertSpaces.jl

Target: `quality-control` (base `1cee0550`). 42 commits, listed in branch order. Sizes are `git show --shortstat`. "Ratio" means time of the commit divided by time of its parent (below 1 is faster); "allocs" is the allocation count change. Gains are the implementers' own-commit measurements; the reviewers did not re-measure them.

## 1. Summary

- The branch removes measured bottlenecks (Phase-1 IDs B01-B28, A/E/G items) in permutation sectors, partial trace/embed, matrix representation, lazy operators, constrained-state generation, reshape and symbolic states, and adds a reproducible benchmark suite.
- Measurement: each commit against its parent; minimum of at least 3 alternating runs, plus exact allocation counts. Same-code noise on this machine is up to +-14% per leaf, so time ratios between 0.86 and 1.14 are not evidence; allocation changes are the reliable signal.
- Seven independent reviews (one per stream) read the diffs; the findings below are condensed from them. Follow-ups they raised have since been folded into the commits or added as two commits at the end (marked "addressed").
- Tests: the full suite passes on the branch head, 5136/5136, including Aqua and ExplicitImports.
- Suite result: across all 86 leaves the geometric mean of time ratios against quality-control is 0.375, and no leaf is slower beyond the noise band. See section 8.
- Commits the reviewers suggest considering dropping: `e3ed89fc` (noise-level gain, aliases the input dict), `617cd677` (memory only, no time gain), `5fbbfb6b` (about 60 lines for 0.74x on one benchmark). A sector-container commit was dropped after the final suite run showed it slowed `state_index`; see section 6.
- Commits that need a maintainer decision:
  - `9b39d9a4`: `SuperOperatorCOO` replaces CSC for partial-trace/embed maps (`.map` type changes; matrix-matrix products and `A + B` still use the generic densifying path).
  - `e1726e99`: cost model calibrated only for 1-2 traced sites; small-complement regime unverified.
  - `4e2a7cce`: `combine_states` result becomes `BigInt`-backed for 64 or more modes.
  - `0f5c7b7f`: `catenate_fock_states` throws on overflow; quality-control's `generate_states/big_int` leaf relied on the old promotion, so it must ship with `676ff2e4`.
  - `16453ff2`: changes the (non-public) `valid_branch` protocol.
  - Deferred design items E3 and B16; NCP `ff99d2d` (non-public Base internals; separate repo).

## 2. Summary table

| Hash | Title | Area | +/- lines | Own-commit gain (time ratio; allocs) | Verdict |
|---|---|---|---|---|---|
| 663de51f | Make the benchmark environment usable without AirspeedVelocity | bench | +15/-2 | none | keep |
| 676ff2e4 | Make the benchmark suite reproducible and its keys stable | bench | +210/-192 | none | keep |
| 155fac89 | Add benchmarks for untested paths and fold in precomputed_vs_eager.jl | bench | +186/-141 | none | keep (check against main) |
| c8bf2fb8 | Add a script to run and compare benchmarks, and time-to-first-use benchmarks | bench | +281/-0 | none | keep |
| c65fdc1f | Hash SpinState through twice(m) | permutation/spin | +6/-2 | 0.47-0.71 (spin symmetric cases) | keep |
| b53f217e | Convert permutations to tuples once per permutation behind a function barrier | permutation | +19/-8 | 0.24 many_small; 4819 to 500 | keep |
| 419efd7a | Check sparse projectors for roundoff through their stored entries | permutation | +3/-1 | 0.18 few_large; 0.62 C10 | keep |
| 617cd677 | Return the sparse permutation projector without copying and rescale it in place | permutation | +5/-7 | 0.94-1.03; memory -4..-11% | optional |
| f02237a1 | Orthogonalize the sector projector block by block with a relative tolerance | permutation | +179/-26 | 0.36-0.64; memory -40..-60% | keep (correctness fix) |
| 20c93575 | Build symmetric and antisymmetric sector bases from orbits of basis states | permutation | +157/-7 | 3x to 2900x faster; allocs /10 to /500 | keep |
| 41b1e9cb | Compute product-space phase factors with map over the factor tuples | partial trace | +4/-12 | 0.22; 15443 to 659 | keep |
| fdbd128c | Loop over complement states first in SubsystemPartialTraceAlg | partial trace | +55/-23 | 0.05-0.29 | keep |
| 8ae47000 | Precompute subsystem indices in FullPartialTraceAlg | partial trace | +47/-29 | 0.31-0.55; bosonprod 0.12 | keep |
| 3ed48f8c | Visit stored entries directly in FullPartialTraceAlg for sparse input | partial trace | +12/-0 | 0.51; memory 404 to 70 KB | keep |
| 9b39d9a4 | Store partial-trace maps as a column-sorted COO superoperator | partial trace/embed | +332/-8 | build 113x faster (270 MB to 1.9 MB); apply 83-131x | decision |
| e1726e99 | Choose the partial-trace algorithm from a cost model of both kernels | partial trace | +44/-2 | 0.30 sparse full-space default | keep, with caveat |
| 87b206c3 | Convert vectors of substates to tuples of static length in ProductSpaceMapper | mapper | +19/-0 | 0.46; 67633 to 2425 | keep |
| cdb0e975 | Keep the ProductSpaceMapper piece table concretely typed | mapper | +30/-22 | 0.45 partial trace; build 54 to 24 us | keep |
| 16453ff2 | Keep running sums for additive branch constraints in generate_states | generate_states | +143/-31 | 0.17-0.18 | keep |
| 899a6eef | Skip the subspace position lookup for constraints on all spaces | generate_states | +2/-2 | 0.75 setup; 5118 to 4391 | keep |
| 009311ca | Fill product sector labels with a typed loop | constraints | +19/-3 | 0.05-0.30 | keep |
| 4e2a7cce | Give fermionic state mappers the mode count of the full space | fermions | +24/-6 | 0.16-0.23; allocs /5-/8 | keep (BigInt decision) |
| 797515b1 | Look up fermionic subspace modes without building single-mode spaces | reshape | +6/-1 | 0.89-0.94; 502 to 326 | keep |
| d856653f | Use typed transition tables and linear offsets in reshape | reshape | +87/-72 | 0.60-0.88; 4192 to 91 | keep |
| 0f5c7b7f | Return the requested type from catenate_fock_states | fermions | +18/-2 | none (bug fix) | keep, with 676ff2e4 |
| f56a225d | Return a Tuple-backed product, or 0, as the effect of an operator on a state | symbolic | +24/-2 | 0.94-0.98; allocs -21% | keep |
| 47fd35a4 | Act on symbolic states with a Tuple-backed one-factor product | symbolic | +22/-1 | 0.90 krylov; allocs -29..-35% | keep |
| 12ab1ba0 | Keep a single amplitude type in generic apply_local_operators | matrix rep | +36/-10 | 0.36; 388397 to 2165 | keep |
| 262980da | Apply product-operator factors by tuple recursion | matrix rep | +11/-43 | 0.53-0.64; 17588 to 3764 | keep |
| 69c8c641 | Partition product operators by looking up each factor's basis | matrix rep | +20/-15 | 0.62 (ps_many_cons); 12169 to 4005 | keep |
| 639f64ea | Use one accumulator for sums on constrained multi-group spaces | matrix rep | +56/-1 | 0.65-0.83; 952 to 499 | keep |
| 5fbbfb6b | Build product-space term matrices with a direct identity kron | matrix rep | +64/-5 | 0.74 one benchmark (noisy) | optional |
| e3ed89fc | Drop the identity of a sum without re-filtering its terms | matrix rep | +3/-1 | 0.90, noise; no alloc change | consider dropping |
| 479f8952 | Check single-particle and BdG operators in one pass | matrix rep | +40/-3 | 0.80-0.86 | keep |
| afab0e85 | Store per-term precomputation in lazy NCAdd operators | lazy | +20/-13 | 0.85 small; allocs 42 to 0 | keep |
| 640d93e3 | Build the lazy multi-space sum as one AddedOperator | lazy | +10/-6 | 0.63; compile 7.0 to 3.3 s | keep |
| 2e1072ef | Overwrite the output for zero beta in lazy NCAdd mul! | lazy | +23/-6 | none (bug fix) | keep |
| 58041ec1 | Group single-group terms of lazy product-space operators | lazy | +94/-0 | cache_operator 0.28 (13 to 1.3 s); sparse 0.50 | keep; small case 1.15x slower |
| b454581f | Collect lazy sparse-vector products as index/value pairs | lazy | +42/-0 | 0.30 (5% density); 115 to 41 | keep |
| fe2fc1fe | Support a global phase in the weights of permutation sectors | permutation | +20/-3 | none (bug fix) | keep |
| 13c9339f | Add sparse methods for transpose, comparison, scaling and display of SuperOperatorCOO | partial trace/embed | +91/-5 | none (removes densifying fallbacks) | keep with 9b39d9a4 |

## 3. Per-commit sections

### Benchmarks

#### 663de51f Make the benchmark environment usable without AirspeedVelocity
- Bottleneck: `PACKAGE_VERSION` exists only under AirspeedVelocity; the benchmark `Project.toml` listed an unused NonCommutativeProducts entry and lacked LinearAlgebra and compat.
- Resolution: `benchmark/benchmarks.jl` falls back to `pkgversion`; environment fixed. No keys change. Size +15/-2.
- Assessment: nothing in the benchmark files uses NCP. CI resolution depends on NCP 0.5 being registered, which affects every CI job equally. Benchmark compat says julia 1.11, the package says 1.12.
- Verdict: keep.

#### 676ff2e4 Make the benchmark suite reproducible and its keys stable
- Bottleneck: global RNG, session-wide `DEFAULT_PARAMETERS` mutation, `tune!`, internal `generate_states` use, complement computed inside samples, redundant leaves.
- Resolution: setup in `let` blocks, fixed seeds, all leaves `evals_set`. 23 keys unchanged, 13 removed; added `generate_states/hilbert_space_N60|N64_bigint`, 12 `partial_trace_algorithms/*/{SubAlg,FullAlg,default}`, `pure_state`, `apply_precomputed`, `reshape/repeat`. Size +210/-192.
- Assessment: removed keys end AirspeedVelocity history for those names. Restoring `DEFAULT_PARAMETERS.seconds` is not exception-safe. Remaining internals used (`FHS.complementary_subsystem`, algorithm types) were already in the base suite.
- Verdict: keep. Required by `0f5c7b7f` (section 4).

#### 155fac89 Add benchmarks for untested paths and fold in precomputed_vs_eager.jl
- Bottleneck: no coverage of spins, Hubbard, bosons, lazy, open systems, partial transpose, tensor_product, sectors, state lookup, Majoranas, precomputed maps.
- Resolution: about 46 new leaves (SUITE 36 to 86); `precomputed_vs_eager.jl` deleted. Also contains reviewer follow-ups now folded in: key renames, Hubbard/spin resizing (Hubbard N 8 to 10, spin N 12 to 14, new `hubbard/constrain_no_double_occupation_N30`), non-deprecated Majorana constructor. Size +186/-141.
- Assessment: local runtime about 6 min instead of 3; AirspeedVelocity runs it twice. It uses internals (`FHS.open_system`, `SciMLOperators.cache_operator`, `AdditiveConstraint(...; maps=)`), and one rename breaks the include and so every benchmark. AirspeedVelocity runs this file against `main`, which lacks quality-control; keyword details were not verified against main. `_N30` timing is noisy, its allocation count is deterministic.
- Verdict: keep, after one full run against main in a clean environment.

#### c8bf2fb8 Add a script to run and compare benchmarks, and time-to-first-use benchmarks
- Resolution: `compare.jl` (minimum over files, exact allocation comparison, 1% memory tolerance, `--band` noise column, summary grouped by allocation change, unknown options rejected, keys present on one side counted), `ttfx.jl` (`TTFX_SUITE`, kept out of `SUITE`, about 66 s on demand), `README.md`. Size +281/-0 (reviewer follow-ups folded in).
- Assessment: logic is sound. `ttfx.jl` is not run in CI, which is the right decision.
- Verdict: keep.

### Permutation symmetry (`src/permutation_symmetry.jl`)

#### c65fdc1f Hash SpinState through twice(m)
- Bottleneck: B20. `hash(::HalfInteger)` is slow and runs on every Dict lookup of spin states.
- Resolution: `hash(twice(x.m), h)`; consistent with `==` because `twice` is a bijection. The hash itself is 17x faster.
- Gain: Heisenberg N=12 Sz=0 0.71, many_small/symmetric 0.68, 8 spin-1/2 symmetric 0.47. Size +6/-2 (4 lines of test).
- Assessment: relies on `==` comparing `m` exactly. A test pins `Int16`/`Int` hash consistency. No risk found.
- Verdict: keep.

#### b53f217e Convert permutations to tuples once per permutation behind a function barrier
- Bottleneck: B19. `Tuple(perm)` is type-unstable inside the `zip(perms, ws)` loop.
- Resolution: `add_permutation_operators!` holds the loop; `_match_perm` converts to `NTuple{N}` and checks the length.
- Gain: many_small/custom_sym 0.24, 4819 to 500 allocs; others neutral. Size +19/-8.
- Assessment: assumes 1-based `perm` (as `TupleTools.permute` did). A wrong length now throws `ArgumentError`, tested.
- Verdict: keep.

#### 419efd7a Check sparse projectors for roundoff through their stored entries
- Bottleneck: regression introduced in quality-control: `maximum(abs, P)` on a `SparseMatrixCSC` visits all d^2 entries (22.5 ms for few_large).
- Resolution: `_max_abs(::SparseMatrixCSC)` reads `nonzeros(P)`.
- Gain: few_large/custom_sym 0.18; C10 and 5 spin-2 0.62. Size +3/-1.
- Assessment: correct (`init` covers nnz = 0). No complexity test.
- Verdict: keep (essential).

#### 617cd677 Return the sparse permutation projector without copying and rescale it in place
- Bottleneck: B21 in part: an extra `sparse(::SparseMatrixCSC)` copy and `P *= c` allocating a copy.
- Resolution: `_to_sparse` passes the matrix through; `_rmul!!` scales in place when the element type is unchanged. Size +5/-7.
- Gain: no time change (0.94-1.03); memory -4 to -11%.
- Assessment: P is always freshly allocated, so no aliasing. After `20c93575`, `:symmetric` no longer reaches this path; it serves only `(perms, weights)` and `orth_method=nothing`.
- Verdict: optional; harmless but no time benefit.

#### f02237a1 Orthogonalize the sector projector block by block with a relative tolerance
- Bottleneck: B02/B21. Absolute cuts (`tol=0.1`, `|R_kk|>0.1`) drop sector vectors for orbits larger than 100 states (a correctness bug, reproduced on 5 spin-2 with m=0). Sparse QR also built a dense d x M `Q`.
- Resolution: split P into connected components of its sparsity pattern (union-find); orthogonalize each dense block with a hand-written column-pivoted Gram-Schmidt (`:qr`) or `eigen!(Hermitian)` (`:eig`). Tolerance `sqrt(eps)` times the largest column norm; each block keeps at most `round(abs2(tr B)/||B||^2)` columns.
- Gain: few_large 0.36-0.49, C8/C10 0.61-0.64, memory -40 to -60%, allocation count up. Size +179/-26, two new testitems.
- Assessment: block decomposition and pivot bound checked by the reviewer and sound. About 30 lines of custom linear algebra; a pivoted-Cholesky alternative was not benchmarked (open follow-up). The column basis differs from before (it was arbitrary); `:eig` now agrees with `:qr` on unnormalized P. For non-group weights the cap gives fewer columns than rank(P) (documented). The reviewer found the `real(tr)^2` cap (inherited from quality-control) returns 0 columns for complex-phase weights: addressed in `fe2fc1fe`. Not tested: non-group weights.
- Verdict: keep.

#### 20c93575 Build symmetric and antisymmetric sector bases from orbits of basis states
- Bottleneck: B01. `:symmetric`/`:antisymmetric` build the projector from all n! permutations (174 ms for 8 spin-1/2 with m=0).
- Resolution: signed union-find over adjacent transpositions (`_signed_orbits`); an orbit is odd when a sign constraint contradicts itself; each non-odd orbit gives one column with entries +-1/sqrt(|orbit|). Used only for `Symbol` sectors with `ispartition(Hs, H)` and `:qr`/`:eig`; a subsystem `Hs` falls back to the projector path.
- Gain: 3x to 2900x, allocations down 10-500x. Size +157/-7; the new testitem compares with the projector across 6 space types.
- Assessment: sign logic checked by hand and correct. It depends on `permute_state` being a pure permutation and a group action; the testitem checking 0/1 permutation matrices is the only guard (the reviewer suggested a comment at `permute_state`). `normalize` and `sparse` are ignored here (documented). A second union-find (signed) now sits next to the unsigned one from `f02237a1`. Not tested: `Hs` in a different order than `factors(H)`.
- Verdict: keep.

### Partial trace, embed, mappers

#### 41b1e9cb Compute product-space phase factors with map over the factor tuples
- Bottleneck: B06. A loop over zipped heterogeneous tuples dispatches dynamically and `pf` changes type.
- Resolution: `prod(map(f, state1.states, state2.states, space.factors); init=1)`.
- Gain: product partial_trace 4.55x faster (15443 to 659 allocs); partial_transpose/product 12.6x; bosonprod 28.7x (11.0M to 1049 allocs). Size +4/-12.
- Assessment: same multiplication order; `init=1` covers the empty tuple; covered by existing tests.
- Verdict: keep.

#### fdbd128c Loop over complement states first in SubsystemPartialTraceAlg
- Bottleneck: B05. `combine_states` and `state_index` ran dsub^2 * dcomp times.
- Resolution: the complement state is the outer loop; hits are buffered, then all pairs visited (`_subsystem_partial_trace_terms`).
- Gain: sparse-space dense SubAlg 21.6x; Hubbard dense 9.4x; spin 3.5x. Size +55/-23; new testitem comparing algorithms, including missing states.
- Assessment: the hit buffer takes its element type from the first combination (addressed in this commit: hit-buffer pushes now typeassert, indices converted explicitly with `Int`). The phase factor is recomputed per complement state (minor). With `skipmissing=false` the state named in the error message may differ. Spins and Majoranas not covered by the new test.
- Verdict: keep.

#### 8ae47000 Precompute subsystem indices in FullPartialTraceAlg
- Bottleneck: B11. `state_index(fsub, Hsub)` ran for every split of every element.
- Resolution: `_full_partial_trace_splits` stores `(f, ((K, fsub, fbar, w), ...))` once per basis state; a shared `@inline` per-pair kernel serves map building and the eager path. The old `@inbounds basisstate` is removed.
- Gain: dense FullAlg 1.8-3.2x; bosonprod 8.2x. Size +47/-29.
- Assessment: the table adds O(dim H) memory per call (tens of MB at dim 10^6 with sparse input). `@inline` is worth about 2x (documented in a comment).
- Verdict: keep.

#### 3ed48f8c Visit stored entries directly in FullPartialTraceAlg for sparse input
- Bottleneck: B26. `findnz` (three temporaries), then `m[J1, J2]` binary search per entry.
- Resolution: a `SparseMatrixCSC` method walks `nzrange`. Size +12.
- Gain: full-space sparse 1.98x, memory 404 to 70 KB; xover 2.5x.
- Assessment: only `SparseMatrixCSC`; other sparse types still use `findnz`. Covered by the algorithm-agreement test.
- Verdict: keep.

#### 9b39d9a4 Store partial-trace maps as a column-sorted COO superoperator
- Bottleneck: B03. A `dim(Hsub)^2 x dim(H)^2` CSC needs a column pointer of length `dim(H)^2+1` (270 MB for the fermions map).
- Resolution: new `SuperOperatorCOO{T} <: AbstractMatrix{T}` with sorted triplets and a pointer over D input columns; hand-written `mul!` (dense, sparse vector, adjoint), pure-state kernel, `getindex`, `==`, `sparse`, `copy`. `EmbedMap` stores its adjoint.
- Gain: fermions map build 113x (270 MB to 1.9 MB); dense apply 83-131x; embed of sparse input 2.43 s to 0.63 ms; sparse input 1.4-3.5x. Size +332/-8; one testitem.
- Assessment: the justification is sound (even a transposed CSC needs a dsub^2+1 pointer). Main risk found by the reviewer: operations without a kernel fell back to generic `AbstractMatrix` code and densified (`2A` gave a `Matrix`; `A ≈ A` allocated 8.4 MB; about 1.7 GB at Hubbard size), where the old CSC stayed sparse. Addressed in `13c9339f` for transpose, `==`, `isapprox`, `Matrix`, `show`, `norm`, scalar `*`, `/` and negation; matrix-matrix products and `A + B` still use the generic path. `partial_transpose.jl` keeps CSC with its own pure-state kernel, so two storage formats coexist. Basis-transformation maps convert CSC to COO through `findnz` (cheap).
- Verdict: maintainer decision. Keep together with `13c9339f`, or drop both.

#### e1726e99 Choose the partial-trace algorithm from a cost model of both kernels
- Bottleneck: B28. The old rule picks SubAlg for full-space sparse input, 3.3x slower there.
- Resolution: `_prefer_subsystem_alg`: `16*dsub*dcomp + term_cost*d^2/dcomp < 25*d + nvisits` with `term_cost` 5 (dense) or 20 (sparse).
- Gain: full-space sparse default 3.3x; xover 5% 1.48x; others unchanged. Size +44/-2; new testitem.
- Assessment: the model charges the d^2/dcomp contributing terms only to SubAlg, so it prefers FullAlg whenever dcomp is small (example: dsub=2^10, dcomp=4 now picks Full, the old rule picked Sub). Weights were fitted on cases traced to 1-2 sites with large dcomp; the small-dcomp regime is extrapolated. The calibration is tied to the kernels from `fdbd128c` and `8ae47000`, and nothing in the code records that.
- Verdict: keep, but benchmark `f, 1:10` traced to 1:8 or 1:9 with dense input first.

#### 87b206c3 Convert vectors of substates to tuples of static length in ProductSpaceMapper
- Bottleneck: G-3. `Tuple(::Vector)` has runtime length, so the following combine dispatches dynamically.
- Resolution: `NTuple{N}` built from the mapper's `TS` parameter after a length check; indexing via `firstindex`.
- Gain: spin `hilbert_space(S, 1:14, AdditiveConstraint(0))` 2.16x, 67633 to 2425 allocs. Size +19; new testitem.
- Assessment: relies on `target_spaces` being an `NTuple`, otherwise the old method runs. Offset vectors untested.
- Verdict: keep.

#### cdb0e975 Keep the ProductSpaceMapper piece table concretely typed
- Bottleneck: B16. A heterogeneous named-tuple vector with `sort`/`only` scans.
- Resolution: typed `NamedTuple` vector for coordinates, `Any[]` for spaces, one stable `sortperm`, a `Dict` for slot lookup.
- Gain: product partial_trace 2.23x; mapper build 54 to 24 us; embed/complement 1.4-1.6x. Size +30/-22.
- Assessment: relies on `order` being sorted by factor index and `sfactors` starting at 1 (both hold). Three parallel structures are harder to read.
- Verdict: keep.

### Constrained generation, fermions, reshape

#### 16453ff2 Keep running sums for additive branch constraints in generate_states
- Bottleneck: B04. Each search node re-summed the whole partial state (O(depth)).
- Resolution: `AdditiveBranchConstraint` plus per-call `branch_state` (prefix sums); 5-argument `valid_branch`. Size +143/-31.
- Gain: generate_states/int 10.6 to 1.77 ms (0.17); N=64 0.18; allocations flat.
- Assessment: the invariant (depth d is evaluated only after `prefix[d]` was written on the current branch) is documented and holds; stale prefix entries after a short-circuit are harmless. State is per call, so sharing a constraint across threads is safe. The 4-argument `valid_branch(::ProductConstraint, ...)` was removed, so a user wrapper calling it on an inner built-in constraint would get a `MethodError` (non-public protocol); addressed in this commit: a 4-argument `valid_branch` for `AdditiveBranchConstraint` with a test. Accumulator type is the `promote_type` of the first value and the sum; floating-point sums are added in generation order. Not covered: Float/Rational maps, hand-built nested `ProductConstraint`.
- Verdict: keep.

#### 899a6eef Skip the subspace position lookup for constraints on all spaces
- Bottleneck: B25. Setup ran `findfirst` over the atom ids of every space.
- Resolution: with no subspaces named, positions are `1:n`; `NumberConservation` without spaces passes `missing`. Size +2/-2.
- Gain: setup 0.134 to 0.100 ms; 5118 to 4391 allocs. A proposed id-to-position `Dict` was slower and rejected.
- Assessment: trivially correct; it also avoids picking the first match when two spaces have equal atom ids. Whether `ParityConservation` takes the same path was not verified.
- Verdict: keep.

#### 009311ca Fill product sector labels with a typed loop
- Bottleneck: G-4b. Labels collected from a `Vector{AbstractConstraint}` of closures with unknown element type.
- Resolution: label type inferred by `promote_op`, filled into a typed vector with an early `missing` return. Size +19/-3.
- Gain: Hubbard `tensor_product` 0.054-0.065, `constrain_space` 0.11-0.30, `NumberConservation*ParityConservation` 0.10.
- Assessment: the reviewer found that `promote_type` converted mixed numeric labels (`[0, 1.5]` became `[0.0, 1.5]`). Addressed in this commit: now `typejoin` (`promote_typejoin` is non-public and failed the ExplicitImports test), with a mixed Int/Float test. Falls back to `map(identity, ...)` if inference widens. The full gain depends on `4e2a7cce`.
- Verdict: keep.

#### 4e2a7cce Give fermionic state mappers the mode count of the full space
- Bottleneck: G-4c. Per-site `FockMapper`s had different `N`, hence different types, making closure collections abstract.
- Resolution: `N` is `nbr_of_modes(H)`; the bit permutation is built only when `length(perm) == N`; new check `Nmax <= N`. Size +24/-6.
- Gain: `constrain_space` 0.23 (Nh=12) and 0.16 (Nh=30); compile time for 30 per-site constraints 1.2-1.4 s to 0.4-0.5 s.
- Assessment: `isfullpartition` is still computed against `Nmax`, so two meanings of "full" coexist. `combine_states` on a partition of the first k modes of a space with 64 or more modes now returns `FockNumber{BigInt}` (was `UInt64`), matching the parent space. A partition of `1:Nmax` with `Nmax < N` falls back from `BitPermutation` to `pdep` (correct, unmeasured). Tested.
- Verdict: keep; mention the BigInt change in the changelog.

#### 797515b1 Look up fermionic subspace modes without building single-mode spaces
- Bottleneck: B13 D5. `atomic_factors` built one `FermionicSpace` per mode.
- Resolution: `_fock_atoms` returns `modes(Hsub)`, unwrapping `ConstrainedSpace` and `SectorHilbertSpace`. Size +6/-1.
- Gain: reshape 0.89-0.94 (noise); allocs 502 to 326.
- Assessment: relies on `atom_position(::FermionSym, H)` agreeing with the single-mode-space version; covered implicitly. Majorana and product spaces keep the old path.
- Verdict: keep.

#### d856653f Use typed transition tables and linear offsets in reshape
- Bottleneck: B13 D3+D4. A `Vector{Any}` of transitions narrowed afterwards; the kernel splats Cartesian tuples.
- Resolution: flattened generator collected into a typed vector; `_transitions_and_dropped`; the kernel uses precomputed zero-based linear offsets per block, with a Cartesian fallback for non-linear or offset arrays. Size +87/-72.
- Gain: mat_to_tens 0.60, mixed_split_combine 0.62, splits 0.72, mat_to_vec 0.88 (4192 to 91 allocs); mat_to_vec memory +10%.
- Assessment: two kernel paths; a testitem checks agreement on `PermutedDimsArray`. Deduplication via `last(dropped) != Iin` assumes entries per `Iin` are contiguous (uncommented). `collect` of a nested `Flatten` depends on inference; a failure silently reverts to `Vector{Any}`. Offset arrays still take the Cartesian path indexed from 1 (unchanged).
- Verdict: keep.

#### 0f5c7b7f Return the requested type from catenate_fock_states
- Bottleneck: bug, not performance: `state << shift` promoted the result to `UInt64` regardless of `T`.
- Resolution: each state converted to `T` before shifting; `ArgumentError` when `shift > _max_fock_modes(T)`. Size +18/-2.
- Assessment: `_max_fock_modes` is defined only for `FockNumber{I}`; a bare `T` such as `Int` gives `MethodError` (all in-tree callers pass `FockNumber`). The check runs after the loop (harmless). `CombineFockNumbersProcessor{FockNumber{Int}}` with 64 modes now throws; quality-control's `generate_states/big_int` benchmark did exactly that and only worked through the accidental promotion. Tested for `UInt64`, `Int`, `BigInt` and the error.
- Verdict: keep. Do not cherry-pick onto quality-control without `676ff2e4`.

### Symbolic states

#### f56a225d Return a Tuple-backed product, or 0, as the effect of an operator on a state
- Bottleneck: Phase 1 A6. `mul_effect` returned `amp * state`, allocating `NCMul(amp*1, [state])` each time.
- Resolution: `_scaled_state` returns `0` for zero amplitude, else `NCMul(amp * 1, (state,))`. Size +24/-2; one testitem.
- Gain: allocs -21% on projected_op_on_state (6927 to 5487) and krylov (10251 to 8091); time 0.94-0.98 (noise).
- Assessment: the reviewer compared old and new on zero, sum, bra, mixed-coefficient and double application: identical. `amp * 1` mirrors NCP's coefficient promotion (Bool to Int) and needs a comment pointing at it. Majorana and boson amplitudes covered only indirectly.
- Verdict: keep.

#### 47fd35a4 Act on symbolic states with a Tuple-backed one-factor product
- Bottleneck: Phase 1 A5. `NCMul(1, [op])` and `map` over factors allocate Vectors per application.
- Resolution: the term is `NCMul(1, (op,))`; precomputation works on a Tuple. Size +22/-1.
- Gain: allocs -29% (projected_op_on_state), -35% (krylov); krylov time 0.90.
- Assessment: every `apply_local_operators` / `precomputation_before_operator_application` overload must accept Tuple factors; a future overload annotated `AbstractVector` fails loudly. The testitem covers fermion, constrained, spin, boson and product spaces, kets and bras; Majorana and `FixedNumberFockState` are not covered (known baseline problems, section 5).
- Verdict: keep.

### Matrix representation

#### 12ab1ba0 Keep a single amplitude type in generic apply_local_operators
- Bottleneck: `amp = op.coeff; amp *= _amp` changed type in the loop (Int coefficient on spin/boson), giving a Union; heis12_Sz0 had 388k allocs.
- Resolution: accumulate in `T = mat_eltype(op)` via `convert(T, op.coeff)`, single `return`. Size +36/-10; one testitem.
- Gain: heis12_Sz0 0.36; 388397 to 2165 allocs.
- Assessment: relies on `mat_eltype(op)` covering the coefficient type; the test covers Int/Float/Complex/Rational with `@inferred` and both transpose branches. Fermion and Majorana factors not in the test.
- Verdict: keep.

#### 262980da Apply product-operator factors by tuple recursion
- Bottleneck: `foldl` over a `zip` of heterogeneous tuples with a closure (ps 17588 allocs).
- Resolution: `_apply_group_operators` recurses with `first`/`Base.tail`. Size +11/-43 (also deletes commented-out code).
- Gain: ps 0.53, slow_path 0.64; allocs 17588 to 3764, 73631 to 47135.
- Assessment: drops the `ismissing(op)` skip; safe if `ProductOperator.ops` never contains `missing` (true for `partition_product` output, unenforced, no dedicated test). Used together with `69c8c641`.
- Verdict: keep.

#### 69c8c641 Partition product operators by looking up each factor's basis
- Bottleneck: `partition_product` scanned all factors once per basis.
- Resolution: look up each factor's basis index once (`_basis_index`), group factors per used index. Size +20/-15.
- Gain: ps_many_cons 0.62 (12169 to 4005 allocs); ps_many 0.91 and ps 0.92 are noise.
- Assessment: `ntuple(length(inds))` with runtime length stays type-unstable at construction. Covered only indirectly.
- Verdict: keep.

#### 639f64ea Use one accumulator for sums on constrained multi-group spaces
- Bottleneck: per-term matrices plus a tree sum on constrained/sector product spaces.
- Resolution: new `__matrix_representation(::NCAdd, ...)` for non-`ProductSpace`: partition all terms, accumulate in the promoted eltype, add the identity via `add_identity!!`. Size +56/-1; test "Constrained product space agrees with the unconstrained one".
- Gain: bosehub 0.65 (952 to 499 allocs); heis12 0.83 (edge of noise).
- Assessment: summation order changes (`≈`, not `==`). `tree_split` is ignored on this path. The reviewer found the identity term was always built; addressed in this commit (built as a cheap `Eye` for its eltype only, scaled only when added). A zero-coefficient NCAdd on constrained multi-group spaces keeps the previous eltype promotion. Not covered: empty NCAdd, sector spaces over products, BigFloat.
- Verdict: keep.

#### 5fbbfb6b Build product-space term matrices with a direct identity kron
- Bottleneck: `foldl(kron, ...)` with `Diagonal` identity padding on an unconstrained `ProductSpace`.
- Resolution: for sparse repr, kron over the active span, then `_kron_identities` writes `kron(I(l), core, I(r))` directly in CSC. Size +64/-5.
- Gain: ps_many 0.74 (rerun; first run 0.99/1.13); mixed_product 0.96. Noisy.
- Assessment: indexing checked and correct. `SparseMatrixCSC` only (`AbstractSparseMatrixCSC` is non-public); other sparse types use the generic path. The eltype is taken from the first inactive group. `-0.0` entries are no longer normalized (cosmetic).
- Verdict: optional; about 60 lines for one noisy benchmark.

#### e3ed89fc Drop the identity of a sum without re-filtering its terms
- Bottleneck: `remove_identity` re-filtered the whole dictionary.
- Resolution: `NCAdd(...; filter_zeros=false, filter_scalars=false)` sharing the dict. Size +3/-1.
- Gain: free_fermion/bdg 0.90, no allocation change (noise).
- Assessment: relies on the NCAdd invariant; the result aliases the input dict (the comment about callers only reading it may rot). A test checks the operator is unchanged.
- Verdict: consider dropping.

#### 479f8952 Check single-particle and BdG operators in one pass
- Bottleneck: `_isfermionic` + `isquadratic` + `isnumberconserving` made several passes over about 2000 terms.
- Resolution: `_is_quadratic_fermion_operator(op, number_conserving)` checks each term once; `_isfermionic` removed. Size +40/-3.
- Gain: free_fermion 0.80, bdg 0.86 (no allocation change).
- Assessment: quadratic Majorana operators previously raised `FieldError`; now the documented `ArgumentError`. Assumes dict keys are `NCMul` with `.factors`. The testitem covers error cases including Majorana and spin mixes.
- Verdict: keep.


### Lazy operators

#### afab0e85 Store per-term precomputation in lazy NCAdd operators
- Bottleneck: every lazy NCAdd `mul!` recomputed the per-term precomputation (42-106 allocs per call).
- Resolution: `lazy_precomputation(op::NCAdd, space)` stores `(term, coeff, precomp)` at construction; `lazy_mul!` and the chunked methods iterate it. Size +20/-13.
- Gain: dim-64 `mul!` 0.85, allocs 42 to 0; 12-site chain 0.97; sparsevec 0.58. Hubbard construction allocs 899 to 1011.
- Assessment: `L.precomp` for NCAdd is now a term vector; adjoint/transpose reuse it with `TransposedSpace`, which assumes per-term precomputation is space-independent (already assumed for NCMul); verified on a fermion-fermion product space. Coefficients are held twice (`op.dict` and the tuples). Code that treats `L.precomp` as plain precomputation breaks silently for NCAdd. No new test.
- Verdict: keep.

#### 640d93e3 Build the lazy multi-space sum as one AddedOperator
- Bottleneck: `sum(terms)` of lazy operators nested `AddedOperator`s (6293 allocs, about 7 s compile in Liouvillian construction).
- Resolution: `_lazy_sum` wraps matrices in `MatrixOperator` and builds one `AddedOperator(Tuple(...))`. Size +10/-6.
- Gain: construction 0.63, 6293 to 4876 allocs; TTFX construction 7.0 to about 3.3 s.
- Assessment: a single element is returned unwrapped, as `sum` does; an empty operator throws. Type-unstable at construction (one time).
- Verdict: keep.

#### 2e1072ef Overwrite the output for zero beta in lazy NCAdd mul!
- Bottleneck: bug: `rmul!(y, 0.0)` leaves NaN/Inf in `y`; NCMul and ProductOperator methods already fill zeros.
- Resolution: `iszero(β) ? fill!(y, zero(T)) : rmul!(y, β)` in the three NCAdd `lazy_mul!` methods. Size +23/-6; tests for NaN output with β=0 on base, term and state chunking.
- Assessment: matches BLAS semantics. The pattern is duplicated three times. Can be cherry-picked on its own.
- Verdict: keep.

#### 58041ec1 Group single-group terms of lazy product-space operators
- Bottleneck: a 45-term Liouvillian gave a 45-element `AddedOperator`; `cache_operator` took 13 s of compile time and `mul!` allocated heavily.
- Resolution: new `__matrix_representation(::NCAdd, bases, ::ProductSpace, ::LazyRepr, ...)`: sum single-group terms into one local NCAdd per group, embed with `IdentityOperator` kron factors; multi-group terms stay per-term. Size +94 (about 47 code, 47 test).
- Gain: Liouvillian mul! 0.87, sparse mul 0.50 (8914 to 3125 allocs), cache_operator 0.28 (13 s to 1.3 s), construction 0.55. A small mixed 3-group operator is 1.15x slower (84 to 128 allocs).
- Assessment: `convert(Vector{S}, term.factors)` with `S` the `typejoin` of factor types can be abstract for mixed labels (dynamic dispatch in the inner loop; correct, unmeasured). `T=mat_eltype(op)` is passed before `kwargs...`, so a caller-supplied `T` would give a duplicate-keyword error (not reproduced). Lazy vs eager agree on odd single-group terms, but the commit's tests use odd operators only in multi-group terms. Summation order changes (relative error up to 3.3e-16). Not covered: constrained product spaces. The test asserts `length(ops) == 2+N`, coupling it to the structure.
- Verdict: keep.

#### b454581f Collect lazy sparse-vector products as index/value pairs
- Bottleneck: `L * sparsevec` used `similar(x, T)` plus per-entry insertion (115 allocs at 5% density).
- Resolution: `Base.:*(::LazyOperator, ::AbstractSparseVector)` collects `(outind, amp)` per term and calls `sparsevec` (sums duplicates); `_lazy_terms` unifies NCAdd and single-term operators. Size +42.
- Gain: single-space sparsevec 0.30 (5%), 0.70 (0.1%); 115 to 41 allocs; product-space Liouvillian unchanged.
- Assessment: duplicates the semantics of `_apply_single_term!`; ignores `StateChunking`. An out-of-space image without projection now throws `ArgumentError` instead of `BoundsError` (tested). Stored zeros of the input are skipped; cancellations may leave stored zeros. Sparse-matrix input untested for the new path.
- Verdict: keep.

### Follow-up commits

#### fe2fc1fe Support a global phase in the weights of permutation sectors
- Bottleneck: bug found in the review of `f02237a1`, present on quality-control: `_trace_rank` used `real(tr B)^2`, returning 0 columns for weights with a complex overall phase (`im .* [1, 1]` gave 0 instead of 3). `:eig` additionally needs a Hermitian input.
- Resolution: rank cap `abs2(tr(B))/sum(abs2, B)`; for `:eig` the trace phase is divided out before the Hermitian eigendecomposition. Size +20/-3; the testitem covers `im` and `cis(pi/4)` weights with `:qr` and `:eig`.
- Assessment: non-group weights remain untested.
- Verdict: keep.

#### 13c9339f Add sparse methods for transpose, comparison, scaling and display of SuperOperatorCOO
- Bottleneck: the review of `9b39d9a4` showed that `transpose`, `==`, `≈`, scalar `*`, `Matrix` and `show` on a `SuperOperatorCOO` fell back to generic code that densified or visited all `nrows * D^2` entries.
- Resolution: `transpose` shares the adjoint kernels (`_entry_op` selects `conj` or `identity`); `==`/`isapprox` go through `sparse`; `Matrix`/`Array`, `show`, `norm`, scalar `*` and `/` and negation (adjoint/transpose wrappers included) act on stored values. Size +91/-5, one file. The testitem "SuperOperatorCOO operations stay sparse" compares against `SparseMatrixCSC` for `A`, `A'` and `transpose(A)` and bounds allocations below 8 MB where a dense matrix would take 36 MB.
- Assessment: `A + B` and matrix-matrix products still use the generic `AbstractMatrix` path. Not independently reviewed by a stream reviewer.
- Verdict: keep with `9b39d9a4`.

## 4. Behaviour and API changes

- `PartialTraceMap`/`EmbedMap` `.map` is a `SuperOperatorCOO` (was `SparseMatrixCSC`); `sparse(::EmbedMap)` returns a `SparseMatrixCSC` (was an `Adjoint`). `.map` is undocumented; `sparse(op)` gives the CSC. `PartialTransposeMap` still uses CSC (`9b39d9a4`, `13c9339f`).
- `catenate_fock_states` returns the requested type and throws `ArgumentError` when the modes do not fit; a bare integer `T` gives `MethodError` (`0f5c7b7f`).
- `combine_states` for a fermionic partition of the first modes of a space with 64 or more modes returns `FockNumber{BigInt}` (was `UInt64`) (`4e2a7cce`).
- Error types: a wrong-length permutation gives `ArgumentError` from `_match_perm`; a quadratic Majorana operator in single-particle/BdG gives `ArgumentError` (was `FieldError`); lazy `L * sparsevec` with an out-of-space image and no projection gives `ArgumentError` (was `BoundsError`).
- `valid_branch`: the 4-argument `ProductConstraint` method is gone; `AdditiveBranchConstraint` has 5- and 4-argument methods (non-public protocol) (`16453ff2`).
- Sector labels of product constraints are the `typejoin` of the label types (`009311ca`).
- Numerical changes: summation order differs on constrained multi-group sums (`639f64ea`) and lazy grouped terms (`58041ec1`), so results are `≈`, not `==`; permutation sector bases have a different, equally valid column basis (`:symmetric`/`:antisymmetric` orbit path; `:eig` relative tolerance); `-0.0` entries are no longer normalized (`5fbbfb6b`); `hilbert_space` of an empty grouping may have a different `SectorHilbertSpace` type parameter (`919ec87a`).
- `SpinState` hash values changed (not persisted anywhere).
- `lazy_precomputation` of an NCAdd is a vector of `(term, coeff, precomp)` (`afab0e85`).
- Benchmarks: 13 keys removed or renamed in `676ff2e4`, so AirspeedVelocity history for those names ends; `precomputed_vs_eager.jl` deleted. quality-control's own leaf `generate_states/big_int` (`FockNumber{Int}` with 64 modes) throws after `0f5c7b7f`; it is replaced by `generate_states/hilbert_space_N60|N64_bigint`. `majorana_hilbert_space` is replaced by `hilbert_space` in the benchmark.

## 5. Bugs

Found and fixed on this branch:
- Absolute orthogonalization tolerances dropped sector vectors for orbits larger than 100 states (present on quality-control) (`f02237a1`).
- `real(tr)^2` rank cap returned 0 columns for weights with a global phase (present on quality-control) (`fe2fc1fe`).
- Regression introduced in quality-control: roundoff check visited all d^2 entries of a sparse projector (`419efd7a`).
- `catenate_fock_states` promoted to `UInt64` regardless of the requested type (`0f5c7b7f`).
- Lazy NCAdd `mul!` with `beta = 0` left NaN/Inf from the old output (`2e1072ef`).
- Quadratic Majorana operators passed the fermion check and failed with `FieldError` (`479f8952`).
- Generic `apply_local_operators` changed amplitude type in its loop (`12ab1ba0`).
- Defects found in review and fixed inside the commits: mixed numeric sector labels promoted (`009311ca`), identity term always built (`639f64ea`), missing 4-argument `valid_branch` (`16453ff2`), untyped hit buffer (`fdbd128c`).
- NCP (separate repo): a coefficient rounding to zero left a `0.0` term in `add_term!!`; scaling with underflow left zero terms in `add!` (section 7).

Pre-existing bugs found and not fixed:
- Majorana bras: `ket' * γ[1]` gives `+-im` phases where the matrix product is real; the `transpose=true` path of `apply_local_operators(::NCMul, ::FockNumber, ::MajoranaHilbertSpace, ...)` (`majoranas.jl` about line 282). Kets are correct.
- `group_id` is missing for `SingleParticleHilbertSpace`; operators cannot act on its symbolic kets, and on a constrained `FixedNumberFockState{1}` space `f[1]*ket` yields a state outside the space.
- `BdGHilbertSpace` `matrix_representation` does not canonicalize `repr`; `:dense` or `ComplexF32` give a `MethodError`.
- Majorana x spin product: for `tensor_product(majorana space, spin)`, `s[:x]*m[1]*m[3]` is correct for `L*v` but wrong for `L'*v` and `transpose(L)*v`, on the parent too.
- Non-group weights: the trace-rank cap gives fewer columns than rank(P) (documented, untested).
- `_apply_ptmap_on_vec` allocates `zeros(eltype(ψ), ...)`, so a map with complex entries applied to a real vector throws `InexactError` (all maps the package builds are real).
- On a constrained product space a typed `Float32` repr rounds per product term, while the unconstrained path rounds per group matrix (last-bit differences); whether an `Int` typed repr throws can depend on the space with `projection=true`.
- `get_input(::LazyRepr{<:AbstractArray}, H)` is called per factor space, so a full-size user input throws on product spaces.

## 6. Deferred and rejected items

- B17 (sparse `generalized_kron` copy): rejected; in-place assembly gave 0.85-0.94 with no allocation change.
- B18 "Build sector spaces with typed and presized containers": implemented, then dropped from this branch during the final A/B. It sped up building small sectors (hilbert_space 0.68) but made every later `state_index` lookup on the resulting sector spaces 1.4-1.9x slower (state_index/constrained 35 to 49-57 us, state_index/product_space 43 to 80-83 us, same process, commit vs parent), which also slowed matrix_representation/standard and hubbard/matrix_representation by about 1.15x. The suspected cause is the `sizehint!`-presized state-to-index `OrderedDict`, whose slot table ends up larger and sparser than one built from `zip`. Kept on branch `perf/integration-pre-drop`. A version that keeps only the typed `groupby` part could be re-measured separately.
- B22 (BdG `normal_order_to_bdg` from stored entries): implemented, rejected; allocs 229 to 50 but bytes and time unchanged (bdg 1.10-1.12).
- B27 (complement bookkeeping): rejected as a separate fix; no safe general shortcut for constrained H; B16 already speeds it by 1.37x.
- B16 caller-supplied mapper into partial trace/embed: deferred (API change; saves about 20 us per call).
- E3 (native sparse `L * SparseVector` on product spaces): deferred; needs a new wrapper type and changes the public return type; E1 already halves the sparse path.
- E5 (zero check before the coefficient multiply): rejected; Phase 1's 0.60x did not reproduce (within +-5%).
- A9 (cheaper `hash(::FermionSym)`): implemented, rejected (3-5%, no allocation change); kept on branch `perf/s6-symbolic-a9-rejected` (2d19d7c6). A cached basis hash (needs a struct field) is deferred as a design decision. A10 (`with(_autosort => false)` on every sort) and a `mul_effect(::FermionSym, ::FermionSym)` reordering were rejected.
- G-5 (avoid converting N site states to `Int` and back per term; accumulator size hint for spin chains): not attempted.
- G-6 (`tensor_product` re-evaluates the combined sector function): outside the stream's files, not done.
- Pivoted Cholesky alternative to the custom Gram-Schmidt, and partial transpose on COO storage: not done.
- Returning the bare state when `isone(amp)` (Phase 1 `fhs_state2`): rejected; it would change the coefficient type.

## 7. NonCommutativeProducts.jl changes (separate repo, local branch `perf/ncp-symbolic`, not part of this PR)

| Hash | Title | +/- | Gain | Verdict |
|---|---|---|---|---|
| a56cde8 | Sort powers only for exponents 0 and 1 | +1 src, 39 test | sum_square 0.72, bdg_square 0.76; allocs 10543 to 7961 | keep |
| 94e89dc | Convert a new coefficient to the value type before the zero check in add_term!! | +4/-2 src, 2 tests | none (bug fix) | keep |
| ff99d2d | Hash each new key of a sum once in add_term!! | +21 src, 44 test | deep_product 0.89, heisenberg12 0.85, hubbard12 0.89 | keep with follow-up |
| 41ad5cf | Size the Dict of sorted terms for all terms up front | +3/-1 | deep_product 0.75, bdg_prod 0.89 | keep with follow-up |
| 3fb19cf | Remove terms that underflow to zero when add! scales a sum | +2/-1, 3 tests | none (bug fix) | keep |
| b0786c5 | Skip the filter pass when adding two sums with the same coefficient type | +5/-2, 5 tests | symbolic/sum 0.82 (edge of noise) | keep with follow-up |

Condensed assessment (s6 review):
- `a56cde8`: the invariant "`*` sorts whenever autosort is on" holds and is enforced by a test over p=0:5, global and scoped settings.
- `ff99d2d`: uses `Base.ht_keyindex2_shorthash!`, the 5-argument `Base._setindex!`, `Base._delete!`, and writes `age`, `keys`, `vals` directly. The `@static` guard checks names and a method of the right arity, but `_setindex!` has a defaulted 5th argument, so it cannot detect a change in meaning; a partial Dict change that keeps the names could corrupt Dicts silently. The 2000-step randomized reference test is the real safeguard, and it is effective only if NCP CI runs on nightly (compat `julia = "1.12"`, verified on 1.12.7 and 1.13.2). The measured allocation drops (heisenberg12, hubbard12) likely come from converting values that fit, which a public-API version could also do; the "hash once" part is unproven at about 11%. Follow-up: benchmark a `get` plus `setindex!`/`delete!` variant, keep the internals only if clearly faster, and then add a nightly CI job. Three internals were added to the ExplicitImports ignore list.
- `41ad5cf`: `length(sorted_terms)` is an upper bound; heavily cancelling products (for example the cross terms of `(sum c_i + c_i')^2`) keep an oversized slot table, making later copies and iteration O(capacity). Not measured. Follow-up: shrink when the length is below a quarter of the capacity. Floating-point results differ in the last bits (krylov `≈`).
- `94e89dc`, `3fb19cf`, `b0786c5` form a chain: `b0786c5` skips the zero filter when `C1 === promote_type(C1, C2)` and relies on the invariant "no zero values, no scalar keys", which the two bug fixes establish. The invariant is not asserted anywhere; a constructor with `filter_zeros=false` would now spread zeros through `+`. Follow-up: an `iscanonical` check in the randomized test. Do not reorder or partly revert the three.
- `3fb19cf` costs a second O(n) pass for exact types.
- No `@inbounds`, no restrictive annotations. The FHS commits `f56a225d` and `47fd35a4` do not depend on this branch; NCP gains reach FHS only after an NCP release or dev checkout.

## 8. Final suite comparison

Minimum time over two alternating runs of each side (`benchmark/compare.jl`), quality-control 1cee0550 vs this branch, same machine. Ratios within 0.85-1.15 are inside same-code noise; allocation counts are exact.

| benchmark | qc min | PR min | ratio | qc allocs | PR allocs |
|---|---|---|---|---|---|
| bosons/hilbert_space | 305.600 μs | 118.200 μs | 0.387 | 6173 | 982 |
| bosons/matrix_representation | 442.000 μs | 212.900 μs | 0.482 | 962 | 499 |
| complement/fermions | 56.600 μs | 56.600 μs | 1.000 | 604 | 432 |
| complement/product space | 67.800 μs | 46.400 μs | 0.684 | 1014 | 782 |
| embed/fermions | 777.600 μs | 800.500 μs | 1.029 | 268 | 96 |
| embed/precomputed/apply | 1.041 ms | 770.500 μs | 0.740 | 5 | 5 |
| embed/precomputed/apply_inplace | 317.700 μs | 94.300 μs | 0.297 | 2 | 2 |
| embed/precomputed/construct | 1.521 ms | 63.800 μs | 0.042 | 254 | 92 |
| embed/precomputed/eager | 575.400 μs | 502.300 μs | 0.873 | 265 | 93 |
| embed/product space | 56.800 μs | 34.700 μs | 0.611 | 693 | 438 |
| generate_states/hilbert_space_N60 | 10.864 ms | 3.156 ms | 0.290 | 52983 | 52738 |
| generate_states/hilbert_space_N64_bigint | 46.174 ms | 44.193 ms | 0.957 | 1505725 | 1505484 |
| hilbert_space | 95.000 μs | 101.400 μs | 1.067 | 145 | 145 |
| hubbard/constrain_no_double_occupation | 19.439 ms | 444.400 μs | 0.023 | 134980 | 12897 |
| hubbard/constrain_no_double_occupation_N30 | 85.245 ms | 4.771 ms | 0.056 | 594734 | 56971 |
| hubbard/matrix_representation | 1.381 ms | 1.344 ms | 0.973 | 237 | 237 |
| hubbard/matrix_representation_projected | 699.400 μs | 696.400 μs | 0.996 | 231 | 231 |
| hubbard/space | 15.384 ms | 491.300 μs | 0.032 | 82398 | 11581 |
| hubbard/symbolic | 37.400 μs | 38.500 μs | 1.029 | 2177 | 2177 |
| lazy/construct | 46.400 μs | 52.200 μs | 1.125 | 899 | 1011 |
| lazy/mul! | 821.600 μs | 788.200 μs | 0.959 | 106 | 0 |
| lazy/sparse_mul!_reference | 16.600 μs | 15.900 μs | 0.958 | 0 | 0 |
| majorana/matrix_representation | 747.600 μs | 780.600 μs | 1.044 | 190 | 190 |
| matrix_application/lazy liouvillian/mul sparse | 2.519 ms | 1.259 ms | 0.500 | 9216 | 3391 |
| matrix_application/lazy liouvillian/mul! | 179.500 μs | 152.700 μs | 0.851 | 184 | 79 |
| matrix_representation/bdg | 636.800 μs | 543.800 μs | 0.854 | 229 | 229 |
| matrix_representation/free_fermion | 316.700 μs | 207.300 μs | 0.655 | 23 | 23 |
| matrix_representation/product space | 546.900 μs | 217.700 μs | 0.398 | 17588 | 2063 |
| matrix_representation/product space many spaces | 1.726 ms | 1.258 ms | 0.729 | 9256 | 5738 |
| matrix_representation/product space many spaces constrained | 722.000 μs | 361.700 μs | 0.501 | 12169 | 3051 |
| matrix_representation/standard | 1.282 ms | 1.306 ms | 1.018 | 169 | 169 |
| open_system/liouvillian_sparse | 2.462 ms | 2.201 ms | 0.894 | 3932 | 4068 |
| open_system/liouvillian_symbolic | 63.000 μs | 61.900 μs | 0.983 | 3437 | 3437 |
| partial_trace/fermions/map | 71.378 ms | 682.200 μs | 0.010 | 254 | 93 |
| partial_trace/fermions/pure_state | 601.000 μs | 285.700 μs | 0.475 | 245 | 76 |
| partial_trace/precomputed/apply_matrix | 137.600 μs | 2.800 μs | 0.020 | 5 | 5 |
| partial_trace/precomputed/apply_matrix_inplace | 137.300 μs | 2.700 μs | 0.020 | 2 | 2 |
| partial_trace/precomputed/apply_pure_state | 94.600 μs | 5.000 μs | 0.053 | 4 | 4 |
| partial_trace/precomputed/apply_pure_state_inplace | 93.800 μs | 4.500 μs | 0.048 | 0 | 0 |
| partial_trace/precomputed/apply_vectorized_inplace | 137.200 μs | 2.400 μs | 0.017 | 0 | 0 |
| partial_trace/precomputed/construct | 1.561 ms | 62.800 μs | 0.040 | 253 | 91 |
| partial_trace/precomputed/eager_matrix | 576.300 μs | 28.700 μs | 0.050 | 232 | 63 |
| partial_trace/precomputed/eager_pure_state | 550.400 μs | 40.800 μs | 0.074 | 244 | 75 |
| partial_trace/product space/apply_precomputed | 2.950 μs | 700.000 ns | 0.237 | 4 | 4 |
| partial_trace/product space/map | 234.100 μs | 26.000 μs | 0.111 | 15467 | 436 |
| partial_trace_algorithms/Full space/Dense/FullAlg | 1.490 ms | 847.900 μs | 0.569 | 210 | 66 |
| partial_trace_algorithms/Full space/Dense/SubAlg | 32.200 μs | 23.500 μs | 0.730 | 207 | 66 |
| partial_trace_algorithms/Full space/Dense/default | 33.300 μs | 22.900 μs | 0.688 | 207 | 68 |
| partial_trace_algorithms/Full space/Sparse/FullAlg | 49.800 μs | 18.700 μs | 0.376 | 220 | 67 |
| partial_trace_algorithms/Full space/Sparse/SubAlg | 79.500 μs | 76.600 μs | 0.964 | 208 | 67 |
| partial_trace_algorithms/Full space/Sparse/default | 80.700 μs | 18.700 μs | 0.232 | 208 | 69 |
| partial_trace_algorithms/Sparse space/Dense/FullAlg | 572.800 μs | 194.600 μs | 0.340 | 487 | 63 |
| partial_trace_algorithms/Sparse space/Dense/SubAlg | 5.183 ms | 184.000 μs | 0.036 | 484 | 66 |
| partial_trace_algorithms/Sparse space/Dense/default | 578.600 μs | 174.300 μs | 0.301 | 487 | 66 |
| partial_trace_algorithms/Sparse space/Sparse/FullAlg | 25.000 μs | 12.600 μs | 0.504 | 497 | 64 |
| partial_trace_algorithms/Sparse space/Sparse/SubAlg | 5.208 ms | 236.900 μs | 0.045 | 485 | 67 |
| partial_trace_algorithms/Sparse space/Sparse/default | 25.300 μs | 12.800 μs | 0.506 | 497 | 64 |
| partial_transpose/apply | 353.000 μs | 349.500 μs | 0.990 | 7 | 7 |
| partial_transpose/construct | 2.814 ms | 2.790 ms | 0.991 | 269 | 153 |
| partial_transpose/eager | 1.253 ms | 1.232 ms | 0.983 | 185 | 69 |
| permutations/few_large | 25.489 ms | 868.500 μs | 0.034 | 874 | 319 |
| permutations/many_small | 552.600 μs | 21.700 μs | 0.039 | 9920 | 423 |
| reshape/mat_to_tens | 31.000 μs | 17.800 μs | 0.574 | 498 | 151 |
| reshape/mat_to_vec | 67.900 μs | 54.200 μs | 0.798 | 4362 | 89 |
| reshape/precomputed/apply | 17.025 μs | 10.000 μs | 0.587 | 6 | 11 |
| reshape/precomputed/apply_inplace | 11.625 μs | 6.525 μs | 0.561 | 4 | 9 |
| reshape/precomputed/construct | 15.025 μs | 8.650 μs | 0.576 | 513 | 163 |
| reshape/repeat | 182.600 μs | 87.000 μs | 0.476 | 557 | 222 |
| sectors/construct_all | 76.700 μs | 81.400 μs | 1.061 | 168 | 168 |
| sectors/matrix_representation_sectors | 636.100 μs | 630.300 μs | 0.991 | 121 | 121 |
| sectors/sectors | 36.500 μs | 37.300 μs | 1.022 | 73 | 73 |
| spin/hilbert_space_Sz0 | 3.972 ms | 819.200 μs | 0.206 | 67633 | 1994 |
| spin/matrix_representation_heisenberg | 7.255 ms | 2.614 ms | 0.360 | 2812 | 1348 |
| spin/symbolic_heisenberg | 64.600 μs | 66.100 μs | 1.023 | 2630 | 2630 |
| state_index/constrained | 36.700 μs | 31.200 μs | 0.850 | 0 | 0 |
| state_index/product_space | 37.200 μs | 39.400 μs | 1.059 | 0 | 0 |
| symbolic/deep_product | 531.200 μs | 545.400 μs | 1.027 | 2424 | 2424 |
| symbolic/sum | 139.600 μs | 140.100 μs | 1.004 | 5417 | 5417 |
| symbolic/sum_square | 717.000 μs | 700.200 μs | 0.977 | 10393 | 10393 |
| symbolic_states/op_on_ket | 293.500 μs | 252.300 μs | 0.860 | 14888 | 7668 |
| symbolic_states/parse_kets | 49.800 μs | 49.100 μs | 0.986 | 4003 | 4003 |
| symbolic_states/projected_op_on_state | 164.900 μs | 154.300 μs | 0.936 | 7008 | 3984 |
| symbolic_states/vector_representation | 218.700 μs | 207.500 μs | 0.949 | 2492 | 2492 |
| tensor_product/generalized_kron | 5.401 ms | 5.278 ms | 0.977 | 196 | 66 |
| tensor_product/matrices | 3.092 ms | 3.087 ms | 0.998 | 1130 | 870 |
| tensor_product/spaces | 6.320 μs | 6.240 μs | 0.987 | 216 | 216 |

