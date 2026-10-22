# Phase 1: Performance bottlenecks in FermionicHilbertSpaces.jl

Profiled at commit `660c67f4` (branch `perf-review`). Nothing in `src/` or `benchmark/` was changed during this phase.

## 1. Environment and method

- **Machine:** Julia 1.13.2 on Windows 11. CPU is an i7-13700T (8 P-cores, 8 E-cores, 35 W, 1.4–4.9 GHz).
- **Benchmark env:** `benchmark/Manifest.toml` is stale. It pins NonCommutativeProducts (NCP) 0.4.7, but the package needs 0.5, so `julia --project=benchmark` fails to precompile. Every run used a scratch env that devs both the package and the local NCP at `C:\Users\svens\Projects\NonCommutativeProducts.jl` (0.5.0):
  `C:\Users\svens\AppData\Local\Temp\claude\c--Users-svens-Projects-FermionicHilbertSpaces-jl\f27290bf-fcfe-4dfb-b0e3-3e1ec0e36974\scratchpad\phase1\benchenv`
- **Tools env:** JET lives in the sibling `toolsenv`. Stack it with `JULIA_LOAD_PATH="@;<toolsenv>;@stdlib"` and run Julia with `--startup-file=no`.
- **Baseline:** a full serial run of `SUITE`, saved to `perf_review/baseline.json`.
- **Profiling:** six parallel subagents, one per area, using Base `Profile` (flat and tree), `Profile.Allocs`, JET `@report_opt` and `@code_warntype`.
  - Each case got 1,000–5,000 samples. The Windows sampler delivers only about a fifth of the requested rate.
  - Candidate fixes were tried by `@eval`-redefining methods in scratch processes, and each patched result was compared against the unpatched one.
  - **All speedups below are ratios measured within one process.** Absolute times from the parallel runs are not baselines.
- **Raw material:** scripts and profile dumps are in the `scratchpad\phase1\{a_symbolic,b_matrep,c_ptrace,d_reshape_gen,e_lazy,f_perm,g_gaps,h_hygiene}` folders under the session scratchpad.
- **Timing overhead:** `using FermionicHilbertSpaces` takes 0.89 s. Including the suite file takes 58 s, mostly compilation; `tune!` takes 74 s and `run` 50 s.

### Baseline (serial run; noise is high, see H01)

| benchmark | median | min | allocs | memory | samples |
|---|---|---|---|---|---|
| complement/fermions | 79.4 µs | 60.8 µs | 589 | 69.3 KiB | 9371 |
| complement/product space | 148.7 µs | 82.3 µs | 1068 | 51.2 KiB | 6740 |
| embed/fermions | 1.6 ms | 859.3 µs | 256 | 2.61 MiB | 561 |
| embed/product space | 130.2 µs | 66.3 µs | 743 | 83.5 KiB | 7523 |
| generate_states/big_int | 14.55 ms | 12.28 ms | 6663 | 352.2 KiB | 62 |
| generate_states/int | 16.46 ms | 9.81 ms | 5569 | 324.8 KiB | 66 |
| hilbert_space | 196.6 µs | 108.4 µs | 145 | 400.4 KiB | 4344 |
| matrix_application/lazy liouvillian/mul sparse | 4.66 ms | 2.62 ms | 12750 | 3.32 MiB | 210 |
| matrix_application/lazy liouvillian/mul! | 225.95 µs | 206.4 µs | 183 | 6.8 KiB | 3758 |
| matrix_representation/bdg | 993.4 µs | 608.0 µs | 228 | 2.33 MiB | 825 |
| matrix_representation/free_fermion | 436.05 µs | 282.5 µs | 22 | 461.6 KiB | 2014 |
| matrix_representation/product space | 917.9 µs | 568.6 µs | 17598 | 992.1 KiB | 909 |
| matrix_representation/product space many spaces | 3.22 ms | 1.89 ms | 9272 | 4.61 MiB | 252 |
| matrix_representation/product space many spaces constrained | 1.25 ms | 807.6 µs | 12193 | 1.01 MiB | 696 |
| matrix_representation/standard | 2.26 ms | 1.37 ms | 169 | 3.49 MiB | 398 |
| partial_trace/fermions/map | 130.73 ms | 105.03 ms | 242 | 257.26 MiB | 8 |
| partial_trace/fermions/standard | 88.9 µs | 54.2 µs | 234 | 238.6 KiB | 8368 |
| partial_trace/product space/map | 502.9 µs | 267.4 µs | 15517 | 697.9 KiB | 1796 |
| partial_trace/product space/standard | 489.0 µs | 251.8 µs | 15493 | 613.7 KiB | 1666 |
| pt_alg/default=FullAlg/Sparse space/Dense/FullAlg | 864.55 µs | 715.8 µs | 1646 | 179.6 KiB | 924 |
| pt_alg/default=FullAlg/Sparse space/Dense/SubAlg | 6.38 ms | 5.63 ms | 1643 | 169.3 KiB | 141 |
| pt_alg/default=FullAlg/Sparse space/Sparse/FullAlg | 103.6 µs | 55.5 µs | 1657 | 239.1 KiB | 7594 |
| pt_alg/default=FullAlg/Sparse space/Sparse/SubAlg | 8.39 ms | 5.83 ms | 1645 | 169.4 KiB | 130 |
| pt_alg/default=SubAlg/Full space/Dense/FullAlg | 1.96 ms | 1.82 ms | 497 | 51.6 KiB | 400 |
| pt_alg/default=SubAlg/Full space/Dense/SubAlg | 60.3 µs | 44.9 µs | 494 | 27.5 KiB | 10000 |
| pt_alg/default=SubAlg/Full space/Sparse/FullAlg | 123.3 µs | 65.4 µs | 508 | 384.6 KiB | 5607 |
| pt_alg/default=SubAlg/Full space/Sparse/SubAlg | 195.9 µs | 123.7 µs | 496 | 27.6 KiB | 5466 |
| permutations/few_large | 10.46 ms | 6.61 ms | 873 | 16.26 MiB | 88 |
| permutations/many_small | 1.18 ms | 621.4 µs | 9923 | 1.55 MiB | 690 |
| reshape/mat_to_tens | 59.8 µs | 34.4 µs | 482 | 97.1 KiB | 10000 |
| reshape/mat_to_vec | 133.0 µs | 77.8 µs | 4346 | 489.3 KiB | 5616 |
| reshape/mixed_split_combine | 71.5 µs | 34.1 µs | 482 | 97.0 KiB | 10000 |
| symbolic/deep_product | 861.5 µs | 567.5 µs | 2424 | 653.7 KiB | 1036 |
| symbolic/sum | 284.6 µs | 153.7 µs | 5417 | 783.2 KiB | 2661 |
| symbolic/sum_square | 1.23 ms | 726.1 µs | 10393 | 1.34 MiB | 734 |
| symbolic_states/projected_op_on_state | 328.8 µs | 176.9 µs | 7008 | 297.2 KiB | 2784 |

`pt_alg` stands for `partial_trace_algorithms`. Every leaf tunes to `evals=1`.

## 2. Ranked bottlenecks

Ranking weighs the measured speedup, how many benchmarks and user paths are affected, and confidence.

- **Repo:** **FHS** means this package (`src/`); **NCP** means NonCommutativeProducts.jl, a separate repository.
- **Risk** is about correctness and API, not performance.

### Tier 1: large measured wins

#### B01: `symmetric_sector` averages over all n! permutations, then runs a sparse QR (FHS)
- **Affects:** permutations/many_small, permutations/few_large, and any `symmetric_sector` call.
- **Location:** `src/permutation_symmetry.jl:55-160` (`permutation_projector`, `_remove_columns!`).
- **Evidence:**
  - Cost scales as O(n!·dim) for applying the permutations, plus sparse QR and a dense d×rank `Q`.
  - In few_large, `_remove_columns!` is 75% of the time; SPQR is 32% and `F.Q[invperm(prow),1:M]` is 43%. The dense `Q` accounts for 8 of the 16 MiB.
  - 8×spin-½ at m=3 has dim 8 and still takes 123 ms.
- **Why it is slow:** `permute_state` applies no phases, so each group element acts as a permutation matrix on basis states. For the full group S_n the symmetric projector on an orbit O is `ones(|O|,|O|)/|O|`. The orthonormal basis is therefore one column per orbit, the orbit indicator divided by √|O|. Group averaging plus QR rebuilds this closed form at great cost.
- **Fix:**
  - Find orbits with union-find, using the n−1 adjacent transpositions as generators. That is O(dim·n) calls to `permute_state` and `state_index`.
  - Build `Q` from (row, orbit, 1/√|O|) entries.
  - For `:antisymmetric`, use sign(g) entries and drop any orbit whose state has two equal substates.
  - The prototype is in `f_perm/exper2.jl`. `_union_orbits!` acts as a function barrier.
- **Measured** (`:symmetric` only; results match: `Q*Q' ≈ ref*ref'` and the ranks agree):
  - many_small: **36×** faster.
  - few_large: **8×** with a dense `Matrix` output, **48×** with a sparse output.
  - 8×spin-½ at m=0: about 2000×.
- **Risk:**
  - The column basis differs from the current QR output. That basis is arbitrary today, and the tests check `Q'Q≈I` and `QQ'≈P`.
  - Returning a sparse matrix would change the output type; keep `Matrix` to stay safe.
  - The antisymmetric sign tracking is untested.
  - The approach relies on `permute_state` never introducing phases.
- **Confidence:** high.

#### B02: Correctness bug: QR rank tolerance is absolute (0.1) in `symmetric_sector` (FHS)
- **Location:** `src/permutation_symmetry.jl:158-159` (`qr(P; tol=0.1)`) and `:144` (dense path, `findall(>(0.1)∘abs, diag(F.R))`).
- **Evidence:**
  - A column of P on an orbit of size k has norm 1/√k, which drops below 0.1 once k exceeds 100.
  - 5×spin-3: rank 441 against an expected 462, with ‖PQ−Q‖ = 4.56. `Q'Q ≈ I` still holds, so the result is silently wrong.
  - 7×spin-1 also fails. Reproduced in `f_perm/check5.jl` and `check5b.jl`.
- **Fix:** B01 removes the problem for `:symmetric` and `:antisymmetric`. For custom `(perms, weights)`, use a relative tolerance, or take the eigenvectors with eigenvalue near 1 block by block (see B21). Add a regression test.
- **Confidence:** high.

#### B03: The partial-trace map is stored as a very wide CSC matrix (FHS)
- **Affects:**
  - partial_trace/fermions/map: 131 ms and 257 MiB, of which 99% is in `sparse!`.
  - Every `PartialTraceMap` apply, and every `EmbedMap`, which applies `map'`.
- **Location:** `src/tensor_product.jl:708-737`: `sparse!(Is, Js, Vs, dim(Hsub)^2, dim(H)^2)`.
- **Why it is slow:**
  - CSC memory and build cost grow with the number of columns. Here that is dim(H)² = 16.7M columns for 32k nonzeros, and the colptr alone takes 134 MB.
  - Applying the map loops over all dim(H)² columns.
- **Fix:**
  - Store the transpose `At` (dim(H)² × dim(Hsub)²) and expose `map = transpose(At)`.
  - Build `At` with a counting-sort builder that allocates nothing of length m (`c_ptrace/exp_c1b.jl`), because `sparse!` allocates a length-m `csrrowptr`.
  - `EmbedMap` uses `conj(At)`.
  - `_apply_ptmap_on_vec!` loops over the `nzrange` of `At`. This also removes its function-wide `@inbounds` (`tensor_product.jl:651`).
- **Measured:**
  - Build: fermions **44×** faster (115.9 → 2.6 ms; 257 MiB → 2.8 MiB); full space 42×.
  - Dense apply: full space **80×**; the `precomputed_vs_eager.jl` case **41×** (64 → 1.5 µs).
  - Embed apply: 5.6–8.1× faster.
- **Risk:**
  - A sparse-vector input to `transpose(At)*x` takes a generic path that is 35× slower (24 → 853 µs). Mitigation: `PartialTraceMap(::AbstractSparseMatrix)` calls the eager FullAlg kernel, with `skipmissing` passed explicitly.
  - The type of `.map` changes to a `Transpose`.
- **Confidence:** high.

#### B04: `generate_states` recomputes the additive sum at every search node (FHS)
- **Affects:** generate_states/int and big_int, and constrained `hilbert_space` construction with `NumberConservation` weights or `AdditiveConstraint`.
- **Location:** `src/generate_constrained_states.jl:148-163`. The closure from `_additive_branch_constraint` calls `_additive_function_application`.
- **Evidence:** 85–86% of samples, spent rebuilding an O(depth) sum at each node for each of the 3 constraints.
- **Fix:** keep a per-depth prefix buffer inside the closure. `prefix[d+1] = prefix[d] + contribution(partial[d])`, using an `at_position` index from position to constraint entries. The sketch is in the report and `d_reshape_gen/s3_gen_exp.jl`. It works because `backtrack!` evaluates depth d only after depth d−1 has been accepted.
- **Measured:** **4.0×** (int) and **6.8×** (big_int), with identical output.
- **Risk:**
  - The closure becomes stateful and relies on DFS call order.
  - It stays fresh per call, but it is not thread-safe.
  - A cleaner design is an explicit branch-state protocol in `backtrack!`.
  - Add a test that combines a ProductConstraint with a constraint on a subset of spaces.
- **Confidence:** high.

#### B05: SubAlg partial trace enumerates every (f1, f2, fbar) triple (FHS)
- **Affects:** all SubAlg leaves, and especially constrained spaces: Sparse space SubAlg takes 6.4–8.4 ms. Also the map builders.
- **Location:** `src/tensor_product.jl:483-518`.
- **Evidence:**
  - `combine_states` is 46% and `state_index` 33%.
  - In the sparse space there are 661,696 iterations for 4,215 real terms; 96% of combined states are not in H.
- **Fix:**
  - Put the fbar loop outermost.
  - For each fbar, combine every subsystem state once and keep the hits `(K, J, w, ff, f)` in a typed buffer.
  - Then double-loop over the buffer.
  - The sketch is in `c_ptrace/exp_c2.jl`.
- **Measured:**
  - Sparse space: dense SubAlg **25×**, sparse SubAlg **18.6×**, SubAlg map build 5.7×.
  - Full space: 1.33× (dense) and 1.14× (sparse).
- **Risk:** the floating-point summation order changes; results stay `≈`.
- **Confidence:** high.

#### B06: Type-unstable product-space phase factor (FHS)
- **Affects:** partial_trace/product space (standard and map), embed/product space, product-space kron, and partial transpose.
- **Location:** `src/product_space.jl:261-268` (`partial_trace_phase_factor(…, ::ProductSpace)`); the same pattern in `partial_transpose_phase_factor` at `:269+`.
- **Evidence:** 63% of samples and about 15k allocations per call. JET reports runtime dispatch, because `zip` over heterogeneous tuples gives a `Union` loop variable.
- **Fix:** `prod(map(partial_trace_phase_factor, state1.states, state2.states, space.factors); init=1)`, and the same for `partial_transpose_phase_factor`.
- **Measured:** product standard **4.7×**, product map 3.9×; allocations drop from 15,493 to 717.
- **Risk:** none; results are identical.
- **Confidence:** high.

#### B07: `apply_local_operators(op::NCMul, …)` has a type-unstable amplitude (FHS)
- **Affects:** state-loop `matrix_representation` on non-fermion atoms (spins, bosons), which covers constrained sectors and product spaces, the typical Hamiltonian-in-a-sector case. The suite does not cover it; see the gap section §4.
- **Location:** `src/matrix_representation.jl:105-126`.
- **Evidence:**
  - The return type is `Union{Nothing, Tuple{SpinState, Union{Float64, Int64}}}`: `amp = op.coeff` is an `Int`, it is multiplied by a `Float64`, and the `if/elseif` has no `else`.
  - A spin chain with N=10 in the Sz=0 sector allocates 87,890 times per call.
  - With N=12, the Sz=0 sector (dim 924) is 4.3× slower than the full space (dim 4096).
- **Fix:** `T = promote_type(typeof(op.coeff), mat_eltype(op)); amp = convert(T, op.coeff)`, an explicit `else` branch, and early returns of `zero(T)`.
- **Measured:** spin Sz=0 N=10 **2.7×** faster (2,988 → 1,097 µs); allocations 87,890 → 1,706.
- **Risk:** low.
- **Confidence:** high.

#### B08: `_apply_local_operators_fast` folds over a zip of heterogeneous tuples (FHS)
- **Affects:**
  - matrix_representation/product space: 44% of samples.
  - Any constrained or sector product-space representation, eager or lazy.
  - The native lazy sparse path (B15).
- **Location:** `src/product_space.jl:346-367`. The related `_apply_local_operators_slow` at `:368` has the same pattern.
- **Evidence:** JET reports runtime dispatch at `:362`; there are 5 boxed allocations per state per term.
- **Fix:** recurse with `first`/`Base.tail` over the `ops`, `spaces` and `precomps` tuples. The sketch is in reports (b) and (e). The terminal method must carry the same type parameters, or the call is ambiguous.
- **Measured:**
  - product space: **1.7×** (818 → 474 µs); allocations 17,598 → 3,774.
  - The native dense apply: 3.75×, with 0 allocations.
  - Combined with B10 on ps_many_cons: 1.4×.
- **Risk:** low.
- **Confidence:** high.

#### B09: Symbolic algebra (NCP): redundant re-sort, double hashing, Dict growth, O(n²) `sum`
- **Affects:** symbolic/* and symbolic_states. This bottleneck is in NCP, not FHS.
- **Combined measured effect** of A1+A2+A3+A4a below plus B14:

  | benchmark | change |
  |---|---|
  | sum | −25..−28% |
  | sum_square | −45..−52% |
  | deep_product | −41..−42% |
  | projected_op_on_state | −8..−16% (allocations −44%) |

- **A1, redundant re-sort in `^`:**
  - Location: NCP `src/muladd.jl:58-64`, line 62: `autosort() && return bubble_sort(ret)`.
  - Why: for an exponent ≥ 2 every `*` inside `power_by_squaring` has already sorted the result.
  - Fix: `autosort() && b < 2 && return bubble_sort(ret)`.
  - Measured: sum_square −29..−34%. Risk: low. Confidence: high.
- **A2, every new Dict key is hashed twice:**
  - Location: NCP `src/add.jl:176-182`, where `add_term!!` calls BangBang `modify!!`. That uses the deprecated `ht_keyindex2!`, so `_setindex!` hashes the key a second time; it also compares the key with itself.
  - Fix: use `Base.ht_keyindex2_shorthash!` plus `Base._setindex!(dict, v, k, -idx, sh)`, keeping the current method as the fallback for mismatched types.
  - Measured: −6..−31% across the four benchmarks. Process-to-process variation is high; the conservative run gave −9/−6/−15/−2%.
  - Risk: medium, because it depends on Base Dict internals. Confidence: high on the mechanism, medium on the size.
- **A3, `_sum_sorted_terms` Dict grows from size 1:**
  - Location: NCP `src/sorting.jl:84-93`.
  - Evidence: `rehash!` is 17% of deep_product.
  - Fix: `sizehint!(d, length(sorted_terms))`.
  - Measured: deep_product −20..−25%. Risk: low. Confidence: high.
- **A4a, the `+(::NCAdd,::NCAdd)` constructor re-filters:**
  - Location: NCP `src/add.jl:154-161`, which calls `filter_ncadd_dict!` (`add.jl:1-15`).
  - Fix: `NCAdd(coeff, dict; filter_zeros=false, filter_scalars=false)`.
  - Measured: sum −13..−18%. Risk: low.
- **A4b, `sum` copies the accumulator, which is inherently O(n²):** a mutating fix is unsafe. An `add!!` loop is 35% faster; document it.
- **A7, runtime dispatch in sorting mixed-type products:**
  - Location: NCP `src/sorting.jl:124,126`.
  - Evidence: 27% self time plus 25% of allocations in projected_op_on_state.
  - This is a deliberate design (inference barriers). **Adding a `::Tuple{Vector,Bool,Int}` assertion was tested and made it 2.5× slower. Do not do this.**
  - A possible fix is an `if effect isa …` split; untested.
- **A8, copying factors for each new term:** small gain (about 3–5%); untested.
- **A9, `hash(::FermionSym)` uses 4 hash rounds:**
  - Location: FHS `src/symbolic_fermions.jl:85`.
  - Estimated gain is 5–10% on deep_product. It could not be measured in-process, because redefining `hash` breaks existing dicts.
- **A10, `with(_autosort=>false)` on every sort:** ≤5%.
- **Note:** fixes in NCP need a separate release of that package (or a local dev). Phase 2 must decide whether to change NCP.

### Tier 2: solid, medium-sized wins

#### B10: `partition_product` loops over every group with dynamic dispatch (FHS)
- **Affects:** all product-space representations: ps 40%, ps_many 13%, ps_many_cons 66%.
- **Location:** `src/matrix_representation.jl:153-170`.
- **Evidence:** JET reports runtime dispatch, and the `bases` tuple is re-boxed on every iteration.
- **Fix:** iterate over the 1–3 factors instead of the n groups, using a recursive tuple search or a precomputed Dict from basis to index. The sketch is in report (b).
- **Measured:** ps_many_cons 1.2× (1,144 → 958 µs); allocations 12,193 → 4,929.
- **Confidence:** high.

#### B11: FullAlg partial trace looks up `state_index(f1sub, Hsub)` in a dict for every (J1, J2) pair (FHS)
- **Location:** `src/tensor_product.jl:520-551`. The lookup at `:530` is 24–49% of samples; `splitsamps` (`:522`) allocates on every call.
- **Fix:** precompute `(K, fsub, fbar)` once per basis state.
- **Measured:**
  - fermions/standard **3.0×**.
  - Dense FullAlg: sparse space 3.1×, full space 1.9×.
  - Very sparse inputs regress slightly (0.9×). Compute K lazily, or skip the precompute when nnz < dim(H).
- **Confidence:** high.

#### B12: Per-term kron chain in unconstrained `ProductSpace` (FHS)
- **Affects:** product space many spaces: 37% of samples, plus 6% for a zero-identity add.
- **Location:** `src/matrix_representation.jl:172-185` (`foldl(kron, Iterators.reverse(::Vector{Any}))`); `kron(sparse, Diagonal{Ones})` densifies the identity.
- **Fix:**
  - Build `I_l ⊗ A ⊗ I_r` directly in CSC (`b_matrep/exp_kron.jl`).
  - Skip `+ op.coeff*Eye` when `iszero(op.coeff)`, at `:235`, mirroring `:295`.
- **Measured:** per-term kron 44 → 15 µs, an estimated −30..−40% for ps_many. The coefficient check alone saves 3–6%.
- **Risk:** the kron ordering must match the current `Iterators.reverse` ordering.
- **Confidence:** medium-high.
- **Also:** the `map(enumerate(spaces)) do` block at `:176-180` boxes the captured `op`, because it reassigns `op`. Rename the inner variable to `local_op` (part of B16).

#### B13: `reshape` builds transition tables in `Vector{Any}` (FHS)
- **Affects:** all reshape benchmarks. In mat_to_vec there are 4,096 boxed tuples (one per element), and that path is 62% of samples.
- **Location:** `src/reshape.jl:240-284` (`_transitions`) and the kernel at `:296-322`.
- **Fixes:**
  - **D3:** a concretely typed collection with a size hint; convert `UInt64` indices to `Int`.
  - **D4:** a linear-offset kernel instead of splatted Cartesian tuples, with no `@inbounds`. Keep a fallback for non-`Array` inputs.
  - **D5:** `_fock_mapper` (`src/physics/fermions.jl:96`) builds a one-mode `FermionicSpace`, including an OrderedDict, for every mode just to read its id. Use `get(H.mode_ordering, id, 0)` and `map` over the tuple instead.
- **Measured** (D3+D4+D5): mat_to_tens 1.4–1.9×, mat_to_vec 1.46×, mixed 1.76×; mat_to_vec allocations 4,346 → 69.
- **Risk:** low-medium. D4 assumes linear indexing.
- **Confidence:** high.

#### B14: FHS symbolic-state application allocates (FHS)
- **Affects:** symbolic_states/projected_op_on_state.
- **A6:** `mul_effect(op, ::SymbolicState)` (`src/symbolic_states.jl:199-219`) always returns `amp * state`. Return `0` when `iszero(amp)` and `newstate` when `isone(amp)`.
- **A5:** `_apply_symbolic_operator_to_state` (`:148-155`) allocates `NCMul(1,[op])` and a precomputation vector. Use tuples, after checking that other `apply_local_operators` overloads accept them.
- **Measured** (A5+A6): −8..−16% time, −44% allocations.
- **Confidence:** high (A6), medium (A5).

#### B15: Lazy operator: term structure and per-call precomputation (FHS)
- **Affects:** matrix_application/lazy liouvillian, and every `:lazy` representation.
- **E1:** the open-system lazy operator is 45 `TensorProductOperator`s. The 32 one-sided terms each pay a fill/copy pass.
  - Group terms by `partition_product(...).inds` and sum the one-sided groups on their factor space (`src/lazy_matrix_representation.jl:250-260`).
  - Measured with E2, using a hand-built equivalent: mul! **0.825×**; allocations 183 → 79.
- **E2:** `lazy_mul!(::LazyOperator{<:NCAdd})` (`:217`) recomputes the per-term precomputation on every call.
  - Store the precomputed data in the constructor.
  - Measured: single-space mul! 0.73×, 32 → 0 allocations.
  - Check the OhMyThreads chunked path.
- **E3:** `L * SparseVector` goes through SciMLOperators' generic kron path and is 250–340× slower than sparse.
  - An FHS-native sparse path measured **88×** faster.
  - This is a design change. It also touches `Base.:*(L::LazyOperator, x)` (`:229`), where `similar` produces sparse output that is then filled by scalar inserts.
- **E5:** the single-space vector kernel multiplies by `coeff` before checking whether the term acts (`:128`). Reordering measured **0.60×**.
- **E6:** construction:
  - The boxed captured `op` (B12 note).
  - The pairwise `sum(terms)` that grows an `AddedOperator` 44 times (`:253`). Build it once with `AddedOperator(Tuple(mats))`.
  - Measured: construction 0.55×.
- **Confidence:** high, except E1 (hand-built measurement) and E3 (API fit).

### Tier 3: smaller or narrower wins

- **B16 (FHS): `ProductSpaceMapper` is rebuilt on every call.**
  - Location: `src/product_space.jl:504-577`. It uses `Any`-keyed dicts and `Any[]` vectors, and JET reports about 30 runtime dispatches.
  - It costs 47 µs per build: 26–65% of product-space partial_trace, embed and complement, and 77% of product partial trace once B06 is applied.
  - Fix: concrete containers built with `ntuple`/`map`, or let callers pass in a mapper.
  - The non-concrete mapper type is also the root cause of the permutation dispatch (B19).
- **B17 (FHS): sparse `generalized_kron` copies its result.**
  - Location: `src/tensor_product.jl:182` (`mout .= sparse!(...)` into a freshly allocated empty sparse matrix).
  - Fix: return the `sparse!` result.
  - Measured: embed/fermions 1.18×, embed/product 1.56× (that figure includes B06).
- **B18 (FHS): `hilbert_space` sector construction.**
  - Location: `src/sectors.jl:43-65`. `OrderedDict(zip(states, 1:n))` has no size hint (65%), and `groupby` uses `OrderedDict{Any,Vector}` (27%).
  - Fix: a sized `OrderedDict{B,Int}`, and a concrete key type via `promote_op`.
  - Measured: about −15% of hilbert_space.
- **B19 (FHS): permutation loop dispatch and `Tuple(perm)`.**
  - Location: `src/permutation_symmetry.jl:28,32,88`.
  - Matters only if B01 is not adopted, or for custom perms.
  - Fix: one function barrier with fused loops and `ntuple(i->perm[i], Val(N))`.
  - Measured: many_small 2.1×.
- **B20 (FHS): `SpinState` hash goes through HalfIntegers' `hashhalf`.**
  - Location: `src/physics/spin.jl:86`.
  - Fix: `hash(twice(x.m), hash(SpinState, h))`.
  - Measured: many_small 1.12×, 8×spin-½ 1.36×. This affects every spin `state_index`.
- **B21 (FHS): custom `(perms, weights)` path uses dense `Q` extraction.**
  - Location: `src/permutation_symmetry.jl:157-160`.
  - Fix: block-wise `eigen` over the connected components (this also fixes B02 for custom weights), or `lmul!` into a preallocated matrix.
  - Not measured. Also: `P *= c` allocates; use `rmul!`. `sparse(finalize!(…))` copies needlessly. `sizehint!` reserves 3·d·n! entries.
- **B22 (FHS): BdG `normal_order_to_bdg` uses sparse slicing and `hvcat`.**
  - Location: `src/physics/bdg.jl:16-24`.
  - Fix: emit the transformed COO entries directly.
  - Measured: bdg −11%; allocations 207 → 22.
- **B23 (FHS): `remove_identity` re-filters, and aliases, the user's dict.**
  - Location: `src/matrix_representation.jl:597`.
  - Fix: pass `filter_zeros=false, filter_scalars=false`, or drop the step.
  - Measured: free_fermion −5%. The aliasing is a latent correctness concern.
- **B24 (FHS): free_fermion and BdG validity checks make extra passes.**
  - Location: `src/physics/fixednumberfock.jl:290-300` and `src/physics/bdg.jl:55-70`. `isquadratic` plus `isnumberconserving` is 17% of free_fermion; `mode_position` hashing is another 17%.
  - Fix: fuse the checks into one pass. Untested.
- **B25 (FHS): `generate_states` setup is O(n²).**
  - Location: `src/generate_constrained_states.jl:112,121,125-130`: `findfirst(==(collect(...)))` per subspace, and a `hilbert_space` built per mode.
  - About 12% of the time once B04 is applied. Fix: build one id → position Dict.
- **B26 (FHS): sparse input to FullAlg goes through `findnz` and then `m[J1,J2]`.**
  - Location: `src/tensor_product.jl:120,566`.
  - Fix: iterate `nzrange` and pass `nonzeros(m)[p]` directly. Estimated 10–25%.
- **B27 (FHS): complement bookkeeping.**
  - Location: `src/spaces.jl:499-507`. `_find_compatible_complementary_states` splits every state of H (65% of complement/fermions).
  - No obvious general fix.
- **B28 (FHS): `default_partial_trace_alg` picks badly.**
  - Location: `src/tensor_product.jl:469-472`. It picks SubAlg for sparse input on the full space, where FullAlg is 1.6–2× faster.
  - Once B05 is in, SubAlg wins for every dense input measured.
  - Proposed rule:
    - Dense input: Sub when `dim(Hsub)*dim(Hcomp) + dim(H)^2 ÷ dim(Hcomp) < length(m)`.
    - Sparse input: Sub only when the term estimate times a lookup factor of 4–8 is below `nnz(m)`.

### Latent correctness issues found while profiling
- **B02:** the QR tolerance in `symmetric_sector` (above).
- **`catenate_fock_states` ignores its `T`** (`src/physics/fock.jl:131`). `num |= FockNumber{UInt64}` promotes the value, so the function returns a `Union`. If `T` were honoured, mode 64 would overflow `Int`. Fix: `num |= convert(T, state) << shift`.
- **`remove_identity` aliases** the caller's dict (B23).
- **`_operator_type(::NCAdd)`** (`src/symbolic_states.jl:249-272`) boxes captured booleans. This is a type instability, not a bug.

### Type instabilities that are not hot (once per call)
These are recorded only, with no action proposed:
- `hilbert_space(…)` returns a Union of `FockNumber{UInt64}` and `{BigInt}` spaces.
- `matrix_representation` makes a kwcall dispatch.
- `partition_product` returns a non-inferable `ProductOperator`.
- `allocate_tensor_product_result` uses try/catch and returns a Union.
- `symmetric_sector` infers as `Any`.
- `_fock_mapper` constructs `FockMapper(::Tuple{Vararg})`.
- `_validate_from_spaces` compares `dim::Union{Int,BigInt}`.
- `generate_states` merges `kwargs`, so `process_result` is inferred as `::Any`.
- The lazy inner `FunctionOperator`s use Float64 prototypes.

## 3. Benchmark hygiene

### Reliability
- **H01 (P1): the noise is environmental, not GC.**
  - Sample times are bimodal, at 1.0–1.1× and 1.7–2.0× the minimum. A loop with zero allocations shows the same spread.
  - The likely cause is frequency and P/E-core effects on the 35 W part, plus contention.
  - As a result, medians cannot reliably detect regressions smaller than about 30%. Compare **minimum** times, plus allocations and memory, which are deterministic.
  - Locally: use the High-performance power plan, stop background processes, and pin Julia to P-cores (`cmd /c start /affinity FFFF /b julia …`).
- **H02: CV above 2 comes from rare full-GC pauses** (under 1% of samples). Ignore mean and standard deviation, and don't enable `gcsample`.
- **H03 (P2): partial_trace/fermions/map gets only 8 samples** and is GC-dominated. B03 fixes the root cause (131 ms → about 3 ms). Otherwise set `seconds=6`.
- **H11/H12: inputs are not mutated across evals,** and `$` interpolation is correct everywhere. No `setup=` or `evals=1` changes are needed.
- **H14 (P2): `tune!` takes 74 s for nothing,** since every leaf tunes to `evals=1`. Add this at the end of the suite file:
  ```julia
  for (_, b) in BenchmarkTools.leaves(SUITE); b.params.evals = 1; b.params.evals_set = true; end
  ```
- **H08 (P3): `DEFAULT_PARAMETERS.seconds = 1` is a global mutation.** Prefer setting it per benchmark.
- **H07 (P3): every input comes from one global RNG,** so adding a `rand` call earlier shifts all later inputs. Use one `Xoshiro(seed)` per section.

### Benchmarks that measure the wrong thing
- **H04 (P1): `partial_trace_algorithms` keys contain the heuristic's result (`default=$def`).**
  - If the heuristic changes, every key moves and history is lost.
  - The path users actually get (no `alg` keyword) is never benchmarked.
  - Fix: use stable keys and add a `"default"` leaf (snippet in report (h), H04).
- **H05 (P2): the same leaves recompute the complement on every call,** which is up to 59% of a leaf. Pass `complement=`.
- **H06 (P2): `generate_states` leaves use internals,** and "big_int" is misnamed. Its `CombineFockNumbersProcessor{FockNumber{Int}}` overrides the default processor, so it never does BigInt arithmetic. Replace them with public `hilbert_space(f, 1:N, cons)` leaves for N=60 and N=64.
- **H09 (P2): applying a precomputed map is never benchmarked.** The product-space "map" and "standard" leaves both measure about 320 µs of setup, while applying a built map takes 4 µs.
- **H10 (P3): "mul sparse" applies a vector with about 4 nonzeros through the uncached operator,** so it measures overhead. Rename it or document it.

### Redundancies
- **R1:** `reshape/mat_to_tens` and `mixed_split_combine` are identical in allocations, memory and minimum time. Make "mixed" actually differ, or drop it.
- **R2:** `partial_trace/product space/map` and `standard` both measure setup. Replace "standard" with an apply-precomputed leaf.
- **R3:** `partial_trace/fermions/standard` duplicates `pt_alg/…/Sparse space/Sparse/FullAlg`.
- **R4:** the complement cost is inside every `pt_alg` leaf (H05).

### Gaps (ranked; proposed snippets are in hygiene report §4)
- **G1: time to first use.** `using` takes 1.4 s; a small first-use workload takes 9.5–11 s.
- **G2: precomputed partial trace and embed apply,** in place, and the pure-state partial trace.
- **G3: spin chains.** Heisenberg `matrix_representation` takes 59 ms at Ns=14 Sz=0 (dim 3432), against 7 ms for Hubbard at dim 14520 (see §4 and B07).
- **G4: Hubbard sector,** the documentation's headline example. Sector construction takes 85 ms and `constrain_space` 112 ms (see §4).
- **G5:** Bose–Hubbard.
- **G6:** lazy operator on a closed system. Lazy `mul!` is 50× slower than sparse `mul!`.
- **G7:** partial transpose.
- **G8:** `tensor_product` of matrices and `generalized_kron`, which take 20–40 ms at 5+5 modes.
- **G9:** eager Liouvillian.
- **G10:** symbolic states, `sectors`, and block-diagonal representations.
- **G11:** `state_index` lookup.
- **G12:** Majoranas.

### Infrastructure
- **I1: `benchmarks.jl` needs `PACKAGE_VERSION`** (defined by AirspeedVelocity). Default it to `pkgversion(FermionicHilbertSpaces)` when it is undefined.
- **I2: NonCommutativeProducts is a direct dependency in `benchmark/Project.toml`,** yet no benchmark script uses it. That pinned entry is what went stale. Remove it, add `[compat]`, and document the local setup. OhMyThreads is not in the env, so the threaded extension is never measured.
- **I3: suite setup takes 58 s,** of which 39% is the lazy Liouvillian section. `cache_operator` takes 25.7 s and `representation(…, :lazy)` 10.6 s, mostly SciMLOperators compilation. This is a TTFX finding.
- **I6: `precomputed_vs_eager.jl` is outside SUITE,** and its header comments are stale: they cite fixed timings, and item 5 about pure states is outdated. Fold it into SUITE.

## 4. Gap paths profiled separately (spin chain, Hubbard sector)

None of these four workloads is in the suite.

- **Correctness check:** with G-1 through G-4 loaded by `@eval`, the full package test suite passed (3151/3151). Every output was identical to the baseline.
- **Files:** `scratchpad\phase1\g_gaps\` holds `patches.jl`, `x_final.jl`, `x_g4*.jl`, the profiles and the JET reports.

| Path (same process) | Baseline | All fixes | Ratio | Main cause |
|---|---|---|---|---|
| Spin Heisenberg `matrix_representation`, Ns=14 Sz=0 (dim 3432) | 50–65 ms | 26–39 ms | 0.43–0.60× | `SpinState` hash (39%); per-term sparse matrices and tree sum (about 25%) |
| Spin `hilbert_space`, Ns=14, `AdditiveConstraint(0)` | 7.3 ms | 3.8 ms | 0.52× (0.34× in some runs) | runtime-length `Tuple(::Vector)` (35%); `SpinState` hash (about 20%) |
| Hubbard sector `tensor_product`, Nh=12 (3,2) | 80 ms | 4.9 ms | **0.06×** | dynamic dispatch in the `ProductConstraint` sector function (87%) |
| `constrain_space` with no double occupancy, Nh=12 | 101 ms | 4.4 ms | **0.04×** | same cause (95%) |

- **G-1 (= B20; stream s1): `SpinState` hash goes through HalfIntegers' `hashhalf`.**
  - Location: `src/physics/spin.jl:86`.
  - Fix: `hash(twice(x.m), h)`.
  - Measured: spin `matrix_representation` 0.52× on its own. Risk: very low.
- **G-2 (stream s3): one sparse matrix per term, then a tree sum, on constrained spaces.**
  - Location: `src/matrix_representation.jl:233-236`. This applies to a constrained or sector space over a `ProductSpace`.
  - Fix: use a single accumulator and call `operator_indices_and_amplitudes!` for each term. Keep the kron path for an unconstrained `ProductSpace` only.
  - Measured: 0.83× on its own, 0.43× together with G-1.
  - Caveat: `add_identity!!` returns `append!`'s vector, so its result must not be reassigned.
- **G-3 (stream s2): `combine_states(::Vector, ::ProductSpaceMapper)` builds a `Tuple` of runtime length.**
  - Location: `src/product_space.jl:602`.
  - Fix: `ntuple(i->substates[i], Val(N))`, with N taken from the mapper's type.
  - Measured: spin `hilbert_space` 0.47×.
- **G-4 (stream s5): the `ProductConstraint` sector function maps over a `Vector` of closures with an abstract element type.**
  - Location: `src/constraints.jl:209-216`.
  - Cause: each closure captures a `FockMapper{N,…}` whose `N` (the maximum position) and `P2` (`BitPermutation` or `nothing`) differ from site to site.
  - Fixes, all measured:

    | Variant | Hubbard | constrain_space | Nh=30 | Notes |
    |---|---|---|---|---|
    | (a) tuple with recursive typed fill | 0.039× | 0.028× | — | compile cost explodes with many constraints (first call 1.08 s vs 0.21 s at Nh=30; slower than baseline at Nh=40) |
    | (b) typed loop over the vector, `V` from `promote_op` | 0.069× | 0.19× | 0.34× | scales |
    | (c) identical per-site `FockMapper` types | — | — | — | touches `FockMapper` semantics; keep `isfullpartition` computed from the maximum position |
    | (b)+(c) | — | 0.03× | 0.07× | first call 0.43 s |

  - Recommendation: (b), plus (c) if it can be done safely.
- **G-5 (low priority): conversion overhead and state lookup.**
  - The fast path converts all N site states to `Int` and back for every term (`product_space.jl:333-335`).
  - Sector `state_index` hashes the whole N-tuple; a mixed-radix linear index could replace it.
  - The accumulator's `sizehint!` underestimates for spin chains.
  - Estimated 10–25%; not measured.
- **G-6 (minor): `tensor_product` re-evaluates the combined sector function on every product state** (`tensor_product.jl:31-33`), even though the input labels are known.
- **Recommended benchmark sizes:**

  | Benchmark | Size |
  |---|---|
  | spin `matrix_representation` | Ns=12 |
  | spin `hilbert_space` | Ns=14 |
  | Hubbard `tensor_product` | Nh=10 (3,2) |
  | `constrain_space` | Nh=10 |
  | many-constraint scaling case | Nh=30 (2,1) |

## 5. Open questions
1. NCP fixes (B09): change the local NCP repository and release it, or leave them as recommendations?
2. B01 output type: keep a dense `Matrix` (8× faster) or return a sparse matrix (48×)?
3. B03 changes the type of the `PartialTraceMap.map` field. Is that field public?
4. B15 E3 (native sparse lazy path) and B16 (passing a mapper in) are API-level design changes.
5. The 64-mode cliff: `default_fock_representation(N) = N < 64 ? UInt64 : BigInt`. It is probably deliberate, since `state_index` computes `f + 1`.
