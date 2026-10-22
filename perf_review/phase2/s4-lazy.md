# Phase 2, stream s4-lazy: lazy operators (B15)

Branch `perf/s4-lazy`, based on `quality-control` 1cee0550. Files changed: `src/lazy_matrix_representation.jl` and `ext/OhMyThreadsExt.jl`. s3's files (`src/matrix_representation.jl`, `src/product_space.jl`) are untouched. All five commits cherry-pick cleanly onto `perf/integration` (checked in a scratch worktree, which was then removed).

## Commits and per-commit gain vs parent

Each commit was measured against its parent: min of 3 alternating rounds, baseline env `baseline-qc-env`. Alloc counts are exact. The noise floor is about ±14% per leaf, so smaller time changes are reported only when allocations corroborate them.

| commit | relevant benchmark | time ratio vs parent | allocs parent → commit |
|---|---|---|---|
| d32c7fa5 Store per-term precomputation in lazy NCAdd operators (E2) | small single space (dim 64), lazy `mul!` | 0.85 | 42 → 0 |
| | 12-site chain (dim 4096), lazy `mul!` | 0.97 | 90 → 0 |
| | Hubbard `hilbert_space(f,1:14,NumberConservation(7))`, lazy `mul!` | 1.05 (noise) | 106 → 0 |
| | single-space `L * sparsevec`, 0.1% density | 0.58 | 147 → 79 |
| | Hubbard, lazy construction | 1.07 (noise) | 899 → 1011 |
| e33ee90d Build the lazy multi-space sum as one AddedOperator (E6b) | Liouvillian construction (`representation(…, :lazy)`) | 0.63 | 6293 → 4876 |
| 0f36b543 Overwrite the output for zero beta in lazy NCAdd mul! (bug fix) | all | 0.93–1.07 (noise) | unchanged |
| 37f5fb6d Group single-group terms of lazy product-space operators (E1) | `matrix_application/lazy liouvillian/mul!` | 0.87 | 184 → 79 |
| | `matrix_application/lazy liouvillian/mul sparse` | 0.50 | 8914 → 3125 |
| | Liouvillian `cache_operator` | 0.28 | 744 → 284 |
| | Liouvillian construction | 0.55 | 4876 → 3194 |
| | fermion⊗boson⊗spin (dim 160) construction | 0.72 | 2418 → 1906 |
| | fermion⊗boson⊗spin lazy `mul!` | **1.15** (two runs: 1.08, 1.15) | 84 → 128 |
| 1bde97d8 Collect lazy sparse-vector products as index/value pairs | single-space `L * sparsevec`, 5% density | 0.30 | 115 → 41 |
| | single-space `L * sparsevec`, 0.1% density | 0.70 | 79 → 20 |
| | Liouvillian `mul sparse` | 0.96 (noise) | unchanged |

**Totals, HEAD vs 1cee0550:**

| benchmark | ratio |
|---|---|
| liouvillian `mul!` | 0.92 |
| liouvillian `mul sparse` | 0.47 |
| construction | 0.34 |
| `cache_operator` | 0.26 |
| small single space | 0.86 |
| sparse-vector apply | 0.29–0.40 |
| Hubbard and chain dense `mul!` | unchanged (zero allocations) |
| mixed product `mul!` | 1.18 (small regression, see E1) |

Outputs are `==` to the baseline for every commit except E1 and later, where the floating-point summation order changes. On the Liouvillian and the mixed case the relative error is ≤ 3.3e-16.

Scripts:
- `scratchpad/phase2/s4-lazy/bench.jl`: cases `liouv` (the suite setup), `closed`, `chain`, `small`, `mixed`, `sv`.
- `compare.jl`: comparison of the runs.

## TTFX (I3), fresh process, suite Liouvillian, seconds

Three runs each; ranges shown.

| | `representation(…, :lazy)` | `cache_operator` | first `mul!` | first `L*vs` | total |
|---|---|---|---|---|---|
| base 1cee0550 | 7.0–7.7 | 13.2–15.3 | 1.5–1.6 | 1.9–2.4 | ≈ 25 |
| + E6b | 2.9–3.7 | 12.9–15.4 | 1.5–1.7 | 1.7–1.9 | ≈ 20 |
| + E1 | 4.1–5.3 | 1.2–1.6 | 2.4–3.1 | 2.3–2.7 | ≈ 11 |
| HEAD | 4.3–4.5 | 1.2–1.4 | 2.3–3.9 | 2.4–2.8 | ≈ 11 |

Effects:
- **E6b** halves the compile time of `representation`.
- **E1** shrinks the `AddedOperator` tuple from 45 to 14 operators, which takes about 13 s off `cache_operator`.
- **E1** costs about 1 s more on the first `mul!`, from compiling the NCAdd kernels.
- **Overall**, the TTFX of this setup drops by about 2.3×.

## Per-item analysis and decisions

### E2: store the per-term precomputation (implemented)
- **What it does:** the constructor builds a vector of `(term, coeff, precomp)` tuples, using `lazy_precomputation`. `lazy_mul!` and both OhMyThreads chunked paths iterate over it.
- **Transposed operators:** `adjoint`, `transpose` and `conj` reuse the vector on `TransposedSpace`. That is valid because `_apply_local_operators(…, ::TransposedSpace, …)` applies on the parent space with the same precomp, which the NCMul path already relied on.
- **Chunked paths:** they previously used `NCterms` (coefficient inside the term) with `α`. They now pass `key, coeff*α`, which is the same as the NoChunking path.
- **API:** the `precomp` field and the `P` type parameter of NCAdd lazies change from `Nothing` to a Vector. These are internal.
- **Gain:** every lazy NCAdd `mul!` now allocates nothing. The time gain is visible only on small spaces and on sparse inputs.

### E5: zero check before the coefficient multiply (rejected)
- Correct, but it gave no measurable gain on the qc baseline: chain, Hubbard and small were all within ±5%.
- Phase 1's 0.60× did not reproduce.
- The commit was dropped from the branch.

### E6b: build the AddedOperator once (implemented)
- `AddedOperator(Tuple(mats))` flattens nested operators and checks sizes exactly as the pairwise `+` did.
- Plain-matrix terms are wrapped in `MatrixOperator`, which matches what `+` does.

### Zero-β bug (found by the reviewer; fixed in its own commit)
- **The bug:** the NCAdd `lazy_mul!` methods (base, TermChunking, StateChunking) used `rmul!(y, β)`, so `β = 0.0` kept NaN/Inf already in `y`.
- **Why it matters now:** E1 routes product-space one-group terms through NCAdd lazies, so this would have become a regression on product spaces.
- **Fix:** the same `iszero(β) ? fill! : rmul!` guard as the NCMul methods.
- **Tests:** single-space tests and chunked tests that start from a NaN output.

### E1: group one-group terms on unconstrained `ProductSpace` (implemented)
- **Correctness:**
  - The sum of `I⊗…⊗A_t⊗…⊗I` over terms equals `I⊗…⊗(ΣA_t)⊗…⊗I`, with the same reversed kron order as the per-term path.
  - The scalar coefficient goes into the first group operator, or stays as an identity term if there is none.
  - Multi-group terms are unchanged.
  - Constrained and sector spaces over products keep the generic per-term path.
  - TermChunking becomes NoChunking here, so the existing "ignoring" warning stays true.
- **Element types:**
  - Group factor vectors use the narrowest common factor type of that group.
  - The lazy element type is `mat_eltype(op)` of the whole operator. The reviewer found an `InexactError` when keys narrowed to different types; that is fixed and covered by a test.
- **Tests:** compared with the explicit sparse matrix on these operators:
  - the Liouvillian, plus a scalar;
  - left-only, right-only and jump-only parts;
  - fermion⊗spin and fermion⊗boson⊗spin;
  - boson-only terms on 3 groups;
  - odd fermion terms;
  - two fermion groups;
  - mixed label types with complex coefficients.

  Each comparison covers `*`, `'`, `transpose`, matrix input, 5-arg `mul!` and `concretize`.
- **API:** the return type is unchanged (`AddedOperator`), but `.ops` is shorter and structured differently.
- **Known cost:** on a small 3-group mixed operator (dim 160), dense `mul!` is about 10–15% slower, with 44 more allocations, from the deeper SciMLOperators kron nesting. This was accepted for the large Liouvillian and TTFX gains.

### `L * SparseVector` on single spaces (implemented)
- **Old behaviour:** `similar(x, T)` produced a sparse output that was then filled by one insertion per entry, each O(nnz).
- **New behaviour:** the method collects `(index, value)` pairs from every term and the scalar part, skips stored zeros of `x`, and calls `sparsevec`, which sums equal indices.
- **Cost:** no O(dim) buffer, so tiny inputs on large spaces stay cheap.
- **Behaviour changes:**
  - Without `projection`, an out-of-space image throws qc's `ArgumentError` instead of a `BoundsError` (tested).
  - The output no longer carries the input's stored zeros.
- **Limits:**
  - StateChunking is not used on this path, which is serial.
  - Sparse matrix inputs keep the old path. A COO version would work the same way.

### E3: native sparse path for product-space `L * SparseVector` (deferred: design proposal)
**The problem:**
- On product spaces, `representation(…, :lazy)` returns a SciMLOperators `AddedOperator` of `TensorProductOperator`s.
- `L * vs` goes through SciMLOperators' generic dense-reshape path.
- An FHS-specific method would be type piracy.

**Proposal:**
- Return a `FunctionOperator` that wraps an FHS type holding two things:
  - the kron-structured `AddedOperator`, used for dense inputs, where it is about 15× faster than a native state loop (Phase 1 exp3);
  - a vector of `(ProductOperator, coeff, precomp)` over the full product space, in the layout `lazy_precomputation` already uses.
- Sparse vectors would use a COO kernel like `_push_single_term!` over those terms. Phase 1 measured this at about 88× faster.
- The new type needs `adjoint`/`transpose`, which rebuild the ProductOperators on transposed factor spaces as `adjoint(::LazyOperator{<:ProductOperator})` does, and `concretize`.

**Why it is deferred:** this changes the public return type on product spaces, so it needs a maintainer decision. E1 already halves the current sparse path, 0.47×.

### Other notes
- `get_input(::LazyRepr{<:AbstractArray}, H)` is called per factor space, so a user-supplied full-size input throws on product spaces. This is pre-existing and unchanged.
- The inner `FunctionOperator` prototypes are `Float64` (`mat_eltype(space)` fallback). Not investigated.
- **Pre-existing bug found by the reviewer, not fixed (outside this stream):** on `tensor_product(majorana space, spin)`, `s[:x]*m[1]*m[3]` gives the correct `L*v` but wrong `L'*v` and `transpose(L)*v`, both on the parent and here. This is a Majorana × spin ProductOperator transpose issue.

## Tests
- Targeted lazy, explicit-import and open-system `@testitem`s pass at each commit.
- Full suite on 1bde97d8 (worktree code confirmed via `pathof`): 3658/3658 passed, including the Aqua and ExplicitImports items.

## Reviewer findings and resolution
One general-purpose reviewer looked at the diff.

| finding | resolution |
|---|---|
| (1) Wrong element type when group keys narrow to different types | Fixed in the E1 commit, with a test |
| (2) β = 0.0 NaN leak in NCAdd `mul!` | Fixed in its own commit, with tests |
| Dispatch-less argument annotations | Removed |
| History wording in comments and commit messages | Rewritten |
| Sparse path keeps stored zeros of `x` | Fixed: they are now skipped |
| Error-type change in the sparse path not mentioned in its commit message | Fixed: the message now says it |

Noted and left as is:
- Multi-group terms take their coefficient as `NCterms` does; one-group terms keep the key prefactor. This is equivalent because keys have prefactor 1.
- The generic method passes TermChunking through, which is pre-existing behaviour.
- Sparse `*` ignores StateChunking.
- The Majorana × spin transpose bug above, which is pre-existing.
