# Phase 2, stream s1-perm: permutation symmetry

Branch `perf/s1-perm`, rebased onto `quality-control` at **1cee0550**. Items: B01, B02, B19, B20, B21, plus a regression introduced by 1cee0550 itself.

## Commits (oldest first)

| # | commit | items |
|---|---|---|
| 1 | d498c8e8 Hash SpinState through twice(m) | B20 |
| 2 | d5a2e52c Convert permutations to tuples once per permutation behind a function barrier | B19 |
| 3 | 203cf5b2 Check sparse projectors for roundoff through their stored entries | qc regression (new) |
| 4 | 37f3d640 Return the sparse permutation projector without copying and rescale it in place | B21 (copy, `P *= c`) |
| 5 | ff07f9ea Orthogonalize the sector projector block by block with a relative tolerance | B02, B21 (dense Q extraction) |
| 6 | 5e8f8a55 Build symmetric and antisymmetric sector bases from orbits of basis states | B01 |

Each commit passes the permutation, spin, Aqua and ExplicitImports test items on its own: 184, 186, 186, 186, 240 and 907 passes. The full suite passes at the final head; see "Tests" below.

## Per-commit measurements

Method:
- Two scratch worktrees hold the parent and the commit, each with its own env.
- The same script (`scratchpad/phase2/s1-perm/bench.jl`) ran alternately against parent and commit, three rounds each.
- Tables show the minimum over rounds. Allocations and memory are deterministic.
- Ratio is commit/parent, so below 1 is faster.
- The machine was shared. Two identical runs differ by up to ±14% per case, so ratios between 0.9 and 1.1 are noise unless allocations change.

Case names:
- `many_small` and `few_large` are the suite cases `permutations/many_small` (6×spin-½, total m=2, dim 15) and `permutations/few_large` (4×spin-3, dim 2401).
- `custom_sym` passes all n! permutations as `(perms, weights)`, which forces the projector path.
- `C8`/`C10` are translation-group irreps (trivial and k=1) on 8 and 10 spin-½ sites with total m=0. These are realistic user calls, as in `examples/translation_symmetry.jl`.
- Other realistic cases: 6×spin-1 (dim 729), 8×spin-½ with m=0 (dim 70), 5×spin-2 with m=0 (dim 381), and the Heisenberg chain N=12 at Sz=0.

### 1. Hash SpinState through twice(m) (B20)
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| hash SpinState (4096 states) | 25.1 µs | 1.5 µs | 0.06 | 0 → 0 | 0 |
| hilbert_space 12×spin-½ Sz=0 | 1.35 ms | 1.16 ms | 0.86 | 17826 → 17826 | 2263.5 |
| Heisenberg matrix_representation N=12 Sz=0 | 5.79 ms | 4.12 ms | 0.71 | 2165 → 2165 | 3742.9 |
| state_index 12×spin-½ (full space, no hashing) | 84.1 µs | 81.1 µs | 0.96 | 0 → 0 | 0 |
| many_small/symmetric | 727 µs | 495 µs | 0.68 | 9920 → 9920 | 1584.1 |
| 8×spin-½ m=0 symmetric | 420 ms | 198 ms | 0.47 | 524828 → 524828 | 111708 |

### 2. Function barrier and tuple permutations (B19)
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| many_small/custom_sym | 411 µs | 99.4 µs | 0.24 | 4819 → 500 | 1299.8 → 40.4 |
| few_large/custom_sym | 35.5 ms | 36.1 ms | 1.02 | 653 → 462 | 16642 → 16611 |
| 5×spin-2 m=0 custom_sym | 3.46 ms | 3.60 ms | 1.04 | 1200 → 481 | 4664 → 4485 |
| 8×spin-½ C8 trivial | 130 µs | 134 µs | 1.03 | 687 → 624 | 205 → 187 |
| 8×spin-½ C8 k=1 | 157 µs | 152 µs | 0.97 | 687 → 624 | 246 → 228 |
| 10×spin-½ C10 trivial | 695 µs | 659 µs | 0.95 | 814 → 735 | 768 → 740 |

### 3. Check sparse projectors for roundoff through their stored entries (qc regression)
At 1cee0550, `_is_negligible` called `maximum(abs, P)` on a `SparseMatrixCSC`. That iterates over all d² entries, which is 5.8M for few_large: 22.5 ms, against 13 µs over `nonzeros(P)`. This is the 7.5 → 27.4 ms jump the orchestrator saw in `permutations/few_large`.
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| few_large/custom_sym | 38.2 ms | 6.72 ms | 0.18 | 462 → 462 | 16611 |
| 5×spin-2 m=0 custom_sym | 3.73 ms | 2.29 ms | 0.62 | 481 → 481 | 4485 |
| 10×spin-½ C10 trivial | 649 µs | 403 µs | 0.62 | 735 → 735 | 740 |
| 8×spin-½ C8 k=1 | 162 µs | 118 µs | 0.73 | 624 → 624 | 228 |
| many_small/custom_sym (dense accumulator) | 107 µs | 99 µs | 0.93 | 500 → 500 | 40.4 |

### 4. Sparse projector without a copy, scaled in place (B21, part)
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| few_large/custom_sym | 6.84 ms | 7.04 ms | 1.03 | 462 → 442 | 16611 → 15425 |
| 5×spin-2 m=0 custom_sym | 2.21 ms | 2.20 ms | 1.00 | 481 → 471 | 4485 → 4147 |
| 10×spin-½ C10 trivial | 404 µs | 378 µs | 0.94 | 735 → 715 | 740 → 657 |
| 8×spin-½ C8 trivial | 105 µs | 100 µs | 0.95 | 624 → 606 | 187 → 169 |

No measurable time gain. Memory drops by 4–11% in every case. The commit is kept as a small, deterministic allocation reduction (one copy of P), but it is optional; the maintainer may drop it.

### 5. Block-wise orthogonalization with a relative tolerance (B02, B21)
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| few_large/custom_sym | 7.23 ms | 2.76 ms | 0.38 | 442 → 4222 | 15425 → 6904 |
| few_large/symmetric (projector path at this commit) | 6.80 ms | 2.45 ms | 0.36 | 665 → 4445 | 15436 → 6915 |
| few_large/antisymmetric | 3.41 ms | 1.67 ms | 0.49 | 1186 → 17338 | 8751 → 3927 |
| 5×spin-2 m=0 custom_sym | 2.20 ms | 1.47 ms | 0.67 | 471 → 547 | 4147 → 1739 |
| 8×spin-½ C8 k=1 | 112 µs | 68.5 µs | 0.61 | 606 → 620 | 201 → 99 |
| 10×spin-½ C10 trivial | 389 µs | 250 µs | 0.64 | 715 → 1050 | 657 → 266 |
| many_small/custom_sym | 106 µs | 88.5 µs | 0.83 | 493 → 364 | 39.6 → 30.0 |
| 6×spin-1 symmetric | 17.6 ms | 16.9 ms | 0.96 | 5627 → 5998 | 9612 → 5541 |
| 8×spin-½ m=0 symmetric | 154 ms | 145 ms | 0.94 | 282902 → 282775 | 18392 → 17841 |

The allocation count rises because there is one small dense matrix per block. Memory falls by 40–60%. 6×spin-1 and 8×spin-½ are dominated by building the projector from 720 or 40320 permutations, which commit 6 removes.

### 6. Orbit construction for `:symmetric` / `:antisymmetric` (B01)
| case | parent min | commit min | ratio | allocs | memory KiB |
|---|---|---|---|---|---|
| many_small/symmetric | 142 µs | 24.2 µs | 0.17 | 5467 → 416 | 314 → 30.7 |
| many_small/antisymmetric | 336 µs | 22.3 µs | 0.066 | 19966 → 413 | 1015 → 30.5 |
| few_large/symmetric | 3.06 ms | 1.03 ms | 0.34 | 4445 → 312 | 6915 → 4067 |
| few_large/antisymmetric | 1.95 ms | 211 µs | 0.11 | 17338 → 308 | 3927 → 775 |
| 6×spin-1 symmetric | 17.9 ms | 170 µs | 0.0095 | 5998 → 426 | 5541 → 219 |
| 8×spin-½ m=0 symmetric | 174 ms | 60.6 µs | 0.00035 | 282775 → 536 | 17841 → 42.8 |
| 5×spin-2 m=0 symmetric | 1.59 ms | 62.6 µs | 0.039 | 1446 → 366 | 1783 → 77.5 |
| 5×spin-2 m=0 antisymmetric | 1.59 ms | 53.7 µs | 0.034 | 6293 → 362 | 1775 → 44.0 |

`few_large/symmetric` is now dominated by filling the dense 2401×210 `Matrix` result (4 MiB), which is kept as the return type.

### Cumulative, suite cases (1cee0550 → head, product of the per-commit ratios)
- `permutations/many_small`: about 0.68 × 0.88 × 0.17 ≈ **0.10**, a 10× speedup.
- `permutations/few_large`: about 0.18 (qc regression fix) × 0.36 × 0.34 ≈ **0.022** against 1cee0550. Against the pre-regression 725b5f6b it is roughly 7.5 ms → 1.0 ms.

A direct A/B against baseline-qc-env at 725b5f6b, before the rebase, gave: many_small/symmetric 838 → 27.6 µs; few_large/symmetric 8.1 → 1.29 ms; 6×spin-1 23.0 → 0.22 ms; 8×spin-½ m=0 407 → 0.068 ms; Heisenberg N=12 Sz=0 7.6 → 5.5 ms.

## Analysis per item

### B20: SpinState hash
- **Correct for all inputs?** `hash(twice(x.m), h)` uses the integer 2m. `==` on `SpinState` compares `m`, which is equal exactly when `twice(m)` is. Integers hash by value across types (`hash(Int16(3)) == hash(3)`), so `SpinState{Int16}` and `SpinState{Int}` stay consistent. A test pins this. Hash values change, but no code persists them.
- **Public behaviour:** none changed.
- **Gain:** 17× on the hash itself. It reaches every Dict lookup of spin states: constrained spin spaces, sectors, and matrix representations of spin chains.
- **Decision:** implement.

### B19: function barrier and `Tuple(perm)`
- **Correct for all inputs?** `_match_perm` converts a permutation to an `NTuple{N}` of the statically known length when the substates are a tuple, and leaves it unchanged otherwise. A wrong-length permutation throws an `ArgumentError` from `_match_perm`. Before, `TupleTools.permute` threw one. A test was added.
- **Gain:** large only when the per-permutation work is small (many_small/custom 4×). Neutral elsewhere, but allocations drop in every case.
- **Decision:** implement.

### qc regression in `_is_negligible`
- The dense `maximum(abs, P)` over a `SparseMatrixCSC` is O(d²). The new version reads `nonzeros(P)`, which gives the same maximum, because stored zeros do not change it.
- **Decision:** implement as a separate commit.

### B21: custom `(perms, weights)` path
- **Copy and scaling (commit 4):** `sparse(::SparseMatrixCSC)` copied the result of `sparse!`. `P *= c` allocated a scaled copy; `_rmul!!` scales in place when `c` fits the element type, for example not for `Int` P with a `Float64` factor. P is always freshly allocated, so in-place scaling has no aliasing risk.
- **`sizehint!`:** already reserves exactly `d * length(perms)` entries per vector, which is the final count. No change.
- **Dense Q extraction:** replaced by the block-wise orthogonalization of commit 5. The d×M dense `F.Q[...]` materialization of a sparse QR no longer happens.
- **Decision:** implement (commit 4 is optional, see the table above).

### B02: absolute QR tolerance, a correctness bug
- **The bug:** `qr(P; tol=0.1)` (sparse) and `|R_kk| > 0.1` (dense) are absolute. On an orbit of k states the projector columns have norm 1/√k, which falls below 0.1 for k > 100.
- **Reproduction:** 5×spin-2 with total m=0 has dim 381 and contains a 120-state orbit. At 1cee0550:
  - Sparse: 12 columns, but ‖PQ−Q‖ = 1.0.
  - Dense: 11 columns instead of 12.
  - `:antisymmetric`: 1 column, ‖PQ−Q‖ = 1.0.
- **Fix (commit 5):**
  - P is block diagonal over the connected components of its sparsity pattern. Each component is a union of orbits of the group generated by `perms`. Each block is orthogonalized densely.
  - `:qr`: a column-pivoted Gram-Schmidt that stops when the next pivot is at most `tol`.
  - `:eig`: the eigenvectors of `Hermitian(B)` with |λ| > `tol`, with the at most `r` largest kept.
  - The tolerance is `tol = sqrt(eps) * max column norm of P`.
  - Each block also keeps at most `round(real(tr B)^2 / sum(abs2, B))` columns. This is 1cee0550's rank formula applied per block, and it is exact for a block proportional to a projector.
- **Why the tolerance is safe:** for P = p·Π, every pivot inside the rank of an n×n block is at least p/√n (the Schur complement of a projector is a projector of trace ≥ 1), while rounding is O(eps·p).
- **Why Gram-Schmidt instead of LAPACK `qr!(B, ColumnNorm())`:** geqp3 took 6.8 ms on a 120×120 rank-1 block, because of rank deficiency and norm downdating. Pivoted Gram-Schmidt with early stopping costs O(rank·n²) and took 0.24 ms for the whole projector.
- **Semantics kept from 1cee0550:**
  - `_is_negligible` zeroing, so an absent irrep gives an empty basis.
  - `isnothing(method) && return P`.
  - Complex default element type.
  - The base test that complex non-character weights give 3 columns. The trace-rank cap is what preserves this.
  - The base test that `:qr` and `:eig` agree with `normalize=false`.
- **Behaviour change:** `:eig` now uses the same relative threshold instead of `|λ| > 0.9`. An unnormalized projector, for example weights scaled by 1/240 with `normalize=false`, previously gave 0 columns for `:eig`. Now `:eig` agrees with `:qr`, and a test covers it.
- **Return type:** still a dense `Matrix{float(T)}`. `Float32`, `ComplexF32` and `BigFloat` now work with `:qr`; SPQR did not support them.
- **Regression tests:**
  - The 120-state orbit, for `:symmetric`/`:antisymmetric` and `(perms, weights)`, with sparse/dense, `:eig`, and `normalize=false`.
  - Cyclic-group irreps with complex weights.
- **Decision:** implement.

### B01: orbit construction
- **The invariant:** `permute_state` returns `only(first(combine_states(...)))`. It returns one basis state and drops the amplitude, and every `combine_states` method (FockMapper, ProductSpaceMapper, FermionicSpace, MajoranaHilbertSpace) returns amplitude 1. So every permutation acts as a permutation matrix in the basis. This holds for every space type by construction, because the projector path uses the same function. For fermions it means mode permutations carry no reordering sign, which was already the case.
- **Test of the invariant:** a test checks that every `permutation_operator` is a 0/1 matrix with unit row and column sums. It covers fermions, constrained fermions, constrained spins, two-mode fermion sites, bosons, and spin×fermion product sites.
- **Algorithm:**
  - A signed union-find over the basis states, using the n−1 adjacent transpositions as generators: O(d·n) calls to `permute_state`/`state_index`.
  - Each generator edge requires amplitude(i) = parity·amplitude(j). A contradiction inside an orbit marks it, and a marked orbit contributes no column.
  - Proof sketch: a contradiction is an odd closed walk, which is an odd stabilizer element (Schreier). That is exactly when the antisymmetrizer's column vanishes.
  - Each surviving orbit gives one column with entries ±1/√|orbit|, computed in `float(T)`, so `BigFloat` is exact; a test covers it.
- **When it is used:** `sector isa Symbol`, `Hs` a partition of `H`, and `orth_method` `:qr` or `:eig`. Otherwise the projector path runs.
- **Why not subsystem `Hs`:** there the projector is embedded with `embed`, which can add fermionic phases.
- **Tests:** the basis is compared with the all-permutations projector (Q'Q ≈ I, QQ' ≈ P, size = tr P) for 6 space types × both sectors × both methods. Also n=1, the 120-state orbit, and `BigFloat`.
- **Combinatorics:** the orbit path does not need Combinatorics.jl. `orth_method=nothing` and subsystem `Hs` still need it. The weak dependency and extension are unchanged, and the docstring states exactly when it is required.
- **Return type:** `Matrix{float(T)}`, as before. The column basis differs from the old QR basis, which was arbitrary. The base test that `(perms, weights)` and `:symmetric` give `==` results still passes.
- **Gain:** 3–2900× on the cases measured.
- **Decision:** implement.

## Reviewer findings and resolution

Two reviews ran: one before the rebase onto 1cee0550, and one after. Neither found correctness bugs in the main paths.

**Fixed:**
- **BigFloat precision.** Orbit entries were computed with Float64 `sqrt(::Int)`. They are now computed in `real(float(T))`, and a test was added.
- **`:eig` vs `:qr` disagreement on unnormalized projectors.** `:eig` used the absolute 0.9 cut. It now uses the shared relative tolerance plus the trace-rank cap, and a test was added.
- **Over-annotated `B::Matrix` in `_block_basis`.** Removed.
- **Lowercase error message in `_match_perm`.** Rewritten in package style.
- **The parameter named `sign`.** Renamed to `parity`.
- **Comment fixes:**
  - The tolerance-justification comment now states the conclusion and refers to the max column norm.
  - The orbit comment says `permute_state` discards the amplitude.
  - The docstring says "one column per orbit that contributes" and mentions the `float(T)` element type.
  - The `_QR` reference was fixed.
  - The long lines and the cryptic test comment were rewritten.
  - The test name that only made sense next to the removed 0.1 cut was renamed.
- **Undocumented trace-rank cap.** The docstring now says both methods assume P is proportional to an orthogonal projector, and gives the per-block column cap.
- **Commit hygiene:**
  - The `permutation_projector` copy/scale change was split into its own commit (4).
  - The `_match_perm` length check got a test in its own commit (2).
  - Commit messages were reworded to describe the code as it is.

**Not changed, with reasons:**
- **Non-group or mixed weights.** For these, `:qr`/`:eig` return at most the per-block trace rank, which can be fewer columns than rank(P). 1cee0550 behaved the same way (it returned the global trace rank, and was inconsistent between paths), so this is documented rather than changed.
- **Orbit path for subsystem `Hs`.** It would need the embedding phases, so it was left out.
- **Two union-find implementations.** One is signed and one is not; they are kept separate for clarity.

## Tests
- Targeted tests pass for each commit; the counts are above.
- Full suite at 596ab07f, which has the same code as the head apart from comments and docstrings: 4364/4364 passed in 17m46s. The worktree code was confirmed loaded via `pathof`.
- Full suite at the final head 5e8f8a55: 4366/4366 passed in 12m42s, with the worktree code loaded.
- Output equivalence against baseline-qc at 1cee0550: `check.jl` serializes QQ', column counts, return types and permutation operators for 135 combinations. These cover 10 space types × sectors × `:qr`/`:eig`/`nothing` × custom and cyclic-irrep weights × sparse/dense. Every combination matches, except the four 5×spin-2 cases where the baseline is wrong (B02).

## Deferred / notes for the maintainer
- Commit 4 gives no measurable time gain, only 4–11% less memory, and can be dropped.
- `few_large/symmetric` now spends most of its time allocating and filling the dense d×rank result. Returning a sparse matrix would be about 6× faster again, but changes the public return type, so that is a maintainer decision.
