# Phase 2, stream s2-ptrace: partial trace, embedding, product-space mapper

Branch `perf/s2-ptrace`, rebased on `quality-control` at 1cee0550. Eight commits, one fix each.
Full test suite at the branch head: 3722 / 3722 passed (19m48s, worktree code loaded).

## Method

- Benchmark cases: `scratchpad/phase2/s2-ptrace/cases.jl`. It contains the suite setups
  (`partial_trace/*`, `partial_trace_algorithms/*` as `pt_alg/*`, `embed/*`, `complement/*`),
  the `benchmark/precomputed_vs_eager.jl` cases (`pve/*`), and extra realistic cases:
  - `hubbard/*`: 6-site spinful Hubbard at half filling (dim 924), traced to 2 sites.
    Dense, sparse and pure-state input, maps, and partial transpose.
  - `bosonprod/*`: 3 fermions times 3 bosons truncated at 4 (dim 512), traced to 2 bosons.
  - `spin/*`: spin-1/2 chain with Sz = 0 (dim 252), and `spin/hilbert_space_14`.
  - `xover/*`: crossover inputs for the algorithm choice (5% and 20% dense sparse matrices,
    `UniformScaling`).
  - `mapapply/*`: precomputed maps applied to dense, sparse and pure-state input.
- Every input comes from its own `Xoshiro` seed.
- **Per commit:** each commit is measured against its parent. Two scratch worktrees are
  checked out at parent and child, and the runs alternate A B A B A B (3 rounds,
  `seconds=1`). Each table cell is the min over the rounds. Allocations come from round 1
  and are exact.
- Every case's output was serialized and compared between parent and child. All were `==`.
  The summation order changes only where noted, and those cases still came out `==`.
- Machine noise is up to ±14% per case. Ratios inside that band are not counted as
  changes unless the allocation counts move too.

## Per-commit results (child vs parent)

| commit | relevant cases (min time parent → child, ratio; allocations) |
|---|---|
| **Compute product-space phase factors with map over the factor tuples** (B06) | product partial_trace standard 274.9 → 60.4 µs **4.55×** (15443 → 659 allocs); product map 297 → 99 µs 3.0×; partial_transpose/product 4391 → 348 µs **12.6×**; bosonprod partial_transpose 295 → 10.3 ms **28.7×** (11.0M → 1049 allocs); bosonprod dense SubAlg 2.62 ms → 137 µs 19×; bosonprod map build 2.65×. Fermion-only cases unchanged. |
| **Loop over complement states first in SubsystemPartialTraceAlg** (B05) | pt_alg Sparse space Dense SubAlg 6.36 ms → 295 µs **21.6×**, Sparse SubAlg 17.4×; SubAlg map build (sparse space) 5.73×; pve eager 6.7×, pve construct 2.9×; hubbard dense SubAlg **9.4×**, pure 7.9×; spin dense SubAlg 3.5×; Full space Dense SubAlg 1.31×; Full space Sparse SubAlg 0.89× (within noise, +9 allocs). |
| **Precompute subsystem indices in FullPartialTraceAlg** (B11) | pt_alg Sparse space Dense FullAlg 818 → 291 µs **2.8×**, Full space Dense FullAlg 1.8×; partial_trace/fermions/standard 1.74×; hubbard dense FullAlg 3.2×, pure 4.3×; bosonprod dense FullAlg 8.2×; spin dense FullAlg 4.4×; very sparse inputs (hubbard sparse, Full space sparse 1%) 0.95–1.02×, unchanged within noise. |
| **Visit stored entries directly in FullPartialTraceAlg for sparse input** (B26) | pt_alg Full space Sparse FullAlg 63.9 → 32.3 µs **1.98×** (memory 404 → 70 KB); xover 5% FullAlg 2.5×; partial_trace/fermions/standard 1.21× (310 → 179 KB); hubbard sparse 1.16×; others 1.06–1.07× with lower memory. |
| **Store partial-trace maps as a column-sorted COO superoperator** (B03) | partial_trace/fermions/map 110 ms → 0.97 ms **113×** (270 MB → 1.9 MB); Full space map build 50×; hubbard map build 24×; pve construct 2.8×; dense apply: fullspace **131×**, hubbard 83×, pve 27×; pure-state apply: fermions 158×, hubbard 35×, pve 11×; embed apply: pve 2.8×, in place 4.3×; fermions embed map with sparse input 2.43 s → 0.63 ms (49.8M → 78 allocs); **sparse-matrix input: fermions 1.37×, fullspace 3.5×, hubbard 1.41×, pve 1.44×** (no regression); eager `embed`/`partial_trace` unchanged. |
| **Choose the partial-trace algorithm from a cost model of both kernels** (B28) | pt_alg Full space Sparse default 106 → 32 µs **3.3×** (now FullAlg); xover 5% default 1.48×; every other default case unchanged within noise (same choice as before, or both algorithms equally fast). |
| **Convert vectors of substates to tuples of static length in ProductSpaceMapper** (G-3) | spin `hilbert_space(S, 1:14, AdditiveConstraint(0))` 5.20 → 2.41 ms **2.16×** (67633 → 2425 allocs); partial-trace cases unchanged. |
| **Keep the ProductSpaceMapper piece table concretely typed** (B16) | partial_trace/product space/standard 47.8 → 21.4 µs **2.23×**; product map 1.75×; embed/product space 1.61×; complement/product space 1.37×; partial_transpose/product 1.44×; bosonprod sparse 1.4–1.5×; mapper build alone 54 → 24 µs. |

Raw tables: `scratchpad/phase2/s2-ptrace/pc/<tag>_summary.txt`. Output comparisons:
`pc/<tag>_compare.txt`.

## Items

### B06: product-space phase factors (implemented)
- **Correctness:** `prod(map(f, s1.states, s2.states, factors); init=1)` multiplies in the
  same order as the old loop, so results are identical. An empty tuple gives 1. The states
  of a ProductSpace always have one entry per factor.
- **API:** no change. Return types become concrete.

### B05: SubsystemPartialTraceAlg loop order (implemented)
- **Semantics:**
  - Each (subsystem, complement) combination is visited once.
  - With `skipmissing=false` the kernel throws exactly when some combination is missing,
    which is the same condition as before. The first state reported may differ.
  - For each output element, terms are added in complement-state order, as before. The
    dense outputs measured were `==` to the baseline.
- **Buffer type:** the hit buffer takes its element type from the first combination. This
  relies on `combine_states` returning the same types for all inputs. Every mapper in the
  package does, and the code comment states the assumption.
- **Test:** a new test checks that the algorithms agree, and that they throw on missing
  states for constrained, product and sparse inputs.

### B11: FullPartialTraceAlg precompute (implemented)
- **Precompute:** the per-state splits carry the subsystem index, computed once per basis
  state.
- **Throw semantics are unchanged:**
  - A missing subsystem state of J1 throws whatever the complement states.
  - A missing subsystem state of J2 throws only when the complement states match.
- **Inlining:** without `@inline` on the per-pair kernel, dense FullAlg was about 2× slower
  than the baseline.
- **Specialization:** the kernels take `op::F` and specialize on it. Without that, the map
  builder dispatched dynamically on every pair (380k allocations).
- **Phase 1 regression on very sparse input:** not reproduced. Those cases measured
  0.95–1.02×.

### B26: sparse input to FullAlg (implemented)
- A `SparseMatrixCSC` walks `nzrange`; other matrix types keep the generic path.

### B03: partial-trace map storage (implemented, as a sorted COO with a block pointer)
- **Design:** instead of the transposed CSC proposed in Phase 1, `PartialTraceMap.map` is a
  `SuperOperatorCOO{T} <: AbstractMatrix{T}`:
  - the nonzeros sorted by (column, row);
  - a block pointer over the `D = dim(H)` columns of the input matrix.

  A transposed CSC still has a `dim(Hsub)^2 + 1` column pointer. More importantly, it makes
  sparse input slow (a dot product per output column), and the brief requires no
  regression for sparse input. The COO form serves every access pattern in O(nnz):
  - dense apply: stream over the entries;
  - pure state: loop over the blocks and skip `ψ[J2] == 0`;
  - sparse matrix input: merge each input column with its block;
  - sparse vector input: a forward scan per block;
  - adjoint (embed): scatter.
- **Public behaviour:**
  - `sparse(op)` returns a `SparseMatrixCSC` for both `PartialTraceMap` and `EmbedMap`.
    Before, `sparse(emb)` returned an `Adjoint` of a CSC.
  - The adjoint relations hold: `pt'.map == emb.map` and `emb'.map == pt.map`. Equality is
    defined directly on `SuperOperatorCOO`.
  - Sparse input still gives sparse output. `show` and `nnz(op.map)` are unchanged, because
    repeated entries are summed exactly like `sparse!` does, stored zeros included.
- **Field types:** `PartialTraceMap.map` and `EmbedMap.map` change type. These fields are
  not documented, and the tests use them only through `==`/`≈`, which still work. The
  no-complement maps (basis transformations) use the same type.
- **`@inbounds`:** `PartialTransposeMap` keeps its CSC map and the qc kernel
  `_apply_superop_on_pure_state!(::SparseMatrixCSC)`. This branch adds no `@inbounds`, and
  the new kernels rely on bounds-checked indexing.
- **Not done:** routing sparse input to the eager FullAlg kernel, the Phase 1 mitigation.
  It differs from the map when a user passes a `complement` whose states do not cover H.

### B28: default algorithm (implemented, re-derived after B05/B11)
- **Cost model**, in units of one FullAlg visit of an input element:
  - Sub costs `16·dsub·dcomp + c·d²/dcomp`, with `c` = 5 for dense input and 20 for sparse
    input (each term does a binary search in the sparse matrix).
  - Full costs `25·d + nvisits`.
- **Calibration:** the weights were fitted to the post-B05/B11 timings of the cases above:
  sparse space, full space, Hubbard, boson product and spin chain; dense, 1%/5%/20%/2-per-row
  sparse, and `UniformScaling`.
- **Choices:** the model picks the faster algorithm in every measured case except
  `pt_alg/Sparse space/Dense`, where the two are within 10%.
- **UniformScaling:** handled with `nvisits = dim(H)`; FullAlg is chosen, and is 1.8× faster.
- **Maps:** use the dense rule with `nvisits = d²`, so SubAlg is chosen.

### G-3: `combine_states(::AbstractVector, ::ProductSpaceMapper)` (implemented)
- The tuple length comes from the mapper type, the vector length is checked, and the index
  is `firstindex`-based.

### B16: ProductSpaceMapper construction (implemented, no API change)
- The piece table uses a concretely typed coordinate vector, one `sortperm`, and a `Dict`
  for slot lookup.
- The remaining cost is the inner `state_mapper` calls and building the type-domain
  `PieceIndex` values, which are inherent to the design. Letting callers pass in a mapper
  would be an API change, so it is deferred.

### B17: sparse `generalized_kron` copy (rejected)
- Assembling in place gave no gain: embed/fermions 0.85×, kron_sparse 0.88×, embed/product
  0.94×, with identical memory.
- That measurement used the old moving baseline. It still showed no allocation change,
  because the broadcast into an empty sparse matrix is cheap. Reverted.

### B27: complement (rejected as a separate fix)
- `_find_compatible_complementary_states` must split every state of H. I found no general
  shortcut that is safe for constrained H.
- B16 already makes `complement/product space` 1.37× faster.

## Reviewer findings (subagent review of `git diff 1cee0550..HEAD`)

No correctness bug was found. These findings were fixed in the commits they belong to:
- **Stale comments:** `partial_transpose.jl` pure-state kernel, the `_foreach_partial_trace_term`
  docstring, and the cost claim in the `SuperOperatorCOO` docstring.
- **Equality and indexing:** added `==` for `SuperOperatorCOO` and its adjoint, so it no
  longer goes through elementwise `getindex`. `getindex` now uses
  `Base.@propagate_inbounds`.
- **Size check:** `_superoperator_coo(::SparseMatrixCSC, D)` checks its size.
- **Fallback:** sparse-matrix input to a map that is not a `SuperOperatorCOO` falls back to
  `mul!`.
- **G-3:** indexing is now `firstindex`-based.
- **Tests:** removed duplicated and unrelated test lines. Added tests for an empty block,
  entries that cancel to a stored zero, `D = 0`, out-of-range indices, no-complement maps,
  and a sparse vector with entries in an empty block.
- **Naming and wording:** `nothing` as a sentinel for the dense term cost became a named
  constant. Narrative wording in the code comments and all commit messages was rewritten
  to describe the code as it is.

Not changed:
- **Hit-buffer element type (B05):** it assumes `combine_states` returns consistent types.
  The assumption is documented and holds for all mappers in the package.
- **Recomputing `s2` per complement state (B05):** at most 2× the phase-factor calls, and
  it did not show in profiles.

## Deferred / notes for the maintainer

- Let callers pass a `ProductSpaceMapper` into partial trace and embed, to skip about 20 µs
  of construction per call (B16).
- **Pre-existing (qc), not changed:** `_apply_ptmap_on_vec` allocates `zeros(eltype(ψ), …)`,
  so a map with complex entries applied to a real ψ would throw `InexactError`. Every map
  the package builds has real entries.
