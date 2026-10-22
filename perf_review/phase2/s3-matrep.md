# Phase 2, stream s3-matrep: matrix representation, sectors, BdG / single-particle

Branch `perf/s3-matrep`, rebased onto `quality-control` 1cee0550. One logical fix per commit, each with its tests.
Baseline for the final comparison: `scratchpad\phase2\baseline-qc-env` (worktree `baseline-qc` at 1cee0550).
Per-commit numbers compare each commit with its parent.

Benchmark script: `scratchpad\phase2\s3-matrep\{cases.jl,bench.jl,ab3.sh,ab2.sh}`. Cases are the
`benchmarks_v0.10.jl` matrix_representation / hilbert_space leaves (same setup and RNG order) plus realistic cases:
spin-½ Heisenberg Ns=12 in Sz=0 (`AdditiveConstraint`), Bose–Hubbard 6 sites / 6 particles (`NumberConservation(6)`),
Hubbard N=20 (2↑,1↓) via `NumberConservation` sectors and Hubbard N=6 with `NumberConservation(Hup)*NumberConservation(Hdn)` sectors,
an unconstrained fermion⊗spin⊗boson product (`mixed_product`), a product with a `GenericHilbertSpace` factor (slow path),
spin Ns=14 Sz=0 space construction, fermion `NumberConservation()` sectors, and an empty sector.
Every run serializes its outputs and compares them with the other side (`==`, else `≈`, plus `typeof`).

## Per-commit gains (each commit against its parent)

Min of the minimum times over 3 alternating rounds (A parent, B commit, 1 s per case; reruns 5 rounds × 2 s), allocations / bytes exact.
Ratio = commit / parent. Noise on minimums is about ±14 % per case, so only ratios corroborated by allocations, or well beyond
the noise, are claimed. Outputs were compared each time: `==` and same type everywhere except where noted (`≈`).

| commit | case | parent µs | commit µs | ratio | allocs (parent → commit) | bytes (parent → commit) |
|---|---|---|---|---|---|---|
| bc160f53 Keep a single amplitude type in generic apply_local_operators | heis12_Sz0 (realistic) | 14164 | 5057 | **0.36** | 388397 → 2165 | 11.09 M → 2.96 M |
| | bosehub_6_6 (realistic) | 584 | 467 | 0.80 | 984 → 984 | = |
| | mixed_product | 340 | 333 | 0.98 | 3634 → 3268 | 784 k → 774 k |
| | ps_many_cons | 810 | 889 | 1.10 (noise) | 12169 → 12169 | = |
| 953c96d6 Apply product-operator factors by tuple recursion | ps (benchmark "product space") | 679 | 359 | **0.53** | 17588 → 3764 | 1015 k → 309 k |
| | slow_path (GenericHilbertSpace factor) | 2563 | 1649 | **0.64** | 73631 → 47135 | 2.44 M → 1.56 M |
| | bosehub_6_6 | 503 | 419 | 0.83 | = | = |
| | heis12_Sz0, ps_many_cons | | | 1.04, 0.95 | = | = |
| 8bfd52f8 Partition product operators by looking up each factor's basis | ps_many_cons (benchmark) | 812 | 506 | **0.62** | 12169 → 4005 | 1059 k → 568 k |
| | ps_many (benchmark) | 2309 | 2096 | 0.91 | 9256 → 6636 | 4.83 M → 4.72 M |
| | ps | 335 | 306 | 0.92 | 3764 → 2605 | 309 k → 278 k |
| | heis12_Sz0 | 4383 | 4708 | 1.07 (noise) | 2165 → 1901 | ≈ |
| e5f6c2b5 Use one accumulator for sums on constrained multi-group spaces | bosehub_6_6 (realistic) | 443 | 288 | **0.65** | 952 → 499 | 766 k → 280 k |
| | heis12_Sz0 (realistic) | 5280 | 4366 | **0.83** | 1901 → 1094 | 2.96 M → 1.46 M |
| | ps | 318 | 262 | 0.82 (≈) | 2605 → 2063 | 278 k → 141 k |
| | slow_path | 1779 | 1575 | 0.89 | 46877 → 46351 | 1.56 M → 1.21 M |
| | ps_many_cons | 515 | 558 | 1.08 (≈; noise, memory −56 %) | 4005 → 3051 | 568 k → 249 k |
| 02df2d58 Build product-space term matrices with a direct identity kron | ps_many (benchmark), rerun 5×2 s | 1791 | 1332 | **0.74** | 6636 → 5738 | 4.72 M → 3.76 M |
| | mixed_product (realistic, small groups), rerun | 296 | 283 | 0.96 | 2678 → 2771 | 744 k → 722 k |
| 8cc60846 Drop the identity of a sum without re-filtering its terms | free_fermion (benchmark) | 409 | 367 | 0.90 | 23 → 23 | = |
| | bdg (benchmark) | 814 | 730 | 0.90 | = | = |
| 4dd08623 Check single-particle and BdG operators in one pass | free_fermion (benchmark) | 325 | 260 | **0.80** | 23 → 23 | = |
| | bdg (benchmark) | 741 | 638 | 0.86 | = | = |
| f5b6e5ff Build sector spaces with typed and presized containers | hilbert_space (benchmark) | 125 | 85 | **0.68** | 160 → 132 | 411 k → 355 k |
| | fermion `NumberConservation()` sectors (realistic) | 101 | 79 | **0.78** | 168 → 143 | 362 k → 307 k |
| | spin Ns=14 Sz=0 space, Hubbard N=6 sectors | | | 1.01, 1.02 | −29, −63 | −55 k, −58 k |

Notes:
- The B23 and B24 rows were measured when the (later dropped) B22 commit was below them; B22 does not touch the code they change.
- B23 and B24 remove whole passes over a 2000-term dictionary without allocating, so their gain shows only in time;
  each is at 0.80–0.90 on two benchmarks; treat B23 (0.90) as within noise but cheap and a safety improvement (no mutation path).
- B12: the first run (3×1 s) gave 0.99 / 1.13; the rerun (5×2 s) 0.74 / 0.96. A microbenchmark of the kron step alone summed over
  all terms: ps_many 530 vs 988 µs, mixed_product 122 vs 112 µs. So the gain is real with large identity padding and neutral on small spaces.

## Branch head (f5b6e5ff) against baseline-qc (1cee0550)

3 alternating rounds × 2 s, min of minimums. `standard`, `hubbard20_*` and `hub6_*` are single-group fermion paths this branch does not change (allocations identical; their ratios are noise). `ps` and `ps_many_cons` are `≈` (summation order), everything else `==` with identical types.

```
case                    base min us   new min us   ratio   allocs/mem base -> new
bdg                           610.4        545.8   0.894   229 / 2440727 -> 229 / 2440807
bosehub_6_6                   480.5        221.6   0.461   984 / 763733 -> 499 / 280086
empty_sector                   47.9         44.6   0.931   719 / 38336 -> 717 / 38288
fermion_numbersectors          76.6         66.0   0.862   168 / 361941 -> 143 / 306631
free_fermion                  309.5        192.3   0.621   23 / 472689 -> 23 / 472673
heis12_Sz0                  10285.3       3210.6   0.312   388397 / 11091370 -> 1094 / 1459490
hilbert_space                  99.3         70.9   0.714   160 / 410732 -> 132 / 355262
hub6_sectors_matrep           776.8        863.4   1.111   149 / 2130182 -> 149 / 2130166
hub6_sectors_space          11479.2      11632.1   1.013   62401 / 2622705 -> 62338 / 2565059
hubbard20_matrep              943.7        981.5    1.04   451 / 1052401 -> 451 / 1052417
hubbard20_space             10384.9      10680.3   1.028   59258 / 3028659 -> 59235 / 2970021
mixed_product                 288.7        240.5   0.833   3634 / 784094 -> 2771 / 722051
ps                            498.1        207.0   0.416   17588 / 1015454 -> 2063 / 140543
ps_many                      1669.8       1204.5   0.721   9256 / 4830386 -> 5738 / 3755085
ps_many_cons                  679.3        351.6   0.518   12169 / 1058693 -> 3051 / 249114
slow_path                    2208.5       1220.4   0.553   94367 / 3032196 -> 46351 / 1211635
spin14_space                 3968.9       3423.1   0.862   67633 / 10958210 -> 67604 / 10902756
standard                     1330.1       1542.8    1.16   169 / 3658566 -> 169 / 3658550
```

## Per-item analysis and decisions

### B07 — type-stable amplitude in generic `apply_local_operators(op::NCMul, …)` — implemented (`Keep a single amplitude type…`)
- The amplitude starts as `convert(mat_eltype(op), op.coeff)`; early returns give `zero(T)`; one `return` after the loop.
- Correctness: `mat_eltype(op) = promote_type(C, mat_eltype(S))` ⊇ the coefficient type, so the conversion is exact.
  Since `mat_eltype(S)` is the amplitude type of the local operators (Float64 for spins/bosons, Int for fermions), the
  arithmetic is the same promotion as before except that a real coefficient with complex amplitudes now multiplies as
  complex×complex (signed zeros only; no current atom type takes this path). For an abstract factor type `S` the fallback
  `Float64` is used exactly as before for the zero; a complex amplitude still widens `amp` (old behaviour, unstable but correct).
- Test: `apply_local_operators amplitude type` (`@inferred`, Int/Float/Complex/Rational coefficients, spins and bosons, both `transpose` branches, values against a factor-by-factor reference).

### B08 — tuple recursion in `_apply_local_operators_fast/_slow` — implemented
- `_apply_group_operators` recurses over the `ops`, `spaces`, `precomps` tuples (`first`/`Base.tail`); the terminal method
  takes three `Tuple{}`, which is strictly more specific, so there is no ambiguity. qc's signature
  `Tuple{InternalRep{T},Vararg{InternalRep{T}}}` is kept.
- `convert(C, local_amp)` replaces the old `::C` annotation (same conversion). The old `ismissing(op)` skip is gone:
  `ProductOperator` ops come only from `partition_product`/`__concretize` and never contain `missing` (qc also removed
  `_apply_local_operators(::Missing, …)`).
- Slow path (no internal representation, e.g. `GenericHilbertSpace` factors): same recursion; the substate access is a
  small Union but no closure boxing.
- Inference: all product cases infer a concrete `Tuple{ProductState{…}, T}` with 0 allocations per application.

### B10 — `partition_product` — implemented
- Looks up each factor's basis index (recursive search over a tuple of bases, `findfirst` otherwise), then builds the
  per-group `NCMul`s in increasing group order with comprehensions (same narrowed vector element types as before).
- Checked field-by-field (`ops`, op types, factor container types, coefficient `===`, `spaces`, `inds`) against the old
  implementation for every term of 8 operators and the three concretizers, plus the error path (same `ArgumentError`, qc wording).

### G-2 (from the gap profiler) — one accumulator for sums on constrained multi-group spaces — implemented
- Applies to `__matrix_representation(::NCAdd, …)` on spaces that are not an unconstrained `ProductSpace` (constrained or
  sector spaces over a product space). The unconstrained `ProductSpace` keeps the per-term kron path.
- Element type: the old result was (per-term matrices in `_repr_eltype(mat_eltype(prodop), repr)`) summed, plus
  `op.coeff * Eye` (Float64). The new code accumulates in the promoted term type `Tterms` (so a typed `Int` repr still
  throws `InexactError`, and `Float32` still rounds per term) and adds the identity in `promote_type(Tterms, eltype(coeff*Eye))`;
  when that equals `Tterms` the identity goes into the same accumulator.
- Floating-point: duplicate entries are summed in COO order instead of the tree sum, so results are `≈` (ps, ps_many_cons
  differ in the last bits; all other cases `==`).
- Tests: `Constrained product space agrees with the unconstrained one` (four operators across atom types, untyped, `ComplexF64`,
  `Matrix{ComplexF64}`, `Float32`, `Int`, plus the `InexactError` case).

### B12 — direct `I ⊗ A ⊗ I` kron on an unconstrained `ProductSpace` — implemented
- `_kron_groups(matrices, active, ::EagerSparseRepr)`: the groups between the first and last active group are folded with
  `kron` (Bool sparse identities in between, which keep values and element type), then `_kron_identities` writes
  `kron(I(l), core, I(r))` directly in CSC form (no `@inbounds`). Element type = core promoted with the identity's (`Eye`, Float64),
  exactly the old `foldl(kron, reverse(…))` type. Dense and lazy representations, and non-CSC term matrices, use the old path.
- First version computed the element type with `promote_type(map(eltype, matrices)...)` over a `Vector{AbstractMatrix}`,
  which cost ~6 µs per term and made small spaces slower; the committed version promotes only two types.
- Kron ordering (first group fastest) is checked by the constrained/unconstrained test on fermion⊗spin⊗boson operators with
  gaps between active groups, by `mixed_product` (`==` to baseline) and by a direct `_kron_identities` vs `kron` test.
- `iszero(op.coeff)` skips `+ 0*Eye` and converts to the same element type; entries are `==`, but a `-0.0` entry is no
  longer normalised to `+0.0` by the addition (reviewer finding 2; accepted and stated in the commit message).
- The closure variable is renamed `local_op` (no boxed capture).

### B22 — BdG `normal_order_to_bdg` — rejected (implemented, measured, dropped from the branch)
- A method for sparse CSC input emitted the `[h Δ; -conj(Δ) -conj(h)]` entries from the stored entries
  (`h = m/2`, `Δ[r,c] = (δ[r,c] − δ[c,r] + conj(δd[c,r]) − conj(δd[r,c]))/4`) and assembled them with one `sparse!`
  (results `==` on the benchmark, `≈` in general).
- Allocations dropped 229 → 50 but bytes only 2.44 M → 2.43 M, and time did not improve: bdg 1.12 and 1.10 in two A/B runs,
  and `normal_order_to_bdg` alone 196 vs 198 µs on the benchmark matrix. The Phase 1 −11 % did not reproduce.

### B23 — `remove_identity` — implemented
- `NCAdd(zero(a.coeff), a.dict; filter_zeros=false, filter_scalars=false)`. With both filters off, NCP's outer constructor
  returns before touching the dictionary, and the inner constructor shares it without mutation. The two callers
  (single-particle and BdG `matrix_representation`) only read the result. Before, the default filtering iterated over (and
  could delete from) the user's dictionary; the NCAdd invariant (no zero or scalar keys) made that a no-op in practice.
- Test: in `Single-particle and BdG operator checks` (operator and dictionary unchanged after representing it).

### B24 — fused validity checks for single-particle / BdG — implemented
- `_is_quadratic_fermion_operator(op, number_conserving)` replaces `_isfermionic(op) && isquadratic(op) (&& isnumberconserving(op))`
  with one pass over the dictionary keys: two `FermionSym` factors, and one creation and one annihilation operator for the
  single-particle space.
- Bug fixed on the way: a quadratic Majorana operator passed `_isfermionic`/`isquadratic` (Majoranas are `AbstractFermionSym`)
  and then failed with a `FieldError` (`creation`) on both spaces (confirmed on the baseline). It now gives the documented
  `ArgumentError`. Test: `Single-particle and BdG operator checks`.
- `mode_position` hashing in `operator_indices_and_amplitudes!` (the other 17% of free_fermion) was not changed.

### B18 — sector construction — implemented
- `groupby(f, itr, B)` keys its `OrderedDict` by `nonmissingtype(promote_op(f, eltype(itr)))` when that is concrete
  (else `Any` as before) and fills it behind a function barrier; `_sector_space` fills a presized state vector and a
  presized `OrderedDict{B,Int}` (same last-wins semantics as `OrderedDict(zip(...))`).
- Inference is sound for the actual element type of `itr`, so keys are never converted; an empty grouping now has a concrete key
  type instead of `Any` (only when inference gives a concrete type; `SectorHilbertSpace` types for all tested cases match the
  baseline, including the empty `ParityConservation(-1)*NumberConservation(2)` sector).
- Test: `groupby key types`.

### Not done / deferred
- G-5 (avoid converting all N site states to `Int` and back per term; better accumulator size hint for spin chains): not attempted.
- `BdGHilbertSpace` `matrix_representation` does not canonicalize `repr` (`:dense`, `ComplexF32` give a `MethodError` in
  `_matrix_representation_single_space`). Pre-existing in qc; reported, not fixed here.
- Summing per-term sparse matrices on an unconstrained `ProductSpace` (`_sum_tree`, ~1 ms of ps_many's ~2 ms) is now the
  dominant cost there; a COO accumulation of the kron entries could replace it but was not measured.

## Tests
- Per commit (separate worktree, each commit checked out on its own): targeted test files all pass —
  bc160f53 matrix_representation.jl 455/455; 953c96d6 product_space.jl + hilbert_space.jl 206/206; 8bfd52f8 matrix_representation.jl 455/455;
  e5f6c2b5 matrix_representation.jl + constrained_space.jl 535/535; 02df2d58 (as 6925480b) matrix_representation.jl 506/506;
  8cc60846 fixednumberfock.jl + bdg.jl 91/91; 4dd08623 matrix_representation.jl + bdg.jl + fixednumberfock.jl 614/614;
  f5b6e5ff (as cc3fe976) sectors.jl + constrained_space.jl 189/189.
- Full suite on the branch head: 4012/4013; the one failure was the ExplicitImports check for `SparseArrays.AbstractSparseMatrixCSC`
  (not public), fixed in 02df2d58; afterwards matrix_representation.jl + aqua.jl + explicit_imports.jl 540/540.
- Cherry-pick onto `perf/integration` (982cca41): all eight commits apply without conflicts. Full suite on that combined tree: 5058/5058.

## Reviewer findings (one general-purpose reviewer subagent, diff 725b5f6b..HEAD before the history clean-up) and resolution
1. **Bug (G-2):** the single accumulator was built in the type promoted with the identity's Float64, so a typed repr was no
   longer applied per term (`Int` repr returned Float64 instead of throwing `InexactError`; `Float32` no longer rounded per term),
   and the commit message overclaimed. **Fixed:** accumulate in the promoted term type, add the identity in the promoted type
   (into the same accumulator when the types agree, otherwise `summed + coeff*Eye` / `convert`). Test extended with `Float32`,
   `Int` and the `InexactError` case. The fix was folded into the G-2 commit.
2. Signed zeros after skipping `+0*Eye` (B12): accepted, documented in the commit message.
3. `_kron_identities` correct; annotations `l::Integer, r::Integer, A::SparseMatrixCSC` were not needed for dispatch: removed;
   the CSC check stays `SparseMatrixCSC` (`AbstractSparseMatrixCSC` is not public and fails the ExplicitImports test).
4. `partition_product` docstring did not state the return value: fixed.
5. B22 comment overclaimed exact equality: moot, B22 dropped.
6. `remove_identity` comment made a promise about callers: reworded to name the callers. (The test checks that the operator is
   unchanged; it is not described as a mutation regression test.)
7. B24 `number_conserving::Bool` annotation: removed.
8. Tuple recursion, B07: no issues.
10. B18: inference on `B` instead of the element type of `itr` could convert keys if a caller passed other states: changed to
    `promote_op(f, eltype(itr))`.

## Notes for the maintainer
- `BdGHilbertSpace` `matrix_representation(op, H, :dense)` / typed reprs raise a `MethodError` (repr not canonicalized). Pre-existing in qc.
- On a constrained product space with a typed `Float32` repr the result is rounded per whole product term, while on the
  unconstrained product space it is a product of rounded group matrices; they differ in the last Float32 bits. Pre-existing.
- Whether a typed representation with an `Int` element type throws can depend on the space when `projection=true`
  (terms that map outside the space are never converted). Pre-existing.
