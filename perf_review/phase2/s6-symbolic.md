# Phase 2, stream s6-symbolic: symbolic algebra (B09, B14)

Covers both repositories:
- **FHS:** branch `perf/s6-symbolic`, based on `quality-control` 1cee0550.
- **NCP:** branch `perf/ncp-symbolic`, based on NCP main 85bf84c. These commits are local and not pushed. FHS gets their gains only after an NCP release (or a dev checkout).

## Commits

| repo | commit | title |
|---|---|---|
| NCP | a56cde8 | Sort powers only for exponents 0 and 1 (A1) |
| NCP | 94e89dc | Convert a new coefficient to the value type before the zero check in add_term!! (bug fix) |
| NCP | ff99d2d | Hash each new key of a sum once in add_term!! (A2) |
| NCP | 41ad5cf | Size the Dict of sorted terms for all terms up front (A3) |
| NCP | 3fb19cf | Remove terms that underflow to zero when add! scales a sum (bug fix) |
| NCP | b0786c5 | Skip the filter pass when adding two sums with the same coefficient type (A4a) |
| FHS | c440e975 | Return a Tuple-backed product, or 0, as the effect of an operator on a state (A6) |
| FHS | 02e0ea71 | Act on symbolic states with a Tuple-backed one-factor product (A5) |
Each commit is one change plus its tests.

A9 ("Hash FermionSym with one hashing round fewer") was implemented and measured but is **not** on the branch: its per-commit gain was 3–5%, inside the noise and with no allocation change. It is kept on the branch `perf/s6-symbolic-a9-rejected` (2d19d7c6).

## Per-commit gains
Each row compares a commit with its parent. The ratio is child/parent of the min-of-3 minimum times. Allocations and memory are exact.

| repo | commit | case | parent µs | commit µs | ratio | allocs | outputs |
|---|---|---|---|---|---|---|---|
| NCP | Sort powers only for exponents 0 and 1 | symbolic/sum_square | 807.8 | 584.3 | **0.72** | 10543→7961 | == |
| NCP | ″ | real/bdg_square | 395.3 | 300.6 | **0.76** | 4032→2966 | == |
| NCP | Convert … before the zero check (bug fix) | symbolic/sum | 199.9 | 204.7 | 1.02 | 5717→5717 | == |
| NCP | ″ | real/hubbard12 | 71.5 | 76.7 | 1.07 | 3477→3477 | == |
| NCP | ″ | real/krylov | 278.9 | 285.5 | 1.02 | 10413→10413 | == |
| NCP | ″ | real/bdg_prod | 218.1 | 196.2 | 0.90 | 1746→1746 | == |
| NCP | ″ | real/heisenberg12 (from the next row's parent) | — | — | — | 3020→3031 | == |
| NCP | Hash each new key of a sum once | symbolic/deep_product | 629.0 | 561.9 | **0.89** | 2454→2454 | == |
| NCP | ″ | symbolic/sum_square | 636.1 | 600.8 | 0.94 | 7961→7961 | == |
| NCP | ″ | symbolic/sum | 192.1 | 185.8 | 0.97 | 5717→5717 | == |
| NCP | ″ | symbolic_states/projected_op_on_state | 203.4 | 197.3 | 0.97 | 7008→6909 | == |
| NCP | ″ | real/heisenberg12 | 92.5 | 78.6 | **0.85** | 3031→2350 | == |
| NCP | ″ | real/hubbard12 | 75.3 | 67.0 | **0.89** | 3477→3165 | == |
| NCP | ″ | real/bdg_square, bdg_prod, krylov | — | — | 0.98–1.09 | ≈ unchanged | == |
| NCP | Size the Dict of sorted terms up front | symbolic/deep_product | 559.3 | 417.0 | **0.75** | 2454→2441 (655→572 KiB) | == |
| NCP | ″ | symbolic/sum_square | 589.4 | 602.7 | 1.02 | 7961→7948 (988→955 KiB) | == |
| NCP | ″ | real/bdg_prod | 177.9 | 158.5 | **0.89** | 1746→1738 | == |
| NCP | ″ | real/krylov | 257.7 | 233.3 | 0.91 | 10296→10251 | ≈ (fp order) |
| NCP | Remove terms that underflow to zero (bug fix) | symbolic/sum | 186.0 | 192.7 | 1.04 | 5717→5717 | == |
| NCP | ″ | real/krylov | 255.0 | 251.8 | 0.99 | 10251→10251 | == |
| NCP | Skip the filter pass when adding two sums | symbolic/sum | 204.8 | 168.4 | **0.82** | 5717→5717 | == |
| NCP | ″ | real/hubbard12 | 71.7 | 66.8 | 0.93 | 3165→3165 | == |
| NCP | ″ | real/heisenberg12 | 80.1 | 81.5 | 1.02 | 2350→2350 | == |
| FHS | Return a Tuple-backed product, or 0, … on a state | symbolic_states/projected_op_on_state | 198.9 | 195.1 | 0.98 | 6927→5487 (−21%) | == |
| FHS | ″ | real/krylov | 250.0 | 235.6 | 0.94 | 10251→8091 (−21%) | == |
| FHS | Act on symbolic states with a Tuple-backed one-factor product | symbolic_states/projected_op_on_state | 199.5 | 191.4 | 0.96 | 5487→3903 (−29%) | == |
| FHS | ″ | real/krylov | 252.4 | 226.0 | **0.90** | 8091→5279 (−35%) | == |
| FHS | (rejected) Hash FermionSym with one hashing round fewer | symbolic/deep_product | 451.3 | 431.1 | 0.96 | 2441→2441 | == |
| FHS | ″ | symbolic/sum_square, sum, hubbard12, bdg_prod | — | — | 0.97–0.98 | unchanged | ==/≈ |

Notes on the table:
- The A1 row was measured on 164f456. That commit has the same `src` as a56cde8; only the test's `try`/`finally` differs.
- The NCP bug fix "Convert …" costs about 11 allocations on heisenberg12 (ComplexF64 sums that get Int/Float64 coefficients through the generic path). The next commit puts those values on the fast path.
- The A9 microbenchmark (`hashab.jl`, min of 3 per side): deep_product 420.7→409.6 µs, sum_square 561.0→539.8 µs, a Dict of 1320 three-factor products 135.3→140.5 µs.

## Method
- **Script:** `scratchpad\phase2\s6-symbolic\bench.jl`. Each case is run with `@benchmark` (2 s) in a fresh process. The script writes a canonical form of each result (type, coefficient, sorted printed terms), and `compare.jl` compares parent against child.
- **Per-commit rows:** parent and child worktrees, alternating A B A B A B. The tables show min-of-3 minimum times and allocations.
  - NCP commits use FHS at 1cee0550.
  - FHS commits use NCP at b0786c5.
- **Cases:**
  - The `benchmark/benchmarks_v0.10.jl` setups: symbolic/sum, sum_square, deep_product, and symbolic_states/projected_op_on_state.
  - Realistic cases:
    - `real/hubbard12`: spinful Hubbard Hamiltonian with N=12.
    - `real/heisenberg12`: spin-½ Heisenberg chain with a field, N=12.
    - `real/bdg_prod` and `real/bdg_square`: products of BdG-type sums on 8 modes.
    - `real/krylov`: 6 steps of `w = H*v - 0.5v - 0.25v_prev` on symbolic kets of a 6-mode chain.
- **Noise:** about ±14% per leaf for identical code. Only ratios below about 0.9, or ones backed by allocation changes, count as real.

## Per-item analysis

### A1 (NCP): no re-sort in `^` for exponent ≥ 2
- **Correctness:**
  - `Base.power_by_squaring` returns `copy(x)` for p=1 and `one(x)` for p=0. For p≥2 every step is a `*`.
  - When autosort is on, every `*` method sorts its result: NCMul*NCMul through `bubble_sort!`, and sums through `_mul!!` → `_add_sorted_products!!`.
  - Sorting runs inside `with(_autosort => false)`, which is restored afterwards. So the setting read by each `*` and by the check in `^` is the same dynamic value, including inside nested `with_autosort` scopes.
  - With autosort off nothing changes.
  - A re-sort of a sorted sum is idempotent, so skipping it gives `==` results.
  - Zero-result grades come from the `*` calls, as before.
  - The only case not covered is a `mul_effect` that toggles the *global* default mid-computation, or another thread doing so. That is racy today anyway.
- **Test:** "Powers are sorted exactly when autosort is enabled" covers global on/off, scoped on/off, nested scopes, atoms, products and sums of bosons and fermions, and p = 0:5.
- **Public behaviour:** unchanged.
- **Decision:** implement.

### NCP bug fix: convert before the zero check in `add_term!!`
- Found by the reviewer.
- `add_term!!` tested `iszero` on the unconverted sum, then stored it converted to the Dict's value type. A coefficient that rounds to zero (`promote_type(Float16, Rational{Int}) == Float16`) therefore left a zero term. Baseline reproducer: `add!!(Float16(1)*f1 + 0, (1//10^8)*f2 + 0)` has a `0.0` term.
- **Fix:** convert first when the value fits the value type (`_fit_value`), then test for zero.
- **Test:** added.

### A2 (NCP): one hash per new key in `add_term!!`
- **Mechanism:** BangBang's `modify!!` calls `ht_keyindex2!` and then `_setindex!` without the short hash, so a new key is hashed twice. It also calls `isequal(key, key0)`, which is an O(length) factor comparison for a key that is already of the key type.
- **New method:** a `Dict{K,V}` method that uses `Base.ht_keyindex2_shorthash!`, `Base._setindex!(d, v, k, -index, sh)` and `Base._delete!`, plus the fields `keys`, `vals` and `age`.
  - Values that fit `V` are converted. Others fall back to the generic method, which widens (for example Bool + Bool).
  - Keys that are not `K` go to the generic method, which converts them.
- **Base internals:** NCP compat is `julia = "1.12"`.
  - The functions are identical in 1.12.7 and 1.13.2. They are the internals that Base's own `setindex!` uses, and that BangBang already depends on.
  - The method is defined only if `@static` checks find the three functions, the 5-argument `_setindex!` method and the three fields.
  - A changed meaning in a future Julia would not be detected by the guard. The reference-comparison test (2000 random insert/update/delete steps against the public Dict API, for Int, Float64 and Bool values) would catch it on CI.
- **Behaviour:** unchanged. The key is replaced on update, as in Base `setindex!` and BangBang.
- **Decision:** implement, with the guard.

### A3 (NCP): `sizehint!` in `_sum_sorted_terms`
- **Correctness:** reserves room for `length(sorted_terms)` entries, the upper bound on terms. When many terms cancel, the extra memory is bounded by the length of the term vector.
- **Iteration order:** the Dict's iteration order changes. Downstream floating-point sums that iterate over the terms can therefore change in the last bits. `real/krylov` and the BdG cases compare `≈`, not `==`.
- **Decision:** implement.

### NCP bug fix: `add!` leaves underflowed zero terms
- Found while checking A4a's invariant.
- `add!(a, b, α, β)` scaled the terms in place and removed zeros only when `iszero(β)`. Baseline reproducer: `add!(1e-200*f1 + 1.0, 0, 1, 1e-200)` keeps `f1 => 0.0`.
- **Fix:** remove zeros after any scaling.
- **Cost:** a second O(n) pass for non-`One` β, including exact types that cannot underflow. This was accepted for simplicity and is noted by the reviewer.
- **Test:** added.

### A4a (NCP): skip the re-filter in `+(::NCAdd, ::NCAdd)`
- **Invariant:** terms of an `NCAdd` have no zero values and no scalar keys.
  - Every constructor filters by default.
  - `_set_ncadd!` filters zeros.
  - `_add_scaled!!` diverts scalar products to the coefficient.
  - `add!` now removes underflowed zeros.
  - `add_term!!` leaves out zero results after conversion.
- **The one remaining hole** (reviewer): `copy_dict(a, K, promote_type(C1, C2))` converts the terms of `a` and can round them to zero when the promoted type is narrower (Rational → Float16).
- **Fix:** zero filtering is kept exactly when `C1 !== promote_type(C1, C2)`. The scalar filter is always skipped.
- **Tests:** cancellation, widening, Bool + Bool, mixed key types, and Rational + Float16 in both orders.
- **Decision:** implement this safer variant.

### A6 (FHS): `mul_effect(op, ::SymbolicState)` returns `0` or a Tuple-backed product
- **Change:** before, it was always `amp * state`, which allocates `NCMul(amp*1, [state])`.
  - A zero amplitude now returns `0`. NCP's `splice!!(…, ::Number)` deletes the term, as the zero product did before.
  - Otherwise it returns `NCMul(amp * 1, (state,))`, with the same coefficient value and type as `amp * state`.
- **Rejected variant:** returning the bare state when `isone(amp)` (Phase 1's fhs_state2). It would change the coefficient type, for example a Float64 amplitude 1.0 of `S⁺` on spin-½ would give an Int-coefficient result.
- **Verification:** baseline and patched results have identical values and types for fermion, spin and boson kets and bras.
- **Test:** "Operators on symbolic states give the coefficient amp * state".

### A5 (FHS): `NCMul(1, (op,))` in `_apply_symbolic_operator_to_state`
- **Change:** the precomputation (a `map` over the factors) becomes a tuple.
- **Overloads checked:**
  - The generic `apply_local_operators`, which uses `Iterators.reverse(zip(...))`.
  - FockNumber fermions, which use `eachindex(factors, bitmasks)` on tuples.
  - `FixedNumberFockState` (`reverse(tuple)`).
  - Majoranas (`Iterators.map` over the precomputation tuple).
  - Constrained and sector spaces (forwarded to the parent).
  - Product spaces and InternalRep.
- **Results:** identical to baseline for all of them, in both directions (ket and `transpose=true` bra).
- **Test:** "Operators on symbolic states agree with matrix representations" covers fermionic, constrained, spin, boson and product spaces.

### A9 (FHS): cheaper `hash(::FermionSym)`
- **Change:** `hash(label, hash(basis, creation ? h ⊻ c : h))` replaces `hash(creation, hash(label, hash(basis, h)))`, saving one hashing round.
  - The constant is `% UInt`, so `h` stays a `UInt` on 32-bit Julia (reviewer finding).
- **Consistency:** the hash is a function of exactly the fields that `==` compares, so it stays consistent with `==`/`isequal`.
- **Reliance on hash values:** none. quality-control orders groups by names, and printing sorts by strings. Dict iteration order changes, so floating-point sums over terms may change in the last bits.
- **Microbenchmark:** in a loop, 13.3 → 9.9 ns per FermionSym hash.
- **Cached hash rejected:** after qc, the group hash costs only about 2.2 ns, and the basis hash (name plus group) about 5.5 ns. A cached basis hash would save about 3 ns more per factor, but needs a new struct field. Deferred as a design decision.
- **Test:** "Hash of fermions".
- **Decision: reject.**
  - Per commit, the end-to-end gain was 3–5% on deep_product, sum_square and sum, inside the noise and with no allocation change.
  - An earlier in-stream measurement gave 0.90 on deep_product, but the per-commit rerun did not reproduce it.
  - The commit is kept on `perf/s6-symbolic-a9-rejected`, with a 32-bit-safe constant (`% UInt`). It is ready if the maintainer wants it.

### Not implemented
- **A7:** `::Tuple{Vector,Bool,Int}` assertion. Not attempted, as instructed.
- **A7 `if effect isa …` split:** not attempted. The effect types are open, since user `mul_effect`s return arbitrary values.
- **A8:** copying factors for each new term. Not attempted. The copies are required because `splice!!` mutates the factor vectors. Phase 1 estimated 3–5%.
- **A10:** `with(_autosort => false)` on every sort. Rejected: FHS enables autosort globally, so the scope is always needed there.
- **mul_effect(::FermionSym, ::FermionSym) reordering:** rejected.
  - Testing `a < b` before `a == b` would raise a MethodError for `f[x]*f[x]` with labels that have no `<` (today the result is 0).
  - `(-1)^Bool` already compiles to the same code as `b ? -1 : 1` (210 ns for both on 1000 Bools).

## Cumulative effect against baseline-qc-env (FHS 1cee0550 + NCP main)
Min of 3 alternating runs of each side. The "NCP only" column uses FHS at 1cee0550 with NCP b0786c5.

| case | baseline µs | all µs | ratio | NCP only ratio | allocs: baseline → NCP only → all |
|---|---|---|---|---|---|
| symbolic/sum | 191.3 | 156.3 | **0.82** | 0.90 | 5717 → 5717 → 5717 |
| symbolic/sum_square | 854.9 | 523.1 | **0.61** | 0.62 | 10543 → 7948 → 7948 |
| symbolic/deep_product | 633.2 | 417.2 | **0.66** | 0.68 | 2454 → 2441 → 2441 |
| symbolic_states/projected_op_on_state | 190.5 | 165.4 | **0.87** | 0.93 | 7008 → 6927 → 3903 |
| real/hubbard12 | 74.8 | 60.7 | **0.81** | 0.96 | 3477 → 3165 → 3165 |
| real/heisenberg12 | 99.3 | 71.8 | **0.72** | 0.73 | 3020 → 2350 → 2350 |
| real/bdg_prod | 181.0 | 159.8 | **0.88** | 0.90 | 1746 → 1738 → 1738 |
| real/bdg_square | 417.1 | 307.0 | **0.74** | 0.64 | 4032 → 2959 → 2959 |
| real/krylov | 248.6 | 207.8 | **0.84** | 0.92 | 10413 → 10251 → 5279 |

Outputs:
- All cases are `==` the baseline (type, coefficient and terms) except real/krylov, which is `≈`. Its Dict iteration order changes with the sizehint, so the order of floating-point additions changes.
- The NCP-only and "all" columns are separate runs, so a difference smaller than about 10% between them is noise, for example hubbard12 0.96 against 0.81, where the allocations are equal.

## Tests
- **Targeted tests:** each commit's tests were run on that commit.
- **NCP full suite:** run on b0786c5 with NCP's `test/` env, via `TestItemRunner.run_tests(".")`: 10458/10458 pass.
- **FHS full suite:** run on 02e0ea71, with FHS and NCP from the worktrees (`pathof` was checked): 3820/3820 pass, in 14.5 min.

## Reviewer findings and resolution
One general-purpose reviewer read both diffs. Its findings and how each was handled:

1. **Bug.** Skipping the zero filter in `+(::NCAdd,::NCAdd)` let zero terms through when the promoted coefficient type is narrower, for example Rational + Float16. Two causes: `copy_dict` converts the terms of `a`, and `add_term!!` converted after its zero check, which was a baseline bug for `add!!` too.
   - **Fixed** by a new commit, "Convert a new coefficient to the value type before the zero check in add_term!!", placed before A2.
   - A4a now keeps zero filtering when `C1 !== promote_type(C1, C2)`.
   - Tests added for both.
2. **Perf/comment.** The fast `add_term!!` path fell back to the slow path for any value not already `isa V`, for example an Int added into a Float64 Dict.
   - **Fixed:** values that fit are converted, and the comment now matches the code.
   - heisenberg12 improved: the commit's own allocations went 3031→2350, where the earlier version gave 3020→2717.
3. **Risk.** The `@static` guard checked only that the names exist. **Added** a `hasmethod` check for the 5-argument `_setindex!`. A change in meaning is still caught only by the reference-comparison test.
4. **Portability.** `h ⊻ 0x5bd…` widened `h` to UInt64 on 32-bit Julia. **Fixed** with `% UInt`, but the commit was later rejected (A9).
5. **Style.** The underflow commit message described history. **Reworded** to state the current behaviour.
6. **Misleading comment.** The test comment said "every space type". **Reworded**, and the test was extended to a product space.
7. **Coverage.** Majorana, `FixedNumberFockState` and product-space overloads were not exercised.
   - A product space was added to the test.
   - The reviewer checked the other two by hand: results are identical to baseline.
   - Two baseline bugs found there are recorded below.
8. **Weak test.** The converted-key test compared Dicts without checking the key type, using a cancelling case. **Fixed:** the test now uses a non-cancelling case and checks `keytype`.
9. **Test hygiene.** The global autosort setting was restored without `try`/`finally`. **Fixed.**
10. **Perf nit.** `add!` now makes a second O(n) pass for every non-`One` β, also for exact types. **Accepted** for simplicity; the cost is one pass over the terms.

The reviewer found these clean: A1 (including nested scopes and grades), A3, the underflow fix, the remaining ways an NCAdd could hold zero values or scalar keys, the `add_term!!` Dict invariants (`slots`, `ndel`, `count`, `age`, `idxfloor`, rehash, key replacement), A6 and A5 results against the baseline, and the style rules.

## Latent issues found (not fixed, outside scope)
- **Majorana bras:** `ket' * γ[1]` gives `∓im` phases where `vᵀM` is real. The `transpose=true` path of `apply_local_operators(::NCMul, ::FockNumber, ::MajoranaHilbertSpace, …)` (`majoranas.jl` ~282) does not conjugate or negate correctly. Kets are correct. Present on the baseline.
- **Single-particle (`FixedNumberFockState`) spaces:** `group_id` is missing for `SingleParticleHilbertSpace`, so operators can't act on its symbolic kets. On a constrained `FixedNumberFockState{1}` space, `f[1]*ket` yields a state outside the space. Present on the baseline.
