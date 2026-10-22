# Phase 2, stream s5-gen-reshape

Branch `perf/s5-gen-reshape`, based on `quality-control` at 1cee0550. Assigned items: B04, B25, B13 (D3/D4/D5), the `catenate_fock_states` type bug, and G-4 (forwarded from the gap profiler).

## Commits and per-commit gains

Each commit is measured against its parent. Every row is the min of the minimum times over 3 alternating A/B processes (A = parent, B = commit), plus allocations. Outputs were compared by hash in every run, and all were identical. Noise on this machine is about ±14% per leaf, so a change under 10% counts only when allocations confirm it. The first commit is measured against baseline-qc-env (1cee0550).

| Commit | Benchmark | Parent min | Commit min | Ratio | Allocs parent → commit |
|---|---|---|---|---|---|
| **Keep running sums for additive branch constraints in generate_states** (B04) | generate_states/int (N=60, suite case) | 10.64 ms | 1.77 ms | **0.17** | 5570 → 5604 |
| | generate_states N=64 (`UInt64` processor) | 13.19 ms | 2.34 ms | **0.18** | 6664 → 6698 |
| | `hilbert_space(f, 1:60, prod of 3 weighted NumberConservation)` | 12.94 ms | 4.07 ms | **0.31** | 52983 → 53009 |
| | same, N=64 (BigInt) | 71.4 ms | 59.8 ms | 0.84 (noise; sector grouping dominates) | 1505725 → 1505751 |
| | spin-½ Sz=0, N=16, `AdditiveConstraint` | 24.7 ms | 24.2 ms | 0.98 (not on this path's hot spot) | 272995 → 273018 |
| **Skip the subspace position lookup for constraints on all spaces** (B25) | `branch_constraint` setup, N=60 | 0.134 ms | 0.100 ms | 0.75 | 5118 → 4391 |
| | generate_states/int | 1.98 ms | 1.90 ms | 0.96 | 5604 → 4877 |
| | setup of 20 per-site constraints (named subspaces) | 0.347 ms | 0.329 ms | 0.95 (not affected) | 4218 → 4218 |
| **Fill product sector labels with a typed loop** (G-4 b) | Hubbard `tensor_product`, Nh=10 (3,2) | 20.4 ms | 1.32 ms | **0.065** | 82404 → 23031 |
| | Hubbard `tensor_product`, Nh=12 (3,2) | 70.5 ms | 3.82 ms | **0.054** | 219412 → 59719 |
| | `constrain_space` no double occupancy, Nh=12 | 90.7 ms | 9.76 ms | **0.11** | 389998 → 172917 |
| | same, Nh=30 (2,1) | 103.7 ms | 31.1 ms | **0.30** | 594734 → 436380 |
| | `hilbert_space(f, 1:12, NumberConservation()*ParityConservation())` | 15.8 ms | 1.65 ms | **0.10** | 62791 → 17764 |
| **Give fermionic state mappers the mode count of the full space** (G-4 c) | Hubbard `tensor_product`, Nh=12 | 3.77 ms | 2.90 ms | 0.77 | 59719 → 30603 |
| | `constrain_space` no double occupancy, Nh=12 | 9.35 ms | 2.18 ms | **0.23** | 172917 → 36189 |
| | same, Nh=30 (2,1) | 41.7 ms | 6.65 ms | **0.16** | 436380 → 56971 |
| | sector 4,4 on 8 interleaved sites | 1.76 ms | 1.55 ms | 0.88 | 21130 → 11282 |
| **Look up fermionic subspace modes without building single-mode spaces** (B13 D5) | reshape/mat_to_tens | 38.5 µs | 34.4 µs | 0.89 | 502 → 326 |
| | reshape/mixed_split_combine | 42.7 µs | 40.2 µs | 0.94 | 502 → 326 |
| | split of a 256×256 matrix on number-conserving spaces | 513 µs | 472 µs | 0.92 | 911 → 679 |
| **Use typed transition tables and linear offsets in reshape** (B13 D3+D4) | reshape/mat_to_tens | 38.6 µs | 23.0 µs | **0.60** | 326 → 155 |
| | reshape/mat_to_vec | 81.9 µs | 72.3 µs | 0.88 | 4192 → 91 |
| | reshape/mixed_split_combine | 39.6 µs | 24.7 µs | **0.62** | 326 → 155 |
| | split of a 256×256 matrix on number-conserving spaces | 499 µs | 358 µs | **0.72** | 679 → 157 |
| | split of a 1024×1024 matrix on parity spaces | 5.08 ms | 3.67 ms | **0.72** | 2219 → 161 |
| | many-to-many `(N1,N2) => (N2,N1)` on a 16⁴ tensor | 723 µs | 566 µs | 0.78 | 2991 → 2469 |
| **Return the requested type from catenate_fock_states** (bug fix) | generate_states/int, N=64 `UInt64`, public N=60/64 | — | — | 0.94–1.07, no gain claimed | identical |

Cumulative effect of the stream against 1cee0550 (the product of the per-commit ratios): generate_states/int about 0.16×; Hubbard sector `tensor_product` Nh=12 about 0.04×; `constrain_space` with per-site constraints about 0.025× at Nh=12 and 0.05× at Nh=30; reshape benchmarks 0.53–0.88×, with mat_to_vec allocations 4364 → 91.

**Many-constraint compile time.** First call of `constrain_space` at Nh=30 (30 per-site constraints) in a fresh process: 1.2–1.4 s on 1cee0550 + B04/B25, 1.0–1.5 s after G-4 b, and 0.40–0.50 s after G-4 c. The tuple-based variant (G-4 a) that the gap profiler tried was not used, because its compile time grows with the number of constraints.

**TTFX note.** `hilbert_space(f, 1:4, NumberConservation()*ParityConservation())` as a first call compiles in about 6.8 s against 5.7 s at the base. That is within noise for single first-call timings, and was not investigated further.

## Per-item analysis and decisions

### B04: running sums in `generate_states` (implemented)
- **Design.** There were two candidates: a stateful closure (the Phase 1 prototype) and an explicit per-call state owned by `generate_states`. I chose the explicit state; it costs about 20 more lines:
  - `branch_state(constraint, spaces)` creates a scratch object once per `generate_states` call. It is `nothing` by default, and a vector of child states for `ProductConstraint`.
  - `backtrack!` passes it to `valid_branch(constraint, state, partial, depth, spaces)`.
  - The additive constraint is now an immutable `AdditiveBranchConstraint` struct. Its state is a prefix-sum vector: evaluating at depth `d` reads `prefix[d]`, adds the contributions at position `d`, and writes `prefix[d+1]`.
- **Why it is correct.** `backtrack!` evaluates depth `d` only when every shallower depth on the current branch has been accepted, and that acceptance wrote `prefix[d]`. In a `ProductConstraint`, `all` short-circuits, so a later child can keep a stale `prefix[d+1]`. That stale value is harmless: the branch is rejected, and the next evaluation at depth `d` overwrites it.
- **Advantages over the stateful closure.**
  - The constraint object can be reused and shared across threads, since each call allocates its own state.
  - User `BranchConstraint`s and user subtypes of `AbstractBranchConstraint` keep the four-argument `valid_branch`, through the `::Nothing` fallback.
- **Covered cases:**
  - Constraints on subsets of spaces, in any order (positions ≠ 1:n). The subspaces at each position are stored as sorted index ranges built by counting, which is O(n).
  - A single shared map, per-subspace maps, and `WeightedFunction` weights.
  - Spin (HalfInteger), boson and fermion contributions.
- **Accumulator type.** It is `promote_type(T0, typeof(sum(first, contribution_values)))`, where `T0` is the promoted type of all contribution values. Like `sum`, this widens `Bool` and small integer types.
- **Floating point.** With Float maps the running sum adds contributions in generation order rather than in the listed order of the subspaces. Membership tests at an allowed-value boundary can therefore change, which was also possible with the old summation.
- **Removed internals.** The four-argument `valid_branch(::ProductConstraint, …)` and `_additive_function_application` had no other callers and were removed. The `valid_branch` docstring now documents both forms.
- **Tests.** Pruned generation is checked against brute-force filtering on mixed fermion, spin and boson spaces. The cases cover subset constraints in non-increasing order, weighted, shared-map, Bool and Int8 maps, a user `BranchConstraint`, a user subtype of `AbstractBranchConstraint` alone and in a product, and products of all of these.

### B25: `generate_states` setup (implemented in a reduced form)
- The proposed id → position `Dict` was **slower** than the existing `findfirst`: setup went from 0.13 to 0.15 ms at N=60. Hashing `FermionSym` keys costs more than comparing one-element vectors at these sizes, so I rejected the Dict.
- What was implemented: when a constraint names no subspaces, its positions are `1:n` and the lookup is skipped. `NumberConservation` and `ParityConservation` without spaces now pass `missing` down to take this path.
- The remaining setup cost is mostly `hilbert_space(::FermionSym)` per mode and `unique` over each mode's basis states, about 2% of generate_states/int. I left it.

### G-4: `ProductConstraint` sector function (implemented, b and c)
- **(b)** The labels are written into a `Vector{V}`. `V` is inferred with `promote_op` from the sector functions and `statetype(space)`, and the loop returns `missing` at the first missing label. If `V` is not concrete, the vector is narrowed with `map(identity, …)`, as before.
  - Label vectors keep their element type (`Vector{Int}` for number/parity products). A test checks this, and also checks a mixed Int/Symbol label product.
  - Caveat from review: if a sector function were ever called with a state type other than `statetype(space)`, an inferred `V` could be too narrow. No such caller exists.
- **(c)** `FockMapper` gets the mode count of the full space as its `N` parameter (`FockMapper(positions, N)`; the one-argument form keeps the old maximum-position behaviour). The bit permutation is only built when the subsystems cover all `N` modes, so per-site mappers of one space have identical types.
  - Behaviour change: for a full partition of the first modes of a ≥64-mode space, `combine_states` now returns `FockNumber{BigInt}` (the parent's representation) instead of `FockNumber{UInt64}`. A test covers this.
  - `isfullpartition` is still computed against the largest position.

### B13: reshape (D3, D4, D5 implemented)
- **D3.** Each `_transitions` method builds its entries as a flattened generator with `Int` indices and collects it into a concretely typed vector. One pass then records the dropped input indices, and the filtered copy is made only when something was dropped. This keeps the `(; transitions, dropped)` return value of `quality-control` and the `skipmissing`/`_check_dropped` semantics.
  - mat_to_vec memory rises by about 10% (490 → 531 KiB), because `collect` over a flattened iterator over-allocates. A size hint would need inference-dependent code, so I left it.
- **D4.** The kernel converts the index tuples of each block into zero-based linear offsets, and indexes `t` and `tout` linearly. It does this only when both arrays have `IndexLinear` and no offset axes; other arrays take the Cartesian loop. The summation order is unchanged, so results are bitwise identical.
  - A test runs `PermutedDimsArray` inputs and outputs through `reshape` and `ReshapeMap`.
  - Offset arrays are still indexed from 1 in the Cartesian path; that behaviour is unchanged.
- **D5.** `_fock_mapper` passes the modes of a fermionic subspace (also inside `ConstrainedSpace` and `SectorHilbertSpace`) directly to `atom_position`, instead of building one single-mode `FermionicSpace` per mode.
  - Majorana and product subspaces still go through `atomic_factors`.
  - A zero-mode space maps to `(Hsub,)`, as before.

### `catenate_fock_states` ignores `T` (fixed)
- `num |= state << shift` promoted `FockNumber{Int}` and `FockNumber{UInt64}` to `UInt64`, so the result type was a promotion rather than `T`. The function now converts each state to `T`, and throws an `ArgumentError` when the spaces have more modes than `T` holds (63 for `Int`, 64 for `UInt64`, unbounded for `BigInt`). A regression test covers this.
- **Consequence:** the `generate_states/big_int` leaf in `quality-control`'s `benchmark/benchmarks_v0.10.jl` uses `CombineFockNumbersProcessor{FockNumber{Int}}` with 64 modes, and now throws. The baseline only gave correct states there because of the accidental promotion. The s7 suite no longer has that leaf (H06).

## Not done / deferred
- B25's per-mode `hilbert_space(::FermionSym)` and `unique` setup cost (about 2%).
- The remaining cost of `hilbert_space(f, 1:64, …)` (60–75 ms) is BigInt `pext`/`pdep` in the sector grouping and in `combine_states`. A zero-skip in `combine_states(::FockMapper)` and a single-bit fast path in `substate` look cheap, but are untested.
- G-6 (`tensor_product` re-evaluates the combined sector function) is outside this stream's files.
- The Majorana `_fock_mapper` still builds Majorana atoms.

## Tests
- Targeted test items were run after each change: generate_constrained_states, constraints, sectors, constrained_space, fock, fermions, majoranas, reshape and tensor_product.
- Full suite before the final fix-up: 3700/3701 passed. The one failure was ExplicitImports flagging `Base.add_sum`, `Base.BitInteger` and `Base.size_to_strides` as non-public qualified accesses.
  - Those three were replaced by public equivalents in their commits: the `sum`-based promotion, an `isbitstype` check, and an `accumulate(*, …)` of the sizes. The amended commits contain the same computation, so the per-commit timings above still apply.
  - After the replacement, ExplicitImports, Aqua and the generate_constrained_states, fock and reshape items were re-run, and all 301 passed.

## Reviewer findings and resolution
A general-purpose reviewer subagent reviewed `git diff 1cee0550..HEAD`.
1. **Narrow accumulator type** (Bool maps threw `InexactError`, Int8 maps overflowed). Fixed in the B04 commit with the `add_sum` promotion; Bool and Int8 tests added.
2. **User `AbstractBranchConstraint` subtypes with a four-argument `valid_branch` raised a MethodError.** Fixed in the B04 commit: the fallback is defined for `AbstractBranchConstraint`, the additive method takes `prefix::AbstractVector` to avoid an ambiguity (Aqua passes), and a test was added. The docstring now covers both forms.
3. **The B25 shortcut never applied to `NumberConservation`/`ParityConservation`.** Fixed in the B25 commit by passing `missing`.
4. **`FockMapper` comment and `N::Int` annotation.** The comment was reworded to state the rule, and `N` is now untyped. The visible `combine_states` return-type change is documented above.
5. **Floating-point order and the inference of `V`.** Accepted and documented above.
6. **Reshape, `catenate_fock_states`, `_fock_atoms`.** The reviewer found no problems.

The history was rebuilt so that each fix sits in the commit it belongs to; there are no fixup commits.
