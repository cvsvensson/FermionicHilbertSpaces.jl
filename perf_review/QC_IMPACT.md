# Rebasing the performance work onto `quality-control`

The performance review now measures against the branch `quality-control`, head **725b5f6b**. Its merge-base with the profiled commit is 660c67f4. Four code-reading reviews compared the 660c67f4..725b5f6b diff with every bottleneck in PHASE1_BOTTLENECKS.md. No benchmarks were run for that comparison.

## New baseline
- **Worktree:** `C:\Users\svens\Projects\FHS-perf-wt\baseline-qc`, detached at 725b5f6b.
- **Env:** `...\scratchpad\phase2\baseline-qc-env`. It is s7's `benchmark/Project.toml` plus NonCommutativeProducts, with `[sources]` pointing at baseline-qc and at the main NCP checkout.
- **Do not use `phase1\benchenv` any more.** It devs the main checkout, which the user has switched to `quality-control` and may change further.
  - Any A/B run made with benchenv after about 12:00 today used a moving baseline.
  - Re-measure every fix you claim against baseline-qc-env.
- **Full-suite baseline** with the s7 suite on 725b5f6b, run on a quiet machine: `...\scratchpad\phase2\results\qc_run1.json`.

## Overall verdict
`quality-control` does not materially change the bottleneck code paths. All bottlenecks still exist. The exceptions:
- **Symbolic benchmarks will probably shift, likely slower.**
  - d44370ae and 7286706b give `@fermions` and `@majoranas` the group id `Tags((xs,))` (a nested tuple of Symbols) instead of a hashed `UInt`.
  - `hash(::FermionSym)` therefore hashes nested tuples.
  - `_group_isless` compares Symbols.
  - `isless(::SymbolicState)` is now a lexicographic chain instead of a hash comparison.
  - This makes B09 A9 (a cheaper `FermionSym` hash, for example a precomputed group hash) **more valuable**.
- **B24 is slightly worse.** The free-fermion and BdG `matrix_representation` now also runs `_isfermionic(op)`, a full `ncmapreduce`, before `isquadratic`/`isnumberconserving` (`fixednumberfock.jl:308-314`, `bdg.jl:62-66`). Fold it into the fused single pass.
- **Path moves:** physics files now live in `src/physics/fermions/` (`bdg.jl`, `fixednumberfock.jl`, `fock.jl`, `fermions.jl`, `majoranas.jl`, `symbolic_fermions.jl`).

## Per stream: what to adapt when rebasing
- **s1-perm**
  - The commits (6bb7d12a, f41de0c9) merge cleanly.
  - The uncommitted B02/B21 work conflicts:
    - Keep qc's `method = _orth_method(orth_method)` handling of `:qr`/`:eig`/`nothing`, and its rank line `round(Int, real(tr(P))^2 / tr2)` (09633e07).
    - `_remove_columns!` was rewritten in qc.
  - B02 (the absolute 0.1 tolerance) still exists in qc on both the dense and the sparse path.
  - B01: keep the return contract for each `orth_method`, including the new `sparse` keyword.
  - The qc test `symmetric_sector(...; orth_method=:qr) == Ptriv` must still hold.
- **s2-ptrace**
  - B06 merges cleanly.
  - B05 has one cosmetic conflict in the SubAlg kernel's error messages. Use qc's wording: ``"The state `$fullf` is not in the full Hilbert space."``
  - B03 adaptation: qc moved the map-on-pure-state kernel into `_apply_superop_on_pure_state!(w, mat::SparseMatrixCSC, ψ)` in `partial_transpose.jl` (~199). That function is **shared with `PartialTransposeMap`**.
    - Add a method for the transposed layout, or keep CSC for partial transpose.
    - qc already removed the function-wide `@inbounds` from tensor_product.jl. The new kernel's `@inbounds` sits after size checks.
  - B11: the body was reordered and uses `@inbounds basisstate(J1, H)`; the fix still applies.
  - B28: `default_partial_trace_alg` now also accepts `UniformScaling` (c4e8e2f0). Cover that case.
  - B16, B17, B26, B27 and G-3 are unchanged.
- **s3-matrep**
  - Two trivial conflicts:
    - `partition_product`: keep s3's body and qc's message "Not all factors of the operator belong to a basis of the provided space."
    - `_apply_local_operators_fast`: qc changed the signature to `internal_reps::Tuple{InternalRep{T},Vararg{InternalRep{T}}}` with `where {C,T}`. Keep s3's body with qc's signature.
  - qc removed `_apply_local_operators(op::Missing, ...)`. Make sure nothing needs it.
  - **G-2 semantic conflict:** qc added typed `repr` (`EagerSparseRepr{T}`/`EagerDenseRepr{T}`, applied via `_repr_eltype(mat_eltype(op), repr)`).
    - s3's single-accumulator method must use `T = _repr_eltype(promote_type(...), repr)`.
    - Add a test: a typed `repr` on a constrained product space, for example `matrix_representation(op, Hc, ComplexF64)`.
- **s4-lazy**
  - One conflict, in the `import` line of `ext/OhMyThreadsExt.jl`. Take qc's line and remove `precomputation_before_operator_application` if the extension no longer uses it. The ExplicitImports test checks this.
  - Update the hard-coded error strings in the uncommitted sparse `*` path to qc's wording:
    - "The operator maps outside of the provided space. Pass `projection=true` to ignore those states."
    - "The input must have length N, got M"
  - qc made the `LazyOperator` fields `const`, except `cache`.
  - Re-check `adjoint`/`transpose` with the stored precomputation.
- **s5-gen-reshape**
  - The B04 commit and the uncommitted G-4 edit apply cleanly.
  - B13 adaptation: qc (1f5df425) made `_transitions` return `(; transitions, dropped)` and added `skipmissing`/`_check_dropped`.
    - D3 builds the typed vector inside each `_transitions` method.
    - D4 iterates `blocks[i].transitions`.
    - D5 goes into `_fock_mapper` in `src/physics/fermions/fermions.jl:97-101`; keep its new error message.
  - B25 and `catenate_fock_states` are unchanged.
- **s6-symbolic**
  - The uncommitted A6 edit no longer applies: fb2147f9 deleted `apply_local_operators(::NCMul{C,<:SymbolicState}, …)`. Re-anchor `_scaled_state` near `mul_effect(op::AbstractSym, s::SymbolicState)` (~214) and the `return amp * SymbolicState(...)` lines (~217 and ~230).
  - A5 is still open.
  - A9 is now more valuable (see above).
  - Re-measure the NCP fixes against the qc baseline, since the symbolic baseline has shifted.
- **s7-bench**
  - qc's own `benchmark/benchmarks_v0.10.jl` uses `hilbert_space(f, 1:1000, SingleParticle())` and `BdG()`. The old names are deprecated.
  - The suite should target the qc API from now on. It no longer needs to run on 660c67f4.
  - Merge qc's benchmark-file changes into s7's version.
