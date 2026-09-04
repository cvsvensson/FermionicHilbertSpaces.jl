using BenchmarkTools
using FermionicHilbertSpaces
using Random

Random.seed!(42)

# ==============================================================================
# Benchmark Interpretation & Usage Guidelines
# (Benchmarked on 12 fermionic modes with N=3 particles tracing to 4 modes.
# ==============================================================================
#
# 1. Repeated calls (e.g. time-evolution loops, iterative solvers):
#    - Always precompute maps upfront (e.g. `pt = partial_trace(H => Hsub)`,
#      `emb = embed(Hsub => H)`).
#    - Precomputing incurs an initial construction cost (~0.5 ms for this size),
#      but caches index lookups and eliminates allocations.
#    - Calling the precomputed map is significantly faster (3x–8x) than the eager
#      equivalent and amortizes the upfront cost within just a few iterations.
#
# 2. One-off operations:
#    - Prefer the eager form (e.g. `partial_trace(::AbstractMatrix, H => Hsub)`).
#    - Rebuilding a map for a single evaluation is counterproductive because map
#      construction takes considerably longer than a single eager evaluation.
#
# 3. In-place evaluation:
#    - Prefer in-place overloads (`map!(out, input)`) when the destination buffer
#      can be preallocated and reused.
#    - They run at the same speed (or slightly faster) as the allocating precomputed
#      forms, but reduce temporary allocations to zero (or negligible metadata),
#      avoiding garbage collection pressure in tight loops.
#
# 4. Density matrix input formats (Matrix vs. Vectorized):
#    - Matrix input:      `pt(::AbstractMatrix)` of shape `(dim(H), dim(H))`
#    - Vectorized input:  `pt(::AbstractVector)` of length `dim(H)^2`
#    - Both formats perform almost identically (~71 µs in this benchmark).
#      Choose whichever representation your data already uses to avoid reshapes.
#    - In-place transformations across shapes are also supported and allocation-free:
#        * `pt(out::Matrix, in::Matrix)`  — matrix to matrix
#        * `pt(out::Vector, in::Vector)`  — vectorized to vectorized
#        * `pt(out::Vector, in::Matrix)`  — matrix to vectorized buffer
#
# 5. Pure-state vector caveat:
#    - Tracing a pure-state vector `pt(::AbstractVector)` of length `dim(H)`:
#        * Under the hood, this allocates a temporary full density matrix of
#          size `dim(H) × dim(H)` (~785 KB here), making it noticeably slower
#          than tracing an existing density matrix.
#        * The in-place overload `pt(out::Vector, pure_state::Vector)` exhibited
#          a severe performance regression in this benchmark (~1 ms, roughly 3x
#          slower than the allocating version and 5x slower than eager matrix trace).
#    - Recommendation: Avoid the in-place pure-state overload in performance-
#      critical loops. Prefer `pt(pure_state)` (out-of-place), or explicitly form
#      and track a density matrix if repeated partial traces are needed.
# ==============================================================================


function report(name, trial)
    t = run(trial)
    m = minimum(t)
    println(
        rpad(name, 40),
        " min = ", round(m.time / 1_000; digits=2), " us",
        " | allocs = ", m.allocs,
        " | bytes = ", m.memory,
    )
end

function bench_partial_trace()
    println("\n== partial_trace: eager vs precomputed ==")
    @fermions f
    H = hilbert_space(f, 1:12, NumberConservation(3))
    Hsub = subregion(hilbert_space(f, 1:4), H)
    m = rand(ComplexF64, dim(H), dim(H))
    v_dm = vec(m)
    v_pure = rand(ComplexF64, dim(H))

    pt = partial_trace(H => Hsub)
    out_m = zeros(ComplexF64, dim(Hsub), dim(Hsub))
    out_v = zeros(ComplexF64, dim(Hsub)^2)

    report("construct map", @benchmarkable partial_trace($(H => Hsub)))
    report("eager matrix call", @benchmarkable partial_trace($m, $(H => Hsub)))
    report("precomputed matrix call", @benchmarkable $pt($m))
    report("in-place matrix->matrix", @benchmarkable $pt($out_m, $m))
    report("precomputed vectorized dm", @benchmarkable $pt($v_dm))
    report("in-place vector dm->vector", @benchmarkable $pt($out_v, $v_dm))
    report("precomputed pure-state vec", @benchmarkable $pt($v_pure))
    report("in-place pure vec->vector", @benchmarkable $pt($out_v, $v_pure))
    report("in-place matrix->vector", @benchmarkable $pt($out_v, $m))
end

function bench_embed()
    println("\n== embed: eager vs precomputed ==")
    @fermions f
    H = hilbert_space(f, 1:12, NumberConservation(3))
    Hsub = subregion(hilbert_space(f, 1:4), H)
    msub = rand(ComplexF64, dim(Hsub), dim(Hsub))

    emb = embed(Hsub => H; skipmissing=true)
    out = zeros(ComplexF64, dim(H), dim(H))

    report("construct map", @benchmarkable embed($(Hsub => H); skipmissing=true))
    report("eager call", @benchmarkable embed($msub, $(Hsub => H); skipmissing=true))
    report("precomputed apply", @benchmarkable $emb($msub))
    report("in-place apply", @benchmarkable $emb($out, $msub))
end

function bench_reshape()
    println("\n== reshape: eager vs precomputed ==")
    @fermions f
    H1 = hilbert_space(f, 1:2, NoSymmetry())
    H2 = hilbert_space(f, 3:4, NoSymmetry())
    H = tensor_product(H1, H2)
    m = rand(ComplexF64, dim(H), dim(H))

    rmap = reshape(H => (H1, H2))
    out = zeros(ComplexF64, dim(H1), dim(H2), dim(H1), dim(H2))

    report("construct map", @benchmarkable reshape($(H => (H1, H2))))
    report("eager call", @benchmarkable reshape($m, $(H => (H1, H2))))
    report("precomputed apply", @benchmarkable $rmap($m))
    report("in-place apply", @benchmarkable $rmap($out, $m))

    A3 = rand(ComplexF64, dim(H), dim(H), dim(H))
    rmap_repeat = reshape(H => (H1, H2); repeat=true)
    out_repeat = zeros(ComplexF64, dim(H1), dim(H2), dim(H1), dim(H2), dim(H1), dim(H2))
    report("eager call (repeat=true)", @benchmarkable reshape($A3, $(H => (H1, H2)); repeat=true))
    report("precomputed apply repeat", @benchmarkable $rmap_repeat($A3))
    report("in-place apply repeat", @benchmarkable $rmap_repeat($out_repeat, $A3))
end

function main()
    println("Precomputed vs eager microbenchmarks")
    bench_partial_trace()
    bench_embed()
    bench_reshape()
end

main()
