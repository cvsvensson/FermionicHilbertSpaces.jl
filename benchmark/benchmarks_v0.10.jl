using BenchmarkTools
using FermionicHilbertSpaces
using LinearAlgebra
using SparseArrays
using Random
using Combinatorics # loads CombinatoricsExt, which symmetric_sector uses

const FHS = FermionicHilbertSpaces
const SUITE = BenchmarkGroup()

# Benchmarks created below run for 1 second unless they set `seconds`
# themselves. The global default is restored at the end of this file.
const DEFAULT_SECONDS = BenchmarkTools.DEFAULT_PARAMETERS.seconds
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

# Each section draws its random inputs from its own seeded RNG, so adding or
# removing a section does not change the inputs of the others.
#
# Benchmarks of partial traces, embeddings and partial transposes pass
# `complement=` so that the complementary subsystem, which is benchmarked on
# its own under "complement", is not recomputed in every sample.

## Fermions: spaces, symbolic algebra, matrix representations
let rng = Xoshiro(1)
    @fermions f
    N = 12
    H = hilbert_space(f, 1:N, ParityConservation())
    op = sum(rand(rng) * f[n]' * f[n] for n in 1:N) + sum(1im * f[n]' * f[n+1] + hc for n in 1:(N-1))
    SUITE["hilbert_space"] = @benchmarkable hilbert_space($f, $(1:N), $(ParityConservation()))
    SUITE["symbolic"]["sum"] = @benchmarkable sum($f[n]' * $f[n] + hc for n in 1:100)
    SUITE["symbolic"]["sum_square"] = @benchmarkable sum($f[n]' * $f[n] + hc for n in 1:50)^2
    labels = shuffle(rng, 1:10)
    SUITE["symbolic"]["deep_product"] = @benchmarkable prod($f[l] for l in $labels) * prod($f[l]' for l in $labels)
    SUITE["matrix_representation"]["standard"] = @benchmarkable matrix_representation($op, $H)

    Hsp = hilbert_space(f, 1:1000, SingleParticle())
    opsp = sum(rand(rng) * f[n]' * f[n] for n in 1:1000) + sum(rand(rng, ComplexF64) * f[n]' * f[n+1] + hc for n in 1:999)
    SUITE["matrix_representation"]["free_fermion"] = @benchmarkable matrix_representation($opsp, $Hsp)

    opsp_bdg = opsp + sum(rand(rng, ComplexF64) * f[n]' * f[n+1]' + hc for n in 1:999)
    Hbdg = hilbert_space(f, 1:1000, BdG())
    SUITE["matrix_representation"]["bdg"] = @benchmarkable matrix_representation($opsp_bdg, $Hbdg)
end

## Fermions: partial trace and embedding in a parity-conserving space
let rng = Xoshiro(2)
    @fermions f
    N = 12
    H = hilbert_space(f, 1:N, ParityConservation())
    Hsub = hilbert_space(f, 1:div(N, 4), ParityConservation())
    SUITE["complement"]["fermions"] = @benchmarkable FHS.complementary_subsystem($H, $Hsub)
    Hcomp = FHS.complementary_subsystem(H, Hsub)
    SUITE["partial_trace"]["fermions"]["map"] = @benchmarkable partial_trace($(H => Hsub); complement=$Hcomp) seconds = 3
    ψ = normalize(rand(rng, ComplexF64, dim(H)))
    SUITE["partial_trace"]["fermions"]["pure_state"] = @benchmarkable partial_trace($ψ, $(H => Hsub); complement=$Hcomp)
    msub = rand(rng, ComplexF64, dim(Hsub), dim(Hsub))
    SUITE["embed"]["fermions"] = @benchmarkable embed($msub, $(Hsub => H); complement=$Hcomp)
end

## Reshape
let rng = Xoshiro(3)
    @fermions f
    H1 = hilbert_space(f, 1:3, NoSymmetry())
    H2 = hilbert_space(f, 4:5, NoSymmetry())
    H3 = hilbert_space(f, 6:8, NoSymmetry())
    H4 = hilbert_space(f, 9:12, NoSymmetry())
    H12 = tensor_product(H1, H2)
    H34 = tensor_product(H3, H4)
    H = tensor_product(H12, H34)
    M = rand(rng, ComplexF64, dim(H12), dim(H34))
    SUITE["reshape"]["mat_to_tens"] = @benchmarkable reshape($M, $H12 => ($H1, $H2), $H34 => ($H3, $H4))
    SUITE["reshape"]["mat_to_vec"] = @benchmarkable reshape($M, ($H12, $H34) => $H)
    A = rand(rng, ComplexF64, dim(H12), dim(H12), dim(H12))
    SUITE["reshape"]["repeat"] = @benchmarkable reshape($A, $(H12 => (H1, H2)); repeat=true)
end

## Constrained state generation, through the public constructor. With 64 or
## more modes the Fock states are stored as BigInt.
for (N, key) in ((60, "hilbert_space_N60"), (64, "hilbert_space_N64_bigint"))
    @fermions f
    weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
    allowed_ones = [[0, 1], [-1, 0], [2]]
    constraint = prod(NumberConservation(allowed, missing, w) for (allowed, w) in zip(allowed_ones, weights))
    SUITE["generate_states"][key] = @benchmarkable hilbert_space($f, $(1:N), $constraint) seconds = 3
end

## Partial trace algorithms
# Each input is traced with both algorithms forced through `alg`, and with
# the algorithm that `partial_trace` picks by default.
let rng = Xoshiro(4)
    @fermions f
    algs = ("SubAlg" => FHS.SubsystemPartialTraceAlg(), "FullAlg" => FHS.FullPartialTraceAlg())
    function add_cases!(g, m, H, Hsub)
        Hcomp = FHS.complementary_subsystem(H, Hsub)
        for (name, alg) in algs
            g[name] = @benchmarkable partial_trace($m, $H, $Hsub; complement=$Hcomp, alg=$alg)
        end
        g["default"] = @benchmarkable partial_trace($m, $H, $Hsub; complement=$Hcomp)
    end
    g = SUITE["partial_trace_algorithms"]

    # A number-conserving space and a large subsystem, where FullAlg is expected to be faster.
    H = hilbert_space(f, 1:30, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, 1:10), H)
    d = dim(H)
    add_cases!(g["Sparse space"]["Dense"], rand(rng, ComplexF64, d, d), H, Hsub)
    add_cases!(g["Sparse space"]["Sparse"], sprand(rng, ComplexF64, d, d, 0.01), H, Hsub)

    # An unconstrained space and a small subsystem, where SubAlg is expected to be faster.
    H = hilbert_space(f, 1:10)
    Hsub = subregion(hilbert_space(f, 1:2), H)
    d = dim(H)
    add_cases!(g["Full space"]["Dense"], rand(rng, ComplexF64, d, d), H, Hsub)
    add_cases!(g["Full space"]["Sparse"], sprand(rng, ComplexF64, d, d, 0.01), H, Hsub)
end

## Product spaces
let rng = Xoshiro(5)
    @fermions f
    @bosons b
    Hf = hilbert_space(f, 1:2)
    Hb = hilbert_space(b, 1:4, 2)
    H = tensor_product(Hf, Hb; constraint=NumberConservation())
    Hsub = tensor_product(hilbert_space(f, 1:1), [hilbert_space(b[n], 2) for n in 1:2]...)
    SUITE["complement"]["product space"] = @benchmarkable FHS.complementary_subsystem($H, $Hsub)

    op = (1 + sum(rand(rng) * f[n]'f[n] for n in 1:2)) * (1 + sum(rand(rng) * b[n]'b[n] + b[n+1]'b[n] + hc for n in 1:3))
    SUITE["matrix_representation"]["product space"] = @benchmarkable matrix_representation($op, $H)

    Hcomp = FHS.complementary_subsystem(H, Hsub)
    SUITE["partial_trace"]["product space"]["map"] = @benchmarkable partial_trace($(H => Hsub); complement=$Hcomp)
    m = rand(rng, ComplexF64, dim(H), dim(H))
    pt = partial_trace(H => Hsub; complement=Hcomp)
    SUITE["partial_trace"]["product space"]["apply_precomputed"] = @benchmarkable $pt($m) evals = 10
    msub = rand(rng, ComplexF64, dim(Hsub), dim(Hsub))
    SUITE["embed"]["product space"] = @benchmarkable embed($msub, $(Hsub => H); complement=$Hcomp)
end

## Product spaces with many atomic factors
let
    @fermions f
    @bosons b
    Hf = hilbert_space(f, 1:2)
    Hb = hilbert_space(b, 1:10, 2)
    Hprod = tensor_product(Hf, Hb)
    op_prod = sum((0.2n) * f[n]' * f[n] for n in 1:2)
    op_prod += 0.1 * (f[2]' * f[1] + hc)
    op_prod += sum((0.05n) * b[n]' * b[n] for n in 1:10)
    op_prod += sum(0.03 * (b[n+1]' * b[n] + hc) for n in 1:9)
    SUITE["matrix_representation"]["product space many spaces"] = @benchmarkable matrix_representation($op_prod, $Hprod)

    Hb_cons = hilbert_space(b, 1:18, 2, NumberConservation(0:1))
    Hprod_cons = tensor_product(Hf, Hb_cons)
    op_prod_cons = sum((0.15n) * f[n]' * f[n] for n in 1:2)
    op_prod_cons += 0.07 * (f[2]' * f[1] + hc)
    op_prod_cons += sum((0.04n) * b[n]' * b[n] for n in 1:18)
    op_prod_cons += sum(0.02 * (b[n+1]' * b[n] + hc) for n in 1:17)
    SUITE["matrix_representation"]["product space many spaces constrained"] = @benchmarkable matrix_representation($op_prod_cons, $Hprod_cons)
end

## Open systems: the Liouvillian as a lazy operator
let rng = Xoshiro(6)
    @fermions c
    N = 6
    Hopen, _, _, left, right = FHS.open_system(c, 1:N)
    ham = sum((0.3n) * c[n]' * c[n] for n in 1:N) + sum(0.2im * c[n]' * c[n+1] + hc for n in 1:(N-1))
    dissipator(L) = left(L) * right(L') - 0.5 * (left(L' * L) + right(L' * L))
    jump_in = [sqrt(0.05 + 0.01n) * c[n]' for n in 1:N]
    jump_out = [sqrt(0.04 + 0.01n) * c[n] for n in 1:N]
    liouvillian = 1im * (left(ham) - right(ham)) + sum(dissipator(L) for L in jump_in) + sum(dissipator(L) for L in jump_out)
    L = representation(liouvillian, Hopen, :lazy)
    v = randn(rng, ComplexF64, dim(Hopen))
    vout = similar(v)
    L_cached = FHS.SciMLOperators.cache_operator(L, v)
    g = SUITE["matrix_application"]["lazy liouvillian"]
    g["mul!"] = @benchmarkable mul!($vout, $L_cached, $v)
    # The uncached operator applied to a vector with only a few nonzeros.
    # This mostly measures the per-call overhead of the lazy operator.
    vs = sprand(rng, ComplexF64, dim(Hopen), 0.001)
    g["mul sparse"] = @benchmarkable $L * $vs
end

## Permutation symmetry
let
    @spins S 1 // 2
    H = hilbert_space(S, 1:6, FHS.AdditiveConstraint(2; maps=s -> s.m))
    Hs = factors(H)
    SUITE["permutations"]["many_small"] = @benchmarkable symmetric_sector($H, $Hs, :symmetric)
end
let
    @spins S 3
    H = hilbert_space(S, 1:4)
    Hs = factors(H)
    SUITE["permutations"]["few_large"] = @benchmarkable symmetric_sector($H, $Hs, :symmetric)
end

## Symbolic states
let
    # Repeated application of a projected operator, as in Krylov solves.
    # Projecting with Q = I - Σ|b⟩⟨b| produces zero-coefficient terms; if
    # they are kept, they pile up.
    @fermions f c
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(c, 1:6))
    kets = Kets(H)
    pert = sum(0.1 * f[1]' * c[n] + hc for n in 1:6)
    Q = I - sum(b * b' for b in kets.(["0", "10", "11", "01"]))
    Hq = Q * (f[1]' * f[2] + hc + 10.0 * sum(c[n]'c[n] for n in 1:6) + pert) * Q
    ψ = Q * pert * kets("10")
    SUITE["symbolic_states"]["projected_op_on_state"] = @benchmarkable $Hq * ($Hq * ($Hq * $ψ))
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = DEFAULT_SECONDS

# Fixing `evals` makes `tune!` unnecessary. One evaluation per sample is
# accurate for every benchmark that takes more than a few microseconds; the
# faster ones set `evals` themselves.
for (_, b) in BenchmarkTools.leaves(SUITE)
    b.params.evals_set && continue
    b.params.evals = 1
    b.params.evals_set = true
end
