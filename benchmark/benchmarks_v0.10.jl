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
# `tensor_product` of matrices has no such keyword, so its benchmarks include
# computing the complement.

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

## Partial trace and embedding: eager calls and precomputed maps
let rng = Xoshiro(7)
    @fermions f
    H = hilbert_space(f, 1:12, NumberConservation(4))
    Hsub = subregion(hilbert_space(f, 1:4), H)
    Hcomp = FHS.complementary_subsystem(H, Hsub)
    ρ = rand(rng, ComplexF64, dim(H), dim(H))
    ψ = normalize(rand(rng, ComplexF64, dim(H)))
    pt = partial_trace(H => Hsub; complement=Hcomp)
    out_m = zeros(ComplexF64, dim(Hsub), dim(Hsub))
    out_v = zeros(ComplexF64, dim(Hsub)^2)
    g = SUITE["partial_trace"]["precomputed"]
    g["construct"] = @benchmarkable partial_trace($(H => Hsub); complement=$Hcomp)
    g["eager_matrix"] = @benchmarkable partial_trace($ρ, $(H => Hsub); complement=$Hcomp)
    g["apply_matrix"] = @benchmarkable $pt($ρ)
    g["apply_matrix_inplace"] = @benchmarkable $pt($out_m, $ρ)
    g["apply_vectorized_inplace"] = @benchmarkable $pt($out_v, $(vec(ρ)))
    g["eager_pure_state"] = @benchmarkable partial_trace($ψ, $(H => Hsub); complement=$Hcomp)
    g["apply_pure_state"] = @benchmarkable $pt($ψ)
    g["apply_pure_state_inplace"] = @benchmarkable $pt($out_v, $ψ)

    msub = rand(rng, ComplexF64, dim(Hsub), dim(Hsub))
    emb = embed(Hsub => H; complement=Hcomp, skipmissing=true)
    out = zeros(ComplexF64, dim(H), dim(H))
    g = SUITE["embed"]["precomputed"]
    g["construct"] = @benchmarkable embed($(Hsub => H); complement=$Hcomp, skipmissing=true)
    g["eager"] = @benchmarkable embed($msub, $(Hsub => H); complement=$Hcomp, skipmissing=true)
    g["apply"] = @benchmarkable $emb($msub)
    g["apply_inplace"] = @benchmarkable $emb($out, $msub)
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
    rmap = reshape(H12 => (H1, H2), H34 => (H3, H4))
    out = zeros(ComplexF64, dim(H1), dim(H2), dim(H3), dim(H4))
    SUITE["reshape"]["precomputed"]["construct"] = @benchmarkable reshape($(H12 => (H1, H2)), $(H34 => (H3, H4))) evals = 4
    SUITE["reshape"]["precomputed"]["apply"] = @benchmarkable $rmap($M) evals = 4
    SUITE["reshape"]["precomputed"]["apply_inplace"] = @benchmarkable $rmap($out, $M) evals = 4
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

## Open systems: the Liouvillian, symbolically, as a lazy operator and as a sparse matrix
let rng = Xoshiro(6)
    @fermions c
    N = 6
    Hopen, _, _, left, right = FHS.open_system(c, 1:N)
    ham = sum((0.3n) * c[n]' * c[n] for n in 1:N) + sum(0.2im * c[n]' * c[n+1] + hc for n in 1:(N-1))
    dissipator(L) = left(L) * right(L') - 0.5 * (left(L' * L) + right(L' * L))
    jump_in = [sqrt(0.05 + 0.01n) * c[n]' for n in 1:N]
    jump_out = [sqrt(0.04 + 0.01n) * c[n] for n in 1:N]
    make_liouvillian() = 1im * (left(ham) - right(ham)) + sum(dissipator(L) for L in jump_in) + sum(dissipator(L) for L in jump_out)
    liouvillian = make_liouvillian()
    SUITE["open_system"]["liouvillian_symbolic"] = @benchmarkable $make_liouvillian()
    SUITE["open_system"]["liouvillian_sparse"] = @benchmarkable matrix_representation($liouvillian, $Hopen)
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

## Lazy operator for a closed system, with the sparse matrix as reference
let rng = Xoshiro(8)
    @fermions f
    N = 14
    H = hilbert_space(f, 1:N, NumberConservation(7))
    ham = sum(rand(rng) * f[n]' * f[n] for n in 1:N) + sum(f[n]' * f[n+1] + hc for n in 1:N-1) +
          sum(f[n]' * f[n] * f[n+1]' * f[n+1] for n in 1:N-1)
    SUITE["lazy"]["construct"] = @benchmarkable representation($ham, $H, :lazy)
    L = representation(ham, H, :lazy)
    x = randn(rng, ComplexF64, dim(H))
    y = similar(x)
    L_cached = FHS.SciMLOperators.cache_operator(L, x)
    SUITE["lazy"]["mul!"] = @benchmarkable mul!($y, $L_cached, $x)
    M = matrix_representation(ham, H)
    SUITE["lazy"]["sparse_mul!_reference"] = @benchmarkable mul!($y, $M, $x)
end

## Spin-1/2 Heisenberg chain with Sz conservation
let
    @spins S 1 // 2
    N = 12
    Sz0 = FHS.AdditiveConstraint(0; maps=s -> s.m)
    H = hilbert_space(S, 1:N, Sz0)
    heisenberg = sum(S[k][:x] * S[k+1][:x] + S[k][:y] * S[k+1][:y] + S[k][:z] * S[k+1][:z] for k in 1:N-1) + sum(0.1k * S[k][:z] for k in 1:N)
    g = SUITE["spin"]
    g["hilbert_space_Sz0"] = @benchmarkable hilbert_space($S, $(1:14), $Sz0)
    g["symbolic_heisenberg"] =
        @benchmarkable sum($S[k][:x] * $S[k+1][:x] + $S[k][:y] * $S[k+1][:y] + $S[k][:z] * $S[k+1][:z] for k in 1:$N-1)
    g["matrix_representation_heisenberg"] = @benchmarkable matrix_representation($heisenberg, $H)
end

## Spinful Hubbard chain with fixed particle numbers per spin
let
    @fermions f
    hubbard(c, N, t, U) =
        sum(-t * c[i, σ]' * c[i+1, σ] + hc for σ in (:↑, :↓) for i in 1:N-1) +
        sum(U * c[i, :↑]'c[i, :↑] * c[i, :↓]'c[i, :↓] for i in 1:N)
    N = 10
    modes_up = [(i, :↑) for i in 1:N]
    modes_dn = [(i, :↓) for i in 1:N]
    H = tensor_product(hilbert_space(f, modes_up, NumberConservation(3)), hilbert_space(f, modes_dn, NumberConservation(2)))
    ham = hubbard(f, N, 1.0, 4.0)
    no_double = prod(NumberConservation(0:1, [f[k, :↑], f[k, :↓]]) for k in 1:N)
    Hnd = constrain_space(H, no_double)
    g = SUITE["hubbard"]
    g["space"] = @benchmarkable tensor_product(hilbert_space($f, $modes_up, NumberConservation(3)), hilbert_space($f, $modes_dn, NumberConservation(2)))
    g["symbolic"] = @benchmarkable $hubbard($f, $N, 1.0, 4.0)
    g["matrix_representation"] = @benchmarkable matrix_representation($ham, $H)
    g["constrain_no_double_occupation"] = @benchmarkable constrain_space($H, $no_double)
    g["matrix_representation_projected"] = @benchmarkable matrix_representation($ham, $Hnd; projection=true)

    # Many sites, few particles: one constraint per site.
    N = 30
    H30 = tensor_product(hilbert_space(f, [(i, :↑) for i in 1:N], NumberConservation(2)),
        hilbert_space(f, [(i, :↓) for i in 1:N], NumberConservation(1)))
    no_double30 = prod(NumberConservation(0:1, [f[k, :↑], f[k, :↓]]) for k in 1:N)
    g["constrain_no_double_occupation_N30"] = @benchmarkable constrain_space($H30, $no_double30) seconds = 3
end

## Bose-Hubbard chain
let
    @bosons b
    N = 6
    H = hilbert_space(b, 1:N, 6, NumberConservation(6))
    ham = sum(-1.0 * b[n]' * b[n+1] + hc for n in 1:N-1) + sum(0.5 * b[n]' * b[n]' * b[n] * b[n] for n in 1:N)
    SUITE["bosons"]["hilbert_space"] = @benchmarkable hilbert_space($b, $(1:N), 6, NumberConservation(6))
    SUITE["bosons"]["matrix_representation"] = @benchmarkable matrix_representation($ham, $H)
end

## Partial transpose
let rng = Xoshiro(9)
    @fermions f
    H = hilbert_space(f, 1:8)
    Hsub = hilbert_space(f, 1:3)
    Hcomp = FHS.complementary_subsystem(H, Hsub)
    m = rand(rng, ComplexF64, dim(H), dim(H))
    ptmap = partial_transpose(H, Hsub; complement=Hcomp)
    g = SUITE["partial_transpose"]
    g["eager"] = @benchmarkable partial_transpose($m, $H, $Hsub; complement=$Hcomp)
    g["construct"] = @benchmarkable partial_transpose($H, $Hsub; complement=$Hcomp)
    g["apply"] = @benchmarkable $ptmap($m)
end

## Tensor products of matrices
let rng = Xoshiro(10)
    @fermions f
    H1 = hilbert_space(f, 1:4)
    H2 = hilbert_space(f, 5:9)
    H12 = tensor_product(H1, H2)
    A1 = rand(rng, ComplexF64, dim(H1), dim(H1))
    A2 = rand(rng, ComplexF64, dim(H2), dim(H2))
    g = SUITE["tensor_product"]
    g["matrices"] = @benchmarkable tensor_product(($A1, $A2), ($H1, $H2) => $H12)
    g["generalized_kron"] = @benchmarkable generalized_kron(($A1, $A2), ($H1, $H2), $H12)
    g["spaces"] = @benchmarkable tensor_product($H1, $H2) evals = 10
end

## Number sectors: construction and matrix representation in a sector-ordered basis
let
    @fermions f
    N = 12
    H = hilbert_space(f, 1:N, NumberConservation())
    hop = sum(f[n]' * f[n+1] + hc for n in 1:N-1)
    g = SUITE["sectors"]
    g["construct_all"] = @benchmarkable hilbert_space($f, $(1:N), NumberConservation())
    g["sectors"] = @benchmarkable sectors($H)
    g["matrix_representation_sectors"] = @benchmarkable matrix_representation($hop, $H)
end

## Basis state lookup
let rng = Xoshiro(11)
    @fermions f
    H = hilbert_space(f, 1:16, NumberConservation(8))
    states = shuffle(rng, collect(basisstates(H)))
    SUITE["state_index"]["constrained"] = @benchmarkable sum(s -> state_index(s, $H), $states)
    N = 12
    Hprod = tensor_product(hilbert_space(f, [(i, :↑) for i in 1:N], NumberConservation(3)),
        hilbert_space(f, [(i, :↓) for i in 1:N], NumberConservation(2)))
    states_prod = shuffle(rng, collect(basisstates(Hprod)))
    SUITE["state_index"]["product_space"] = @benchmarkable sum(s -> state_index(s, $Hprod), $states_prod)
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
let rng = Xoshiro(12)
    @fermions f
    H = hilbert_space(f, 1:10)
    kets = Kets(H)
    hop = sum(f[n]' * f[n+1] + hc for n in 1:9)
    k0 = kets("1010101010")
    ψ = hop * (hop * (hop * k0))
    strings = [join(rand(rng, ('0', '1'), 10)) for _ in 1:1000]
    g = SUITE["symbolic_states"]
    g["op_on_ket"] = @benchmarkable $hop * ($hop * ($hop * $k0))
    g["parse_kets"] = @benchmarkable map($kets, $strings)
    g["vector_representation"] = @benchmarkable representation($ψ, $H)
end

## Majoranas
let
    @majoranas γ
    H = hilbert_space(γ, 1:20)
    op = sum(1im * γ[2n-1] * γ[2n] for n in 1:10) + sum(1im * γ[n] * γ[n+1] * γ[n+2] * γ[n+3] for n in 1:17)
    SUITE["majorana"]["matrix_representation"] = @benchmarkable matrix_representation($op, $H)
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
