using BenchmarkTools
using FermionicHilbertSpaces
using SparseArrays
using Random
const SUITE = BenchmarkGroup()
Random.seed!(1)

N = 12
H = hilbert_space(1:N, ParityConservation())
@fermions f
op = sum(rand() * f[n]' * f[n] for n in 1:N) + sum(1im * f[n]' * f[n+1] + hc for n in 1:N-1)
Hsub = hilbert_space(1:div(N, 4), ParityConservation())
d = dim(H)
m = sprand(ComplexF64, d, d, 1 / 2^N)

SUITE["hilbert_space"] = @benchmarkable hilbert_space($(1:N), $ParityConservation())
SUITE["symbolic"]["sum"] = @benchmarkable sum(f[n]' * f[n] + hc for n in 1:100)
SUITE["symbolic"]["sum_square"] = @benchmarkable sum(f[n]' * f[n] + hc for n in 1:50)^2

labels = shuffle(1:10)
SUITE["symbolic"]["deep_product"] = @benchmarkable prod(f[l] for l in labels) * prod(f[l]' for l in labels)
SUITE["matrix_representation"]["standard"] = @benchmarkable matrix_representation($op, $H)

Hsp = single_particle_hilbert_space(1:1000)
opsp = sum(rand() * f[n]' * f[n] for n in 1:1000) + sum(rand(ComplexF64) * f[n]' * f[n+1] + hc for n in 1:999)
SUITE["matrix_representation"]["free_fermion"] = @benchmarkable matrix_representation($opsp, $Hsp)

opsp_bdg = opsp + sum(rand(ComplexF64) * f[n]' * f[n+1]' + hc for n in 1:999)
Hbdg = bdg_hilbert_space(1:1000)
SUITE["matrix_representation"]["bdg"] = @benchmarkable matrix_representation($opsp_bdg, $Hbdg)
SUITE["partial_trace"] = @benchmarkable partial_trace($m, $(H => Hsub))
# SUITE["partial_trace_map"] = @benchmarkable FermionicHilbertSpaces.partial_trace_map($(H => Hsub))
d = dim(Hsub)
msub = rand(ComplexF64, d, d)
SUITE["embed"] = @benchmarkable embed($msub, $(Hsub => H))

N = 60
weights = [floor.(2sin.(1:N)), sign.((1:N) .- div(N, 2)), ones(N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
SUITE["generate_states"]["int"] = @benchmarkable FermionicHilbertSpaces.generate_states($weights, $allowed_ones, $N)

N = 64
weights = [floor.(2sin.(1:N)), sign.((1:N) .- div(N, 2)), ones(N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
SUITE["generate_states"]["big_int"] = @benchmarkable FermionicHilbertSpaces.generate_states($weights, $allowed_ones, $N)

# using Symbolics
# @variables x
# symop = x * op
# SUITE["matrix_representation_symbolic"] = @benchmarkable matrix_representation($symop, $H)
