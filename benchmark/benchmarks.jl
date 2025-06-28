using BenchmarkTools
using FermionicHilbertSpaces
using SparseArrays
const SUITE = BenchmarkGroup()

N = 12
H = hilbert_space(1:N, ParityConservation())
@fermions f
op = sum(rand() * f[n]' * f[n] for n in 1:N) + sum(1im * f[n]' * f[n+1] + hc for n in 1:N-1)
Hsub = hilbert_space(1:div(N, 4), ParityConservation())
m = sprand(ComplexF64, size(H)..., 1 / 2^N)

SUITE["hilbert_space"] = @benchmarkable hilbert_space($(1:N), $ParityConservation())
SUITE["symbolic_sum"] = @benchmarkable sum(f[n]' * f[n] + hc for n in 1:N)
SUITE["matrix_representation"] = @benchmarkable matrix_representation($op, $H)
SUITE["partial_trace"] = @benchmarkable partial_trace($m, $(H => Hsub))

msub = rand(ComplexF64, size(Hsub))
SUITE["embed"] = @benchmarkable FermionicHilbertSpaces.embedding($msub, $(Hsub => H))

# using Symbolics
# @variables x
# symop = x * op
# SUITE["matrix_representation_symbolic"] = @benchmarkable matrix_representation($symop, $H)
