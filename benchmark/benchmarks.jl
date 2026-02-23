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
SUITE["partial_trace"]["standard"] = @benchmarkable partial_trace($m, $(H => Hsub))
SUITE["partial_trace"]["map"] = @benchmarkable partial_trace($(H => Hsub))
d = dim(Hsub)
msub = rand(ComplexF64, d, d)
SUITE["embed"] = @benchmarkable embed($msub, $(Hsub => H))

N = 60
weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
SUITE["generate_states"]["int"] = @benchmarkable FermionicHilbertSpaces.generate_states($weights, $allowed_ones, $N)

N = 64
weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
SUITE["generate_states"]["big_int"] = @benchmarkable FermionicHilbertSpaces.generate_states($weights, $allowed_ones, $N)

## Benchmark partial trace algorithms
import FermionicHilbertSpaces: FullPartialTraceAlg, SubsystemPartialTraceAlg, default_partial_trace_alg
# This scenario highlights cases where FullPartialTraceAlg is expected to be faster.
N = 30
H = hilbert_space(1:N, NumberConservation(2))
# Define a subsystem for the first 10 modes
Hsub = subregion(1:10, H)
# Create a random Hermitian matrix on the full Hilbert space
d = dim(H)
mat = rand(ComplexF64, d, d)
matsparse = sprand(ComplexF64, d, d, 0.01)
Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
alg = default_partial_trace_alg(Hsub, H, Hcomp)
SUITE["partial_trace_algorithms"]["default=$alg"]["Sparse space"]["Dense"]["subsystem_alg"] =
    @benchmarkable partial_trace($mat, $H, $Hsub,
        alg=SubsystemPartialTraceAlg(),)

SUITE["partial_trace_algorithms"]["default=$alg"]["Sparse space"]["Dense"]["full_alg"] =
    @benchmarkable partial_trace($mat, $H, $Hsub,
        alg=FullPartialTraceAlg())

SUITE["partial_trace_algorithms"]["default=$alg"]["Sparse space"]["Sparse"]["subsystem_alg"] =
    @benchmarkable partial_trace($matsparse, $H, $Hsub,
        alg=SubsystemPartialTraceAlg())

SUITE["partial_trace_algorithms"]["default=$alg"]["Sparse space"]["Sparse"]["full_alg"] =
    @benchmarkable partial_trace($matsparse, $H, $Hsub,
        alg=FullPartialTraceAlg())

# Setup for Standard Full Fock Space (No Symmetry)
# This scenario tests the standard case where SubsystemPartialTraceAlg is typically efficient.
N_std = 12
H_std = hilbert_space(1:N_std)
Hsub_std = subregion(1:2, H_std)
d_std = dim(H_std)
m_std = rand(ComplexF64, d_std, d_std)
m_sparse = sprand(ComplexF64, d_std, d_std, 0.01)
Hcomp = FermionicHilbertSpaces.complementary_subsystem(H_std, Hsub_std)
alg = default_partial_trace_alg(Hsub_std, H_std, Hcomp)
SUITE["partial_trace_algorithms"]["default=$alg"]["Full space"]["Dense"]["subsystem_alg"] =
    @benchmarkable partial_trace($m_std, $H_std, $Hsub_std,
        alg=SubsystemPartialTraceAlg())

SUITE["partial_trace_algorithms"]["default=$alg"]["Full space"]["Dense"]["full_alg"] =
    @benchmarkable partial_trace($m_std, $H_std, $Hsub_std,
        alg=FullPartialTraceAlg())

SUITE["partial_trace_algorithms"]["default=$alg"]["Full space"]["Sparse"]["subsystem_alg"] =
    @benchmarkable partial_trace($m_sparse, $H_std, $Hsub_std,
        alg=SubsystemPartialTraceAlg())

SUITE["partial_trace_algorithms"]["default=$alg"]["Full space"]["Sparse"]["full_alg"] =
    @benchmarkable partial_trace($m_sparse, $H_std, $Hsub_std,
        alg=FullPartialTraceAlg())