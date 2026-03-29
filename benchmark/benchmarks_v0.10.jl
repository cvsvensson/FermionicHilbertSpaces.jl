using BenchmarkTools
using FermionicHilbertSpaces
using SparseArrays
using Random
const SUITE = BenchmarkGroup()

Random.seed!(1)
@fermions f
N = 12
H = hilbert_space(f, 1:N, ParityConservation())
op = sum(rand() * f[n]' * f[n] for n in 1:N) + sum(1im * f[n]' * f[n+1] + hc for n in 1:N-1)
Hsub = hilbert_space(f, 1:div(N, 4), ParityConservation())
d = dim(H)
m = sprand(ComplexF64, d, d, 1 / 2^N)

SUITE["hilbert_space"] = @benchmarkable hilbert_space($f, $(1:N), $ParityConservation())
SUITE["symbolic"]["sum"] = @benchmarkable sum(f[n]' * f[n] + hc for n in 1:100)
SUITE["symbolic"]["sum_square"] = @benchmarkable sum(f[n]' * f[n] + hc for n in 1:50)^2

labels = shuffle(1:10)
SUITE["symbolic"]["deep_product"] = @benchmarkable prod(f[l] for l in labels) * prod(f[l]' for l in labels)
SUITE["matrix_representation"]["standard"] = @benchmarkable matrix_representation($op, $H)

Hsp = single_particle_hilbert_space(f, 1:1000)
opsp = sum(rand() * f[n]' * f[n] for n in 1:1000) + sum(rand(ComplexF64) * f[n]' * f[n+1] + hc for n in 1:999)
SUITE["matrix_representation"]["free_fermion"] = @benchmarkable matrix_representation($opsp, $Hsp)

opsp_bdg = opsp + sum(rand(ComplexF64) * f[n]' * f[n+1]' + hc for n in 1:999)
Hbdg = bdg_hilbert_space(f, 1:1000)
SUITE["matrix_representation"]["bdg"] = @benchmarkable matrix_representation($opsp_bdg, $Hbdg)

SUITE["complement"]["fermions"] = @benchmarkable FermionicHilbertSpaces.complementary_subsystem($H, $Hsub)

complement = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
SUITE["partial_trace"]["fermions"]["map"] = @benchmarkable partial_trace($(H => Hsub); complement=$complement)
SUITE["partial_trace"]["fermions"]["standard"] = @benchmarkable partial_trace($m, $(H => Hsub), complement=$complement)

d = dim(Hsub)
msub = rand(ComplexF64, d, d)
SUITE["embed"]["fermions"] = @benchmarkable embed($msub, $(Hsub => H); complement=$complement)

N = 60
weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
H = hilbert_space(f, 1:N)
constraint = prod(NumberConservation(allowed, missing, w) for (allowed, w) in zip(allowed_ones, weights))
SUITE["generate_states"]["int"] = @benchmarkable FermionicHilbertSpaces.generate_states($H.modes, $constraint, $H; process_result=FermionicHilbertSpaces.CombineFockNumbersProcessor{FockNumber{Int}}())

N = 64
weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
allowed_ones = [[0, 1], [-1, 0], [2]]
H = hilbert_space(f, 1:N)
constraint = prod(NumberConservation(allowed, missing, w) for (allowed, w) in zip(allowed_ones, weights))
SUITE["generate_states"]["big_int"] = @benchmarkable FermionicHilbertSpaces.generate_states($H.modes, $constraint, $H; process_result=FermionicHilbertSpaces.CombineFockNumbersProcessor{FockNumber{Int}}())

## Benchmark partial trace algorithms
import FermionicHilbertSpaces: FullPartialTraceAlg, SubsystemPartialTraceAlg, default_partial_trace_alg
_name(::FullPartialTraceAlg) = "FullAlg"
_name(::SubsystemPartialTraceAlg) = "SubAlg"
# This scenario highlights cases where FullPartialTraceAlg is expected to be faster.
N = 30
H = hilbert_space(f, 1:N, NumberConservation(2))
# Define a subsystem for the first 10 modes
Hsub = subregion(hilbert_space(f, 1:10), H)
# Create a random Hermitian matrix on the full Hilbert space
d = dim(H)
mat = rand(ComplexF64, d, d)
matsparse = sprand(ComplexF64, d, d, 0.01)
Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
def_alg = default_partial_trace_alg(Hsub, H, Hcomp)
def = _name(def_alg)
for alg in [SubsystemPartialTraceAlg(), FullPartialTraceAlg()]
    name = _name(alg)
    SUITE["partial_trace_algorithms"]["default=$def"]["Sparse space"]["Dense"]["$name"] =
        @benchmarkable partial_trace($mat, $H, $Hsub; alg=$alg)
    SUITE["partial_trace_algorithms"]["default=$def"]["Sparse space"]["Sparse"]["$name"] =
        @benchmarkable partial_trace($matsparse, $H, $Hsub; alg=$alg)
end

# Setup for Standard Full Fock Space (No Symmetry)
# This scenario tests the standard case where SubsystemPartialTraceAlg is typically efficient.
N_std = 10
H_std = hilbert_space(f, 1:N_std)
Hsub_std = subregion(hilbert_space(f, 1:2), H_std)
d_std = dim(H_std)
m_std = rand(ComplexF64, d_std, d_std)
m_sparse = sprand(ComplexF64, d_std, d_std, 0.01)
Hcomp = FermionicHilbertSpaces.complementary_subsystem(H_std, Hsub_std)
def_alg = default_partial_trace_alg(Hsub_std, H_std, Hcomp)
def = _name(def_alg)
for alg in [SubsystemPartialTraceAlg(), FullPartialTraceAlg()]
    name = _name(alg)
    SUITE["partial_trace_algorithms"]["default=$def"]["Full space"]["Dense"]["$name"] =
        @benchmarkable partial_trace($m_std, $H_std, $Hsub_std; alg=$alg)
    SUITE["partial_trace_algorithms"]["default=$def"]["Full space"]["Sparse"]["$name"] =
        @benchmarkable partial_trace($m_sparse, $H_std, $Hsub_std; alg=$alg)
end

## Product spaces
@fermions f
@bosons b
Hf = hilbert_space(f, 1:2)
Hb = hilbert_space(b, 1:4, 2)
H = tensor_product(Hf, Hb)
Hsub = tensor_product(hilbert_space(f, 1:1), [hilbert_space(b[n], 2) for n in 1:2]...)
SUITE["complement"]["product space"] = @benchmarkable FermionicHilbertSpaces.complementary_subsystem($H, $Hsub)

complement = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
SUITE["partial_trace"]["product space"]["map"] = @benchmarkable partial_trace($(H => Hsub); complement=$complement)
m = rand(ComplexF64, dim(H), dim(H))
SUITE["partial_trace"]["product space"]["standard"] = @benchmarkable partial_trace($m, $(H => Hsub); complement=$complement)
msub = rand(ComplexF64, dim(Hsub), dim(Hsub))
SUITE["embed"]["product space"] = @benchmarkable embed($msub, $(Hsub => H); complement=$complement)