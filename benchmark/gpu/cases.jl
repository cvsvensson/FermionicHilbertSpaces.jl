# Cases and helpers shared by baseline.jl and hybrid.jl.
using FermionicHilbertSpaces, SparseArrays, LinearAlgebra
using BenchmarkTools: @belapsed
using OhMyThreads: StaticScheduler, index_chunks, tmap
const FHS = FermionicHilbertSpaces

hubbard(c, N) = sum(-1.0 * c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(N - 1)) +
    sum(4.0 * c[i, :↑]'c[i, :↑] * c[i, :↓]'c[i, :↓] for i in 1:N)
hubbard_modes(N) = [(i, σ) for i in 1:N for σ in (:↑, :↓)]
chain(c, N) = sum(0.1n * c[n]' * c[n] for n in 1:N) + sum(c[n]' * c[n + 1] + hc for n in 1:(N - 1)) +
    sum(c[n]' * c[n] * c[n + 1]' * c[n + 1] for n in 1:(N - 1))
holstein(c, b, N) = chain(c, N) + 0.5 * b' * b + sum(0.2 * c[n]' * c[n] * (b + b') for n in 1:N)

# Each case is (; name, op, H, kwargs), with kwargs passed to `representation`.
function cases(large)
    @fermions f
    @boson b
    N = large ? 11 : 9
    M = large ? 12 : 10
    Msmall = large ? 8 : 7
    K, nmax = large ? (16, 32) : (14, 16)
    Hf, Hb = hilbert_space(f, 1:K), hilbert_space(b, nmax)
    return [
        (; name = "hubbard full space, N = $N", op = hubbard(f, N), H = hilbert_space(f, hubbard_modes(N)), kwargs = (;)),
        (; name = "hubbard half filling, N = $M", op = hubbard(f, M), H = hilbert_space(f, hubbard_modes(M), NumberConservation(M)), kwargs = (;)),
        (; name = "hubbard half filling, N = $Msmall", op = hubbard(f, Msmall), H = hilbert_space(f, hubbard_modes(Msmall), NumberConservation(Msmall)), kwargs = (;)),
        (; name = "fermions ⊗ boson, K = $K, nmax = $nmax", op = holstein(f, b, K), H = tensor_product(Hf, Hb), kwargs = (;)),
        # The coupling changes the total number, so it needs projection
        (; name = "fermions ⊗ boson, total number ≤ $K", op = holstein(f, b, K),
            H = tensor_product(Hf, Hb; constraint = NumberConservation(0:K)), kwargs = (; projection = true)),
    ]
end

threaded() = FHS.StateChunking(StaticScheduler(; nchunks = Threads.nthreads()))

# Format whose result is the (I, J, V) entries of the matrix, from the state loop that FHS uses for sparse matrices.
# Duplicate entries are not summed. `IJVFormat{T}()` fixes the element type, as `Matrix{T}` does.
struct IJVFormat{T} end
IJVFormat() = IJVFormat{Missing}()
FHS._canonicalize_repr(repr::IJVFormat) = repr
FHS._repr_eltype(default, ::IJVFormat{T}) where {T} = T === Missing ? default : T
function FHS._terms_matrix(::Type{T}, terms, coeff, space, ::IJVFormat, chunking; kwargs...) where {T}
    precomps = FHS._precomputed_terms(terms, space)
    collect_cols(cols) = FHS._collect_ijv(T, precomps, coeff, space, cols, 0; kwargs...)
    chunking isa FHS.NoChunking && return collect_cols(1:dim(space))
    parts = tmap(collect_cols, index_chunks(1:dim(space); n = Threads.nthreads()); scheduler = chunking.scheduler)
    return ntuple(k -> reduce(vcat, getindex.(parts, k)), 3)
end
host_ijv(op, H, T = Missing; kwargs...) = representation(op, H, IJVFormat{T}(); kwargs...)

# The large cases take seconds per run, so only a few runs
besttime(f) = @belapsed $f() samples = 5 evals = 1 seconds = 30
report(label, t) = println("  ", rpad(label, 44), lpad(round(t * 1.0e3; digits = 2), 12), " ms")
