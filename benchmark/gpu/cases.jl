# Untested prototype: written without being run. Cases and helpers shared by baseline.jl and hybrid.jl.
using FermionicHilbertSpaces, SparseArrays, LinearAlgebra
using BenchmarkTools: @belapsed
using OhMyThreads: StaticScheduler, tmap
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

# The (I, J, V) entries of `op` on `space`, from the accumulator and state loop that FHS uses for sparse matrices.
# Duplicate entries are not summed.
host_ijv(op, H; chunking = FHS.NoChunking(), projection = false) = host_ijv(op, FHS._checked_group_ids(op, H), H, chunking; projection)
function host_ijv(op, bases, space, ::FHS.NoChunking, cols = 1:dim(space); kwargs...)
    terms = length(bases) == 1 ? collect(FHS.NCterms(op)) :
        [FHS.partition_product(term, bases, FHS.groups(FHS._input_space(space))) for term in FHS.NCterms(op)]
    T = mapreduce(FHS.mat_eltype, promote_type, terms; init = typeof(op.coeff))
    ijv = FHS.sparse_matrix_accumulator(T, (0, 0))
    FHS._collect_block!(ijv, 0, FHS._precomputed_terms(terms, space), op.coeff, space, cols, 0; kwargs...)
    return ijv
end
function host_ijv(op, bases, space, chunking::FHS.StateChunking; kwargs...)
    chunks = Base.get_extension(FHS, :OhMyThreadsExt)._state_chunk_data(space, chunking)
    parts = tmap(cols -> host_ijv(op, bases, space, FHS.NoChunking(), cols; kwargs...), chunks; scheduler = chunking.scheduler)
    return ntuple(k -> reduce(vcat, getindex.(parts, k)), 3)
end

# The large cases take seconds per run, so only a few runs
besttime(f) = @belapsed $f() samples = 5 evals = 1 seconds = 30
report(label, t) = println("  ", rpad(label, 44), lpad(round(t * 1.0e3; digits = 2), 12), " ms")
