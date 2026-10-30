# Cases for compare.jl, which runs this file in fresh processes:
#   julia cases.jl runtime <default|full> <out>   all runtime cases
#   julia cases.jl firstcall <name> <out>         the load time and one first-call workflow
#   julia cases.jl list                           the names of the first-call workflows
# Each line of <out> is `name<TAB>seconds<TAB>bytes`, with bytes -1 when not measured.
# A case that throws, e.g. because the ref lacks the API, is left out.
t0 = time_ns()
using FermionicHilbertSpaces
const LOAD_TIME = (time_ns() - t0) / 1.0e9
using FermionicHilbertSpaces.FreeFermions, LinearAlgebra, SparseArrays, Random
using OhMyThreads: StaticScheduler
const FHS = FermionicHilbertSpaces

hubbard(c, N) = sum(-1.0 * c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(N - 1)) +
    sum(4.0 * c[i, :↑]'c[i, :↑] * c[i, :↓]'c[i, :↓] for i in 1:N)
hubbard_modes(N) = [(i, σ) for i in 1:N for σ in (:↑, :↓)]
chain(c, N) = sum(0.1n * c[n]' * c[n] for n in 1:N) + sum(c[n]' * c[n + 1] + hc for n in 1:(N - 1)) +
    sum(c[n]' * c[n] * c[n + 1]' * c[n + 1] for n in 1:(N - 1))
bdg_chain(c, N) = sum(0.1n * c[n]' * c[n] for n in 1:N) + sum(c[n]' * c[n + 1] + 0.5 * c[n]' * c[n + 1]' + hc for n in 1:(N - 1))

const FIRSTCALL = Dict(
    "bdg eigen" => function ()
        @fermions f
        eigen(representation(bdg_chain(f, 4), hilbert_space(f, 1:4, BdG()), :dense), BdGEigen())
    end,
    "dense bdg" => function ()
        @fermions f
        representation(bdg_chain(f, 4), hilbert_space(f, 1:4, BdG()), :dense)
    end,
    "dense sector" => function ()
        @fermions f
        representation(chain(f, 4), hilbert_space(f, 1:4, NumberConservation(2)), :dense)
    end,
    "tensor_product constraint" => function ()
        @fermions f
        tensor_product(hilbert_space(f, 1:2), hilbert_space(f, 3:4); constraint = NumberConservation(1) * ParityConservation())
    end,
    "mean_field" => function ()
        @fermions f
        H, G = hilbert_space(f, 1:4, BdG()), Matrix(0.5I, 8, 8)
        mean_field(chain(f, 4), G, H), expectation(chain(f, 4), G, H)
    end,
)

# Each setup takes a size and returns the function to time. A case runs at each of its
# default sizes, or at each of its full sizes with `full`.
function runtime_cases(full)
    @fermions f
    @spins S 1 // 2
    rng = Xoshiro(1)
    cases = Pair{String, Function}[]
    case!(setup, name, sizes, full_sizes) = foreach(n -> push!(cases, "$name $n" => () -> setup(n)), full ? full_sizes : sizes)

    case!("hubbard full space sparse", (9,), (11,)) do N
        H, ham = hilbert_space(f, hubbard_modes(N)), hubbard(f, N)
        () -> matrix_representation(ham, H)
    end
    for nchunks in (1, 8)
        case!("hubbard full space sparse $nchunks chunks", (9,), (11,)) do N
            H, ham = hilbert_space(f, hubbard_modes(N)), hubbard(f, N)
            chunking = FHS.StateChunking(StaticScheduler(; nchunks))
            () -> representation(ham, H; chunking)
        end
    end
    case!("hubbard sector sparse", (11,), (14,)) do N
        sector(σ) = hilbert_space(f, [(i, σ) for i in 1:N], NumberConservation(N ÷ 2))
        H, ham = tensor_product(sector(:↑), sector(:↓)), hubbard(f, N)
        () -> matrix_representation(ham, H)
    end
    case!("constrain no double occupation", (30,), (50,)) do N
        sector(σ) = hilbert_space(f, [(i, σ) for i in 1:N], NumberConservation(2))
        H = tensor_product(sector(:↑), sector(:↓))
        no_double = prod(NumberConservation(0:1, [f[k, :↑], f[k, :↓]]) for k in 1:N)
        () -> constrain_space(H, no_double)
    end
    case!("heisenberg sparse", (16,), (18,)) do N
        H = hilbert_space(S, 1:N, FHS.AdditiveConstraint(0; maps = s -> s.m))
        ham = sum(S[k][:x] * S[k + 1][:x] + S[k][:y] * S[k + 1][:y] + S[k][:z] * S[k + 1][:z] for k in 1:(N - 1))
        () -> matrix_representation(ham, H)
    end
    case!("dense half filling", (12,), (14,)) do N
        H, ham = hilbert_space(f, 1:N, NumberConservation(N ÷ 2)), chain(f, N)
        () -> representation(ham, H, :dense)
    end
    case!("partial_trace dense (n, k)", ((11, 2), (11, 6), (11, 9)), ((12, 2), (12, 6), (12, 10))) do (N, k)
        H = hilbert_space(f, 1:N)
        ρ = rand(rng, ComplexF64, dim(H), dim(H))
        () -> partial_trace(ρ, H => subregion(hilbert_space(f, 1:k), H))
    end
    case!("partial_trace pure (n, k)", ((16, 4), (16, 8), (16, 12)), ((20, 4), (20, 10), (20, 16))) do (N, k)
        H = hilbert_space(f, 1:N)
        ψ = normalize(rand(rng, ComplexF64, dim(H)))
        () -> partial_trace(ψ, H => subregion(hilbert_space(f, 1:k), H))
    end
    lazy_mul(H, N) = (L = representation(chain(f, N), H, :lazy); x = randn(rng, ComplexF64, dim(H)); y = similar(x);
        Lc = FHS.SciMLOperators.cache_operator(L, x); () -> mul!(y, Lc, x))
    case!(N -> lazy_mul(hilbert_space(f, 1:N), N), "lazy mul! full space", (14, 16), (16, 18))
    case!(N -> lazy_mul(hilbert_space(f, 1:N, NumberConservation(N ÷ 2)), N), "lazy mul! half filling", (16,), (20,))
    case!("bdg eigen", (300,), (800,)) do N
        A = representation(bdg_chain(f, N), hilbert_space(f, 1:N, BdG()), :dense)
        () -> eigen(A, BdGEigen())
    end
    case!("mean_field", (200,), (400,)) do N
        H = hilbert_space(f, 1:N, BdG())
        G = correlation_matrix(eigen(representation(bdg_chain(f, N), H, :dense), BdGEigen()))
        op = chain(f, N)
        () -> mean_field(op, G, H)
    end
    return cases
end

function measure(f; budget = 0.3)
    f()
    t, bytes, spent, n = Inf, typemax(Int), 0.0, 0
    while n < 2 || spent < budget
        s = @timed f()
        t, bytes, spent, n = min(t, s.time), min(bytes, s.bytes), spent + s.time, n + 1
    end
    return t, bytes
end

function run_all(io, full)
    for (name, setup) in runtime_cases(full)
        try
            t, bytes = measure(setup())
            println(io, name, '\t', t, '\t', bytes)
            flush(io)
        catch err
            @warn "Skipping $name" exception = (err, catch_backtrace())
        end
        GC.gc()
    end
    return
end

if ARGS[1] == "firstcall"
    open(ARGS[3], "w") do io
        println(io, "load\t", LOAD_TIME, "\t-1")
        try
            println(io, "first call ", ARGS[2], '\t', @elapsed(FIRSTCALL[ARGS[2]]()), "\t-1")
        catch err
            @warn "Skipping $(ARGS[2])" exception = err
        end
    end
elseif ARGS[1] == "runtime"
    open(io -> run_all(io, ARGS[2] == "full"), ARGS[3], "w")
else
    foreach(println, sort!(collect(keys(FIRSTCALL))))
end
