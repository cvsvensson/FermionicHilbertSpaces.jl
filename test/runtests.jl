using TestItemRunner

# The suite is dominated by compilation, so the test items are split by name over separate processes.
# `FHS_TEST_SHARDS=1` runs them all in this process.
nshards = parse(Int, get(ENV, "FHS_TEST_SHARDS", string(min(Sys.CPU_THREADS, 4))))
shard = parse(Int, get(ENV, "FHS_TEST_SHARD", "0"))
if nshards == 1 || shard > 0
    @run_package_tests verbose = true filter = ti -> nshards == 1 || mod(hash(ti.name), nshards) == shard - 1
else
    outputs = [IOBuffer() for _ in 1:nshards]
    procs = map(1:nshards) do s
        cmd = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__)`, "FHS_TEST_SHARD" => s, "FHS_TEST_SHARDS" => nshards)
        run(pipeline(ignorestatus(cmd); stdout = outputs[s], stderr = outputs[s]); wait = false)
    end
    foreach(wait, procs)
    foreach(io -> write(stdout, take!(io)), outputs)
    failed = findall(!success, procs)
    isempty(failed) || error("Test shards $failed failed.")
end


@testitem "Basis" setup = [JordanWigner] begin
    using SparseArrays, LinearAlgebra, Random
    Random.seed!(1234)
    @fermions f
    N = 2
    H = hilbert_space(f, 1:N)
    labels = vec(collect(Base.product(1:N, (:↑, :↓))))
    Hspin = hilbert_space(f, labels, NumberConservation())
    @test representation(f[1], H) isa SparseMatrixCSC
    @test parityoperator(H) isa SparseMatrixCSC
    @test parityoperator(Hspin) isa SparseMatrixCSC

    H = hilbert_space(f, 1:3)
    Hs = (hilbert_space(f, 1:1), hilbert_space(f, 2:2), hilbert_space(f, 3:3))
    Hw = tensor_product(Hs)

    v = [FermionicHilbertSpaces.basisstate(i, H).f for i in 1:8]
    t1 = reshape(v, H => Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == t2

    H = hilbert_space(f, 1:3, ParityConservation())
    v = [FermionicHilbertSpaces.basisstate(i, H).f for i in 1:8]
    t1 = reshape(v, H => Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == t2

    import FermionicHilbertSpaces: atomic_factors
    labels = vec(collect(Base.product(1:N, (:↑, :↓))))
    H = hilbert_space(f, labels)
    Hparity = hilbert_space(f, labels, ParityConservation())
    ρ = Matrix(Hermitian(rand(2^4, 2^4) .- 0.5))
    ρ = ρ / tr(ρ)
    function bilinears(H, labels)
        c = JordanWigner.annihilators(H)
        ops = reduce(vcat, [[c[l], c[l]'] for l in labels])
        return [op1 * op2 for (op1, op2) in Base.product(ops, ops)]
    end
    getlabels(Hsub) = map(FermionicHilbertSpaces.label, atomic_factors(Hsub))
    function bilinear_equality(H, Hsub, ρ)
        subsystem = getlabels(Hsub)
        ρsub = partial_trace(ρ, H, Hsub)
        @test tr(ρsub) ≈ 1
        all((tr(op1 * ρ) ≈ tr(op2 * ρsub)) for (op1, op2) in zip(bilinears(H, subsystem), bilinears(Hsub, subsystem)))
    end
    function get_subsystems(c, N)
        t = collect(Base.product(ntuple(i -> getlabels(c), N)...))
        (collect(t[I]) for I in CartesianIndices(t) if issorted(Tuple(I)) && allunique(Tuple(I)))
    end
    for N in 1:4
        @test all(bilinear_equality(H, hilbert_space(f, subsystem), ρ) for subsystem in get_subsystems(H, N))
        @test all(bilinear_equality(H, hilbert_space(f, subsystem, ParityConservation()), ρ) for subsystem in get_subsystems(H, N))
        @test all(bilinear_equality(H, hilbert_space(f, subsystem, ParityConservation()), ρ) for subsystem in get_subsystems(Hparity, N))
        @test all(bilinear_equality(H, hilbert_space(f, subsystem), ρ) for subsystem in get_subsystems(Hparity, N))
    end
end


@testitem "Partial trace and embed maps match eager evaluation" begin
    using LinearAlgebra, SparseArrays, LinearMaps
    import FermionicHilbertSpaces: complementary_subsystem
    @fermions f

    function check(H, Hsub, ops)
        d, dsub = dim(H), dim(Hsub)
        pt, emb = partial_trace(H => Hsub), embed(Hsub => H)
        ptmat = Matrix(LinearMap(x -> vec(partial_trace(reshape(x, d, d), H => Hsub)), dsub^2, d^2))
        embmat = Matrix(LinearMap(x -> vec(embed(reshape(x, dsub, dsub), Hsub => H; projection = true)), d^2, dsub^2))
        @test ptmat ≈ embmat'
        @test sparse(pt) ≈ ptmat
        @test sparse(emb) ≈ embmat

        m, ψ, msub = rand(ComplexF64, d, d), rand(ComplexF64, d), rand(ComplexF64, dsub, dsub)
        ref = partial_trace(m, H => Hsub)
        for op in ops
            @test tr(representation(op, Hsub) * ref) ≈ tr(representation(op, H) * m)
        end
        @test pt(m) ≈ pt(vec(m)) ≈ ref
        @test pt(ψ) ≈ partial_trace(ψ, H => Hsub)
        for (out, x, y) in (
                (zeros(ComplexF64, dsub, dsub), m, ref), (zeros(ComplexF64, dsub^2), vec(m), vec(ref)),
                (zeros(ComplexF64, dsub^2), m, vec(ref)), (zeros(ComplexF64, dsub^2), ψ, vec(partial_trace(ψ, H => Hsub))),
            )
            @test pt(out, x) === out
            @test out ≈ y
        end
        embref = embed(msub, Hsub => H; projection = true)
        @test emb(msub) ≈ embref
        for (out, x, y) in ((zeros(ComplexF64, d, d), msub, embref), (zeros(ComplexF64, d^2), vec(msub), vec(embref)))
            @test emb(out, x) === out
            @test out ≈ y
        end
    end

    for qn in (nothing, ParityConservation(), NumberConservation())
        space(sites) = isnothing(qn) ? hilbert_space(f, sites) : hilbert_space(f, sites, qn)
        H = space(1:3)
        for sites in ([1], [2], [1, 2], [1, 3], [2, 3])
            check(H, space(sites), [[f[k] for k in sites]; f[first(sites)]' * f[last(sites)]])
        end
    end

    H = hilbert_space(f, 1:4)
    Hsub = hilbert_space(f, [2, 4])
    msub = rand(ComplexF64, dim(Hsub), dim(Hsub))
    @test partial_trace(H => Hsub)(embed(msub, Hsub => H)) ≈ msub * dim(complementary_subsystem(H, Hsub))

    H = hilbert_space(f, 1:4, NumberConservation(2))
    for sites in ([1, 2], [1, 3, 4])
        check(H, subregion(hilbert_space(f, sites), H), [[f[k]' * f[k] for k in sites]; f[sites[1]]' * f[sites[2]]])
    end
end

@testitem "Partial trace and embed maps: edge cases and errors" begin
    using LinearAlgebra, SparseArrays
    @fermions f
    H = hilbert_space(f, 1:4, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, 1:2), H)
    d, dsub = dim(H), dim(Hsub)
    pt = partial_trace(H => Hsub)
    @test_throws DimensionMismatch pt(zeros(ComplexF64, dsub^2 - 1), rand(ComplexF64, d^2))
    @test_throws DimensionMismatch pt(zeros(ComplexF64, dsub, dsub + 1), rand(ComplexF64, d, d))
    @test_throws DimensionMismatch pt(zeros(ComplexF64, dsub^2 - 1), rand(ComplexF64, d, d))

    # Eager and precomputed embedding must use the same projection default.
    msub = rand(ComplexF64, dsub, dsub)
    @test_throws ArgumentError embed(msub, Hsub => H)
    @test embed(Hsub => H)(msub) ≈ embed(msub, Hsub => H; projection = true)

    # Tracing onto the same atoms is a basis transformation, and onto the whole space it returns the input.
    Hfull = hilbert_space(f, 1:2)
    Hparity = hilbert_space(f, 1:2, ParityConservation())
    m = rand(ComplexF64, dim(Hfull), dim(Hfull))
    @test sparse(partial_trace(Hfull => Hfull)) == sparse(I(dim(Hfull)^2))
    @test partial_trace(Hfull => Hparity)(m) ≈ partial_trace(m, Hfull => Hparity)
    for K in (Hfull, H)
        mK, vK = rand(ComplexF64, dim(K), dim(K)), rand(ComplexF64, dim(K))
        @test partial_trace(mK, K => K) ≈ mK
        @test partial_trace(vK, K => K) ≈ vK * vK'
    end

    Hvac = hilbert_space(f, 3:3, NumberConservation(0))
    Hmode = hilbert_space(f, 4:4)
    Hproduct = tensor_product(Hvac, Hmode)
    ρ = [1 2 + im; 3 - im 4]
    @test partial_trace(ρ, Hproduct => Hvac) ≈ reshape([tr(ρ)], 1, 1)
    @test partial_trace(ρ, Hproduct => Hmode) ≈ ρ

    @test partial_trace(2I, Hfull, hilbert_space(f, [2])) ≈ 2 * partial_trace(I(dim(Hfull)), Hfull, hilbert_space(f, [2]))
end

@testitem "Identity in matrix_representation" begin
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:2)
    @test representation(1 + 0 * f[1], H) == I
    @test representation(1, H) == I
    @test representation(I, H) == I
    @test representation(1 + f[1], H) == I + representation(f[1], H)
end

@testitem "Representation of products with interleaved groups" begin
    using FermionicHilbertSpaces: NCMul
    @fermions a
    @fermions b
    @fermions c
    H = tensor_product(hilbert_space(a, 1:2), hilbert_space(b, 1:2), hilbert_space(c, 1:2))
    # NCMul is used directly since `*` would normal order the factors. Operators on different groups commute.
    for (op, parts) in (
        (NCMul(2.0, [b[2], c[1]', a[1]', c[2], b[1]', a[2]]), (a[1]' * a[2], b[2] * b[1]', c[1]' * c[2])),
        (NCMul(-1.5, [c[2]', a[1], c[1], a[2]']), (a[1] * a[2]', c[2]' * c[1])),
    )
        expected = op.coeff * prod(representation(part, H) for part in parts)
        @test !iszero(expected)
        @test representation(op, H) ≈ expected
    end
end

@testitem "state_index for states which are convertible but not identical" begin
    # These tests failed when state_index was overly restrictive on the type of the state argument.
    using FermionicHilbertSpaces: state_index, BosonicState, ProductState
    @fermions f
    H = hilbert_space(f, 1:3)
    Hcons = hilbert_space(f, 1:3, NumberConservation(0:1))
    sf0 = FockNumber(0)
    state = first(basisstates(H))
    # state has UInt and sf0 has Int
    # they are == but not ===
    @test state == sf0 
    @test state !== sf0
    @test state_index(sf0, H) == 1
    @test state_index(sf0, Hcons) == 1
    Hsector = tensor_product(Hcons, hilbert_space(f, 4:5, NumberConservation(1)))
    @test all(state_index(FockNumber(Int(s.f)), Hsector) == i for (i, s) in enumerate(basisstates(Hsector)))

    @boson b
    Hb = hilbert_space(b, 1)
    state = first(basisstates(Hb))
    sb0 = BosonicState(0)
    @test 1 == state_index(sb0, Hb)

    for Hprod in [tensor_product(Hcons, Hb), tensor_product(H, Hb)]
        local state = first(basisstates(Hprod))
        @test 1 == state_index(state, Hprod)
        state2 = ProductState((sf0, sb0))
        @test state == state2
        @test state !== state2
        @test 1 == state_index(state2, Hprod)
    end
end