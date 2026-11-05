# Runs on JLArrays, a CPU implementation of the GPU array interface that disallows scalar indexing.
# Not part of the main suite: `julia --project=test/gpu test/gpu/runtests.jl`.
using FermionicHilbertSpaces, JLArrays, Test
using FermionicHilbertSpaces: NoScalarIndexingAlg, change_basis
JLArrays.allowscalar(false)

@testset "partial_trace of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    for (H, Hsub) in ((hilbert_space(f, 1:4), hilbert_space(f, [1, 3])), (Hcons, subregion(hilbert_space(f, 1:2), Hcons)))
        for x in (rand(ComplexF64, dim(H), dim(H)), rand(ComplexF64, dim(H)))
            out = partial_trace(JLArray(x), H => Hsub; alg)
            @test out isa JLArray
            @test Array(out) ≈ partial_trace(x, H => Hsub)
        end
    end
    @test_throws ArgumentError partial_trace(JLArray(rand(dim(Hcons), dim(Hcons))), Hcons => hilbert_space(f, 1:2, NumberConservation(1)); alg)

    # States 1 and 2 share the entry (K, b) = (1, 1), with weights 1 and -1.
    ψ = rand(ComplexF64, 3)
    A = [ψ[1] - ψ[2], ψ[3]]
    splits = [((1, 1, 1),), ((1, 1, -1),), ((2, 1, 1),)]
    @test Array(FermionicHilbertSpaces._gather_partial_trace(JLArray(ψ), splits, 2, 1)) ≈ A * A'
end

@testset "change_basis of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsec = hilbert_space(f, 1:4, NumberConservation(2))
    for x in (rand(ComplexF64, dim(Hsec)), rand(ComplexF64, dim(Hsec), dim(Hsec)))
        out = change_basis(JLArray(x), Hsec => H; alg)
        @test out isa JLArray
        @test Array(out) == change_basis(x, Hsec => H)
    end
    ρ = rand(ComplexF64, dim(H), dim(H))
    @test Array(partial_trace(JLArray(ρ), H => Hsec; alg, projection = true)) == partial_trace(ρ, H => Hsec; projection = true)
    @test_throws ArgumentError change_basis(JLArray(ρ), H => Hsec; alg)
end

@testset "embed of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    for (Hsub, H) in ((hilbert_space(f, [1, 3]), hilbert_space(f, 1:4)), (subregion(hilbert_space(f, 1:2), Hcons), Hcons))
        m = rand(ComplexF64, dim(Hsub), dim(Hsub))
        out = embed(JLArray(m), Hsub => H; alg, projection = true)
        @test out isa JLArray
        @test Array(out) ≈ embed(m, Hsub => H; projection = true)
    end
    Hpar = hilbert_space(f, 1:4, ParityConservation())
    v = rand(ComplexF64, 8)
    @test Array(embed(JLArray(v), sector(-1, Hpar) => Hpar; alg)) ≈ embed(v, sector(-1, Hpar) => Hpar)
    @test_throws ArgumentError embed(JLArray(rand(4, 4)), hilbert_space(f, 1:2) => hilbert_space(f, 1:2, NumberConservation(0:1)); alg)
end

@testset "reshape of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    H1, H2 = hilbert_space(f, 1:2), hilbert_space(f, 3:4)
    H = hilbert_space(f, 1:4, NumberConservation(2))
    for x in (rand(ComplexF64, dim(H)), rand(ComplexF64, dim(H), dim(H)))
        T = reshape(JLArray(x), H => (H1, H2); alg)
        @test T isa JLArray
        @test Array(T) == reshape(x, H => (H1, H2))
        @test Array(reshape(T, (H1, H2) => H; alg)) == x
    end
    @test_throws ArgumentError reshape(JLArray(rand(dim(H1), dim(H2))), (H1, H2) => H; alg)
end
