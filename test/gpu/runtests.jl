# Runs on JLArrays, a CPU implementation of the GPU array interface that disallows scalar indexing.
# Not part of the main suite: `julia --project=test/gpu test/gpu/runtests.jl`.
using FermionicHilbertSpaces, JLArrays, Test
using FermionicHilbertSpaces: GatherAlg, FactorizedPartialTransposeAlg, FactorizedKronAlg
JLArrays.allowscalar(false)

@testset "partial_trace of GPU arrays" begin
    alg = GatherAlg()
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
end

@testset "embed of GPU arrays" begin
    alg = GatherAlg()
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
    alg = GatherAlg()
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

@testset "partial_transpose of GPU arrays" begin
    alg = FactorizedPartialTransposeAlg()
    @fermions f
    H, HA = hilbert_space(f, 1:4), hilbert_space(f, [1, 3])
    for x in (rand(ComplexF64, dim(H), dim(H)), rand(ComplexF64, dim(H)))
        out = partial_transpose(JLArray(x), H, HA; alg)
        @test out isa JLArray
        @test Array(out) ≈ partial_transpose(x, H, HA)
    end
end

@testset "generalized_kron and tensor_product of GPU arrays" begin
    alg = FactorizedKronAlg()
    @fermions f
    Hs = (hilbert_space(f, [1, 3]), hilbert_space(f, [2, 4, 5]))
    H = hilbert_space(f, 1:5)
    ms = map(Hk -> rand(ComplexF64, dim(Hk), dim(Hk)), Hs)
    vs = map(Hk -> rand(ComplexF64, dim(Hk)), Hs)
    for (op, xs) in ((generalized_kron, ms), (generalized_kron, vs), (tensor_product, ms))
        out = op(map(JLArray, xs), Hs => H; alg)
        @test out isa JLArray
        @test Array(out) ≈ op(xs, Hs => H)
    end
    Hc = hilbert_space(f, 1:5, NumberConservation(2))
    Hcs = map(Hk -> subregion(Hk, Hc), Hs)
    mcs = map(Hk -> rand(ComplexF64, dim(Hk), dim(Hk)), Hcs)
    out = tensor_product(map(JLArray, mcs), Hcs => Hc; alg, projection = true)
    @test out isa JLArray
    @test Array(out) ≈ tensor_product(mcs, Hcs => Hc; projection = true)
    @test_throws ArgumentError tensor_product(map(JLArray, mcs), Hcs => Hc; alg)
end
