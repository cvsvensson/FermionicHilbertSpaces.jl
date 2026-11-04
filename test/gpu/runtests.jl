# Runs on JLArrays, a CPU implementation of the GPU array interface that disallows scalar indexing.
# Not part of the main suite: `julia --project=test/gpu test/gpu/runtests.jl`.
using FermionicHilbertSpaces, JLArrays, Test
JLArrays.allowscalar(false)

@testset "partial_trace of GPU arrays" begin
    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    for (H, Hsub) in ((hilbert_space(f, 1:4), hilbert_space(f, [1, 3])), (Hcons, subregion(hilbert_space(f, 1:2), Hcons)))
        for x in (rand(ComplexF64, dim(H), dim(H)), rand(ComplexF64, dim(H)))
            out = partial_trace(JLArray(x), H => Hsub)
            @test out isa JLArray
            @test Array(out) ≈ partial_trace(x, H => Hsub)
        end
    end
    @test_throws ArgumentError partial_trace(JLArray(rand(dim(Hcons), dim(Hcons))), Hcons => hilbert_space(f, 1:2, NumberConservation(1)))
end
