# Runs on JLArrays, a CPU implementation of the GPU array interface that disallows scalar indexing.
# Not part of the main suite: `julia --project=test/gpu test/gpu/runtests.jl`.
using FermionicHilbertSpaces, JLArrays, Test
using FermionicHilbertSpaces: GatherPartialTraceAlg
JLArrays.allowscalar(false)

@testset "partial_trace of GPU arrays" begin
    alg = GatherPartialTraceAlg()
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

@testset "representation as GPU arrays" begin
    @fermions f
    @boson b
    @spin s 1 // 2
    hopping = sum(f[i]' * f[i + 1] + hc for i in 1:3) + sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:3)
    mixed = f[1]' * f[2] * s[:+] + 1im * b' * s[:-] + hc + f[1]' * f[1] * b' * b + s[:z] + 2
    Hmixed = tensor_product(hilbert_space(f, 1:2, NumberConservation(1)), hilbert_space(s), hilbert_space(b, 3))
    cases = (
        (hopping, hilbert_space(f, 1:4)),
        (hopping, hilbert_space(f, 1:4, NumberConservation(2))),
        (b' * b + b + b', hilbert_space(b, 4)),
        (s[:+] + s[:-] + 0.5 * s[:z], hilbert_space(s)),
        (mixed, Hmixed),
    )
    for (op, H) in cases, format in (JLArray, JLSparseMatrixCSC, JLSparseMatrixCSR{ComplexF32})
        M = representation(op, H, format)
        @test M isa format
        @test collect(M) ≈ representation(op, H, Matrix)
    end
    @test_throws ArgumentError representation(f[1], hilbert_space(f, 1:4, NumberConservation(2)), JLArray)
    @majoranas γ
    @test_throws ArgumentError representation(1im * γ[1] * γ[2], hilbert_space(γ, 1:4), JLArray)
end
