@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))

@testset "Stage 6: fermions ⊗ spins ⊗ bosons" begin
    @fermions f
    @spin S 1 // 2
    @boson b
    Hf = hilbert_space(f, 1:3)
    Hs = hilbert_space(S)
    Hb = hilbert_space(b, 4)
    op = sum(f[i]' * f[i + 1] + hc for i in 1:2) + f[1]' * f[1] * S[:z] + 0.5 * (b + b') * S[:x] +
        0.2 * f[2]' * f[2] * b' * b + f[1] * S[:+] + hc + 0.4
    check(op, tensor_product(Hf, Hs, Hb); T = ComplexF64)
    # A constraint on the product
    conserving = sum(f[i]' * f[i + 1] + hc for i in 1:2) + f[1]' * f[1] * S[:z] + 0.5 * (b + b') * S[:x]
    check(conserving, tensor_product(Hf, Hs, Hb; constraint = NumberConservation(1, [Hf])); T = ComplexF64)
    # A constrained group inside the product
    Hf2 = hilbert_space(f, 1:3, NumberConservation(2))
    check(conserving, tensor_product(Hf2, Hs, Hb); T = ComplexF64)
end
