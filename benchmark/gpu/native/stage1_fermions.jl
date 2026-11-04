@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))

@testset "Stage 1: fermions without a constraint" begin
    @fermions f
    H = hilbert_space(f, 1:6)
    hop = sum(f[i]' * f[i+1] + hc for i in 1:5)
    op = hop + 2.0 * sum(f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:5) + 0.3 * f[1]' * f[3]' + hc + 1.5
    check(op, H)
    check(op, H; T=Float32)
    check(1im * f[1]' * f[2] + hc, H; T=ComplexF64)
    check(f[2]' * f[4] * f[5], H)
    check(f[3], H)
    check(2, H)
    @test_throws ArgumentError device_matrix(1im * f[1], H, backend, Float64)
end
