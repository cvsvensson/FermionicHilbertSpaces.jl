@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))

@testset "Stage 2: fermions with NumberConservation" begin
    @fermions f
    hop(N) = sum(f[i]' * f[i + 1] + hc for i in 1:(N - 1))
    # A lookup table index
    H = hilbert_space(f, 1:6, NumberConservation(3))
    op = hop(6) + 2.0 * sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:5) + 0.5
    check(op, H)
    # An index keyed by the position in the parent space, from a Dict
    H = hilbert_space(f, 1:12, NumberConservation(1))
    check(hop(12) + 0.1 * f[1]' * f[12], H)
    # Leaving the space
    check(f[1], H; projection = true)
    # Asynchronous backends throw when the result is used.
    @test_throws Exception Array(representation(f[1], H, DeviceDenseRepr(backend)))
end
