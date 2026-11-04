@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))

@testset "Stage 4: bosons" begin
    @bosons b
    @boson a
    # A single mode
    check(a' * a' * a + 0.5 * a + hc, hilbert_space(a, 5))
    # Several modes
    H = hilbert_space(b, 1:3, 4)
    op = sum(b[i]' * b[i+1] + hc for i in 1:2) + 0.5 * sum(b[i]' * b[i]' * b[i] * b[i] for i in 1:3) + 0.2 * b[1]' + 1
    check(op, H)
    # A conserved total number
    check(sum(b[i]' * b[i+1] + hc for i in 1:2) + b[2]' * b[2], hilbert_space(b, 1:3, 4, NumberConservation(3)))
end
