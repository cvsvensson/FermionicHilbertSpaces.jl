@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))

@testset "Stage 5: spins" begin
    @spin S 1
    check(S[:+]^2 + S[:z] + hc, hilbert_space(S))
    @spins s 1 // 2
    H = hilbert_space(s, 1:6)
    heisenberg = sum(s[i][:x] * s[i+1][:x] + s[i][:y] * s[i+1][:y] + 0.7 * s[i][:z] * s[i+1][:z] for i in 1:5)
    check(heisenberg + 0.3 * s[1][:x], H; T=ComplexF64)
    check(sum(s[i][:+] * s[i+1][:-] + hc for i in 1:5) + s[2][:z], H)
    # Three spins up
    Hm = hilbert_space(s, 1:6, AdditiveConstraint(3, st -> st.m + 1 // 2))
    check(sum(s[i][:+] * s[i+1][:-] + hc for i in 1:5) + sum(s[i][:z] * s[i+1][:z] for i in 1:5), Hm)
end
