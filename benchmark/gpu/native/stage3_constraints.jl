@isdefined(check) || include(joinpath(@__DIR__, "common.jl"))
using FermionicHilbertSpaces: FixedNumberFockState

@testset "Stage 3: parity, sectors, FixedNumberFockState, Dict indices" begin
    @fermions f
    op = sum(f[i]' * f[i+1] + 0.4 * f[i]' * f[i+1]' + hc for i in 1:5) + sum(f[i]' * f[i] for i in 1:6)
    # A single parity sector
    check(op, hilbert_space(f, 1:6, ParityConservation(1)))
    # Both parity sectors, in a space ordered by sector
    check(op, hilbert_space(f, 1:6, ParityConservation()))
    # Several number sectors
    hop = sum(f[i]' * f[i+1] + hc for i in 1:5)
    check(hop, hilbert_space(f, 1:6, NumberConservation(1:3)))
    # Listed states
    H = hilbert_space(f, 1:4)
    check(f[1]' * f[2] + hc, constrain_space(H, collect(basisstates(H))[[1, 2, 3, 5, 9]]); projection=true)
    # FixedNumberFockState states, indexed by a Dict of states
    states = FixedNumberFockState{2}.(basisstates(hilbert_space(f, 1:6, NumberConservation(2))))
    Hx = hilbert_space(f, 1:6, states)
    check(hop + 0.5 * f[1]' * f[2]' * f[4] * f[3] + hc, Hx)
    check(f[1]' * f[2]' * f[3], Hx; projection=true)
end
