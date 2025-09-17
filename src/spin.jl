struct SpinState{J,S} <: AbstractBasisState
    m::S
end
SpinState{J}(m::M) where {J,M} = SpinState{J,M}(m)
_labeltype(::Type{<:SpinState{J,S}}) where {J,S} = S

struct SpinSpace{J,S,L} <: AbstractHilbertSpace
    basisstates::Vector{SpinState{J,S}}
    label::L
    function SpinSpace{J}(label::L=uuid4()) where {J,L}
        states = spin_basisstates(J)
        new{J,_labeltype(eltype(states)),L}(states, label)
    end
end
basisstates(H::SpinSpace) = H.basisstates
basisstate(n::Int, H::SpinSpace) = H.basisstates[n]
Base.keys(H::SpinSpace) = (H.label,)
dim(H::SpinSpace) = length(H.basisstates)
state_index(s::SpinState{J,S}, ::SpinSpace{J,S}) where {J,S} = Int(s.m + J + 1)

function spin_basisstates(j)
    states = [SpinState{j}(i - j) for i in 0:2j]
    return states
end

function operators(H::SpinSpace{J,S}) where {J,S}
    Splus = zeros(Float64, dim(H), dim(H))
    Sminus = zeros(Float64, dim(H), dim(H))
    Sz = zeros(Float64, dim(H), dim(H))
    for state in H.basisstates
        m = state.m
        i = state_index(state, H)
        if m < J
            j = state_index(SpinState{J}(m + 1), H)
            Splus[j, i] = sqrt(J * (J + 1) - m * (m + 1))
        end
        if m > -J
            j = state_index(SpinState{J}(m - 1), H)
            Sminus[j, i] = sqrt(J * (J + 1) - m * (m - 1))
        end
        Sz[i, i] = m
    end
    return Dict(:+ => Splus, :- => Sminus, :Z => Sz, :X => (Splus + Sminus) / 2, :Y => (Splus - Sminus) / (2im))
end

@testitem "Spin" begin
    using FermionicHilbertSpaces: spin_basisstates, SpinSpace, SpinState, operators
    using LinearAlgebra
    @test spin_basisstates(1 // 2) == [SpinState{1 // 2}(-1 // 2), SpinState{1 // 2}(1 // 2)]
    @test spin_basisstates(1) == [SpinState{1}(-1), SpinState{1}(0), SpinState{1}(1)]

    H = SpinSpace{1 // 2}()
    S = operators(H)
    @test S[:+] == [0 0; 1 0]
    @test S[:-] == [0 1; 0 0]
    @test S[:Z] == [-1//2 0; 0 1//2]
    @test S[:X] == [0 1//2; 1//2 0]
    @test S[:Y] == [0 im//2; -im//2 0]
    # test pauli algebra
    @test S[:X] * S[:Y] - S[:Y] * S[:X] ≈ im * S[:Z]
    @test S[:Y] * S[:Z] - S[:Z] * S[:Y] ≈ im * S[:X]
    @test S[:Z] * S[:X] - S[:X] * S[:Z] ≈ im * S[:Y]
    H1, H2 = [SpinSpace{1 // 2}() for k in 1:2]
    P = tensor_product(H1, H2)
    @test partial_trace(1.0 * I(dim(P)), P => H1) ≈ dim(H2) * I(dim(H1))

    ops1 = operators(H1)
    @test all(partial_trace(embed(op, H1 => P), P => H1) ≈ dim(H2) * op for op in values(ops1))

    Hf = hilbert_space(1:2)
    mf = rand(dim(Hf), dim(Hf))
    Pf = tensor_product(Hf, P)
    @test partial_trace(embed(mf, Hf => Pf), Pf => Hf) ≈ dim(P) * mf

    mf1 = rand(2, 2)
    Hf1 = hilbert_space(1:1)
    @test partial_trace(embed(mf1, Hf1 => Pf), Pf => Hf) ≈ dim(P) * embed(mf1, Hf1 => Hf)

    mp = rand(4, 4)
    @test embed(mp, P => Pf) ≈ extend(mp, P => Hf)
end