struct SpinField
    name::Symbol
end
Base.:(==)(a::SpinField, b::SpinField) = a.name == b.name
Base.hash(b::SpinField, h::UInt) = hash(b.name, h)
Base.show(io::IO, b::SpinField) = print(io, "SpinField(", b.name, ")")

struct SymbolicSpinBasis{L,P}
    label::L
    field::P
end
SymbolicSpinBasis(label::L) where L = SymbolicSpinBasis(label, nothing)
Base.:(==)(a::SymbolicSpinBasis, b::SymbolicSpinBasis) = a.label == b.label && a.field == b.field
Base.getindex(s::SymbolicSpinBasis, op) = SpinSym(op, s)
Base.hash(x::SymbolicSpinBasis, h::UInt) = hash(x.field, hash(x.label, h))
label(S::SymbolicSpinBasis) = S.label
Base.getindex(s::SpinField, i) = SymbolicSpinBasis(i, s)

function Base.show(io::IO, S::SymbolicSpinBasis)
    if S.field isa Nothing
        print(io, "SpinBasis(", S.label, ")")
    else
        print(io, "SpinBasis(", S.field.name, "[", S.label, "])")
    end
end

"""
    @spin s

Create a symbolic spin basis `s` for constructing spin operators such as
`s[:z]`, `s[:+]`, and `s[:-]`.
"""
macro spin(x)
    Expr(:block, :($(esc(x)) = SymbolicSpinBasis($(Expr(:quote, x)))),
        :($(esc(x))))
end

"""
    @spins s 

Create a spin basis `s`. Index into it to get site-specific
spin bases: `s[1][:z]` gives the z-operator on site 1.
"""
macro spins(name)
    return Expr(:block, :($(esc(name)) = SpinField($(Expr(:quote, name)))), :($(esc(name))))
end


struct SpinState{M} <: AbstractBasisState
    m::M
end
Base.:(==)(a::SpinState, b::SpinState) = a.m == b.m
Base.isless(a::SpinState, b::SpinState) = a.m < b.m
Base.hash(x::SpinState, h::UInt) = hash(x.m, h)


struct SpinSpace{J,M,S} <: AbstractAtomicHilbertSpace{SpinState{M}}
    basisstates::Vector{SpinState{M}}
    sym::S
    state_index::Dict{SpinState{M},Int}
    function SpinSpace{J}(sym::S) where {J,S<:SymbolicSpinBasis}
        states = spin_basisstates(Val(J))
        state_index = Dict(s => i for (i, s) in enumerate(states))
        new{J,typeof(J),S}(states, sym, state_index)
    end
end
SpinSpace{J}(label) where J = SpinSpace{J}(SymbolicSpinBasis(label))
basisstates(H::SpinSpace) = H.basisstates
basisstate(n::Int, H::SpinSpace) = H.basisstates[n]
dim(H::SpinSpace) = length(H.basisstates)
state_index(s::SpinState{S}, ::SpinSpace{J,S}) where {J,S} = Int(s.m + J + 1)
cluster_id(H::SpinSpace) = symbolic_group(H.sym)
atomic_id(H::SpinSpace) = symbolic_group(H.sym)

hilbert_space(sym::SymbolicSpinBasis, J) = SpinSpace{J}(sym)
hilbert_space(sym::SpinField, labels, J, constraint=NoSymmetry()) = tensor_product([hilbert_space(sym[l], J) for l in labels], constraint)
Base.:(==)(a::SpinSpace, b::SpinSpace) = a === b || (a.sym == b.sym && a.basisstates == b.basisstates)
Base.hash(x::SpinSpace, h::UInt) = hash(x.sym, hash(x.basisstates, h))

function spin_basisstates(::Val{J}) where {J}
    states = [SpinState{typeof(J)}(i - J) for i in 0:2J]
    return states
end
spin_basisstates(j) = spin_basisstates(Val(j))

function operators(H::SpinSpace{J,S}) where {J,S}
    Splus = spzeros(Float64, dim(H), dim(H))
    Sminus = spzeros(Float64, dim(H), dim(H))
    Sz = spzeros(Float64, dim(H), dim(H))
    for state in H.basisstates
        m = state.m
        i = state_index(state, H)
        if m < J
            j = state_index(SpinState(m + 1), H)
            Splus[j, i] = sqrt(J * (J + 1) - m * (m + 1))
        end
        if m > -J
            j = state_index(SpinState(m - 1), H)
            Sminus[j, i] = sqrt(J * (J + 1) - m * (m - 1))
        end
        Sz[i, i] = m
    end
    return Dict(:+ => Splus, :- => Sminus, :Z => Sz, :X => (Splus + Sminus) / 2, :Y => (Splus - Sminus) / (2im))
end

@testitem "Spin" begin
    using FermionicHilbertSpaces: spin_basisstates, SpinSpace, SpinState, operators
    using LinearAlgebra
    @test spin_basisstates(1 // 2) == [SpinState(-1 // 2), SpinState(1 // 2)]
    @test spin_basisstates(1) == [SpinState(-1), SpinState(0), SpinState(1)]

    @spin s
    H = hilbert_space(s, 1 // 2)
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
    H1, H2 = [SpinSpace{1 // 2}(k) for k in 1:2]
    P = tensor_product(H1, H2)
    @test partial_trace(1.0 * I(dim(P)), P => H1) ≈ dim(H2) * I(dim(H1))

    ops1 = operators(H1)
    @test all(partial_trace(embed(op, H1 => P), P => H1) ≈ dim(H2) * op for op in values(ops1))

    @fermions f
    Hf = hilbert_space(f, 1:2)
    mf = rand(dim(Hf), dim(Hf))
    Pf = tensor_product(Hf, P)
    @test partial_trace(embed(mf, Hf => Pf), Pf => Hf) ≈ dim(P) * mf

    mf1 = rand(2, 2)
    Hf1 = hilbert_space(f, 1:1)
    @test partial_trace(embed(mf1, Hf1 => Pf), Pf => Hf) ≈ dim(P) * embed(mf1, Hf1 => Hf)

end

struct SpinSym{B} <: AbstractSym
    op::Symbol
    basis::B
    function SpinSym(op, basis::B) where {B}
        if op in _spin_x_aliases
            return (new{B}(:+, basis) + new{B}(:-, basis)) / 2
        elseif op in _spin_y_aliases
            return (new{B}(:+, basis) - new{B}(:-, basis)) / (2im)
        else
            return new{B}(_canonical_spin_alias(op), basis)
        end
    end
end
const _spin_x_aliases = Set((:x, :X))
const _spin_y_aliases = Set((:y, :Y))
const _spin_z_aliases = Set((:z, :Z))
const _spin_plus_aliases = Set((:+, :plus, :p))
const _spin_minus_aliases = Set((:-, :minus, :m))
function _canonical_spin_alias(op)
    if op in _spin_z_aliases
        :z
    elseif op in _spin_plus_aliases
        :+
    elseif op in _spin_minus_aliases
        :-
    else
        throw(ArgumentError("Invalid spin operator symbol: $op."))
    end
end

function Base.show(io::IO, x::SpinSym)
    if x.basis.field isa Nothing
        print(io, x.basis.label, "[:$(x.op)]")
    else
        print(io, x.basis.field.name, "[", x.basis.label, "][:$(x.op)]")
    end
end

Base.:(==)(a::SpinSym, b::SpinSym) = a.op == b.op && a.basis == b.basis
Base.hash(a::SpinSym, h::UInt) = hash(a.op, hash(a.basis, h))
symbolic_group(f::SpinSym) = symbolic_group(f.basis)
symbolic_group(f::SymbolicSpinBasis) = (SymbolicSpinBasis, f.field, f.label)
atomic_id(f::SpinSym) = (SpinSpace, f.basis.field, f.basis.label)

mat_eltype(::Type{<:SpinSym}) = Float64

function Base.adjoint(x::SpinSym)
    if x.op == :+
        SpinSym(:-, x.basis)
    elseif x.op == :-
        SpinSym(:+, x.basis)
    elseif x.op == :z
        x
    else
        throw(ArgumentError("Invalid spin operator symbol: $(x.op)."))
    end
end
_spin_name(a::SymbolicSpinBasis) = a.field isa Nothing ? a.label : Symbol(a.field.name, a.label)
function NonCommutativeProducts.mul_effect(a::SpinSym, b::SpinSym)
    if _spin_name(a.basis) > _spin_name(b.basis)
        return Swap(1)
    end
    if _spin_name(a.basis) < _spin_name(b.basis)
        return nothing
    end

    # Commutation relations: [Sz, S+] = S+, [Sz, S-] = -S-, [S+, S-] = 2Sz
    # Canonical order: :z < :+ < :-
    if a.op == :+ && b.op == :z
        # S+ Sz = Sz S+ - S+
        return AddTerms((Swap(1), -SpinSym(:+, a.basis)))
    elseif a.op == :- && b.op == :z
        # S- Sz = Sz S- + S-
        return AddTerms((Swap(1), SpinSym(:-, a.basis)))
    elseif a.op == :- && b.op == :+
        # S- S+ = S+ S- - 2Sz
        return AddTerms((Swap(1), -2 * SpinSym(:z, a.basis)))
    else
        return nothing
    end
end

@nc SpinSym
NonCommutativeProducts.@commutative FermionSym SpinSym
NonCommutativeProducts.@commutative BosonSym SpinSym
NonCommutativeProducts.@commutative MajoranaSym SpinSym

"""
    apply_local_operator(op::SpinSym, state::SpinState) -> (newstate, amplitude)

Apply a single spin operator to a spin state. Returns (newstate, amplitude) where amplitude is 0 if the operation is not allowed, and a nonzero value otherwise.
"""
function apply_local_operator(op::SpinSym, state::SpinState, ::Val{J}) where J
    m = state.m
    if op.op == :z
        # S_z |m⟩ = m |m⟩
        return (state, m)
    elseif op.op == :+
        # S_+ |m⟩ = √(J(J+1) - m(m+1)) |m+1⟩
        if m < J
            amplitude = sqrt(J * (J + 1) - m * (m + 1))
            newstate = SpinState(m + 1)
            return (newstate, amplitude)
        else
            return (state, 0)
        end
    elseif op.op == :-
        # S_- |m⟩ = √(J(J+1) - m(m-1)) |m-1⟩
        if m > -J
            amplitude = sqrt(J * (J + 1) - m * (m - 1))
            newstate = SpinState(m - 1)
            return (newstate, amplitude)
        else
            return (state, 0)
        end
    else
        throw(ArgumentError("Invalid spin operator symbol: $(op.op)."))
    end
end

"""
    apply_local_operators(factors::Vector{SpinSym}, state::SpinState, H::SpinSpace) -> (newstate, amplitude)

Apply a sequence of spin operators (product) to a spin state. Operators are applied in reverse order (right-to-left, as in operator composition). Returns (newstate, amplitude) or (state, 0) if any step fails.
"""
function apply_local_operators(op, state::SpinState, space::SpinSpace{J}, precomp) where J
    newstate = state
    amplitude = op.coeff * one(typeof(sqrt(J * (J + 1))))  # Start with 1.0 to handle mixed numeric types
    # Apply factors in reverse order (from right to left)
    for factor in reverse(op.factors)
        newstate, factor_amp = apply_local_operator(factor, newstate, Val(J))
        if iszero(factor_amp)
            return (state,), (zero(amplitude),)
        end
        amplitude *= factor_amp
    end
    return (newstate,), (amplitude,)
end

@testitem "SpinSym" begin
    @spin S1
    @spin S2
    @fermions f
    @majoranas γ
    @boson b

    Sz1 = S1[:z]
    Sp1 = S1[:+]
    Sm1 = S1[:-]
    Sx1 = S1[:x]
    Sy1 = S1[:y]

    Sz2 = S2[:z]
    Sp2 = S2[:+]
    Sm2 = S2[:-]

    @test 1 * Sz1 == Sz1
    @test 1 * Sz1 + 0 == Sz1
    @test hash(Sz1) == hash(1 * Sz1) == hash(1 * Sz1 + 0)

    # Test canonical commutation relations
    # [Sz, S+] = S+
    @test Sz1 * Sp1 - Sp1 * Sz1 == Sp1
    # [Sz, S-] = -S-
    @test Sz1 * Sm1 - Sm1 * Sz1 == -Sm1
    # [S+, S-] = 2Sz
    @test Sp1 * Sm1 - Sm1 * Sp1 == 2 * Sz1

    # Test x and y operators expand correctly
    @test Sx1 == (Sp1 + Sm1) / 2
    @test Sy1 == (Sp1 - Sm1) / (2im)

    # Spins at different sites commute
    @test Sz1 * Sz2 - Sz2 * Sz1 == 0
    @test Sp1 * Sp2 - Sp2 * Sp1 == 0
    @test Sp1 * Sm2 - Sm2 * Sp1 == 0

    # Spins commute with fermions
    @test Sz1 * f[1] - f[1] * Sz1 == 0
    @test Sp1 * f[1] - f[1] * Sp1 == 0
    @test Sz1 * f[1]' - f[1]' * Sz1 == 0
    @test Sp1 * f[1]' - f[1]' * Sp1 == 0

    # Spins commute with majoranas
    @test Sz1 * γ[1] - γ[1] * Sz1 == 0
    @test Sp1 * γ[1] - γ[1] * Sp1 == 0

    # Spins commute with bosons
    @test Sz1 * b - b * Sz1 == 0
    @test Sp1 * b - b * Sp1 == 0
    @test Sz1 * b' - b' * Sz1 == 0
    @test Sp1 * b' - b' * Sp1 == 0

    # Test adjoint
    @test Sp1' == Sm1
    @test Sm1' == Sp1
    @test Sz1' == Sz1

    @test iszero(Sz1 - Sz1)
    @test iszero(2 * Sz1 - 2 * Sz1)
    @test iszero(0 * Sz1)
    @test iszero(Sz1 * 0)
    @test iszero(0 * (Sp1 + Sm1))
    @test iszero((Sp1 + Sm1) * 0)

    # Test algebraic properties
    @test 1 + (Sp1 + Sm1) == 1 + Sp1 + Sm1 == Sp1 + Sm1 + 1 == Sp1 + 1 + Sm1
end
