struct SpinField{J}
    name::Symbol
    spin::J
end
SpinField(name::Symbol) = SpinField(name, nothing)
Base.:(==)(a::SpinField, b::SpinField) = a.name == b.name
Base.hash(b::SpinField, h::UInt) = hash(b.name, h)
Base.show(io::IO, b::SpinField) = print(io, "SpinField(", b.name, ")")

struct SymbolicSpinBasis{L,P,J,I}
    label::L
    field::P
    spin::J
    id::I
end
SymbolicSpinBasis(label::L) where L = SymbolicSpinBasis(label, nothing, nothing, label)
Base.:(==)(a::SymbolicSpinBasis, b::SymbolicSpinBasis) = a.label == b.label && a.field == b.field && a.spin == b.spin && a.id == b.id
Base.getindex(s::SymbolicSpinBasis, op) = SpinSym(op, s)
Base.hash(x::SymbolicSpinBasis, h::UInt) = hash(x.id, hash(x.spin, hash(x.field, hash(x.label, h))))
label(S::SymbolicSpinBasis) = S.label
Base.getindex(s::SpinField, i) = SymbolicSpinBasis(i, s, s.spin, (s.name, i, s.spin))
atomic_factors(s::SymbolicSpinBasis) = (s,)
atomic_id(s::SymbolicSpinBasis) = s.id
symbolic_id(s::SymbolicSpinBasis) = s.id
change_id(s::SymbolicSpinBasis, newid) = SymbolicSpinBasis(s.label, s.field, s.spin, newid)
tags(s::SymbolicSpinBasis) = tags(symbolic_id(s))
add_tag(s::SymbolicSpinBasis, tag::Symbol) = change_id(s, _tag_id(symbolic_id(s), tag))

function Base.show(io::IO, S::SymbolicSpinBasis)
    spin_suffix = S.spin isa Nothing ? "" : ", spin=$(S.spin)"
    if S.field isa Nothing
        print(io, "SpinBasis(", S.label, spin_suffix, ")")
    else
        print(io, "SpinBasis(", S.field.name, "[", S.label, "]", spin_suffix, ")")
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
macro spin(x, spin)
    Expr(:block, :($(esc(x)) = SymbolicSpinBasis($(Expr(:quote, x)), nothing, $(esc(spin)), ($(Expr(:quote, x)), $(esc(spin))))),
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
macro spins(name, spin)
    return Expr(:block, :($(esc(name)) = SpinField($(Expr(:quote, name)), $(esc(spin)))), :($(esc(name))))
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
group_id(H::SpinSpace) = symbolic_group(H.sym)
atomic_id(H::SpinSpace) = symbolic_group(H.sym)

hilbert_space(sym::SymbolicSpinBasis{<:Any,<:Any,J}) where J<:Union{Int,Rational} = SpinSpace{sym.spin}(sym)
hilbert_space(sym::SymbolicSpinBasis{<:Any,<:Any,<:Nothing}, J) = SpinSpace{J}(sym)
hilbert_space(sym::SpinField{J}, labels, constraint=NoSymmetry()) where J<:Union{Int,Rational} = tensor_product(map(l -> hilbert_space(sym[l]), labels); constraint)
hilbert_space(sym::SpinField{Nothing}, labels, J, constraint=NoSymmetry()) = tensor_product(map(l -> hilbert_space(sym[l], J), labels); constraint)
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
    exponent::Int
    function SpinSym(op, basis::B, exponent::Integer=1) where {B}
        exponent < 0 && throw(ArgumentError("Spin operator exponent must be >= 0, got $exponent."))
        exponent == 0 && return 1 // 1
        if op in _spin_x_aliases
            exponent == 1 || throw(ArgumentError("Exponentiation is only supported for spin raising/lowering operators."))
            return (new{B}(:+, basis, 1) + new{B}(:-, basis, 1)) / 2
        elseif op in _spin_y_aliases
            exponent == 1 || throw(ArgumentError("Exponentiation is only supported for spin raising/lowering operators."))
            return (new{B}(:+, basis, 1) - new{B}(:-, basis, 1)) / (2im)
        elseif op in _spin_identity_aliases
            return 1 // 1
        else
            canonical = _canonical_spin_alias(op)
            if !(basis.spin isa Nothing) && canonical in (:+, :-) && exponent > Int(2 * basis.spin)
                return 0 // 1
            end
            return new{B}(canonical, basis, Int(exponent))
        end
    end
end
const _spin_x_aliases = Set((:x, :X, 1))
const _spin_y_aliases = Set((:y, :Y, 2))
const _spin_z_aliases = Set((:z, :Z, 3))
const _spin_identity_aliases = Set((:I, 0))
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
    if x.exponent != 1
        print(io, "^", x.exponent)
    end
end

Base.:(==)(a::SpinSym, b::SpinSym) = a.op == b.op && a.basis == b.basis && a.exponent == b.exponent
Base.hash(a::SpinSym, h::UInt) = hash(a.exponent, hash(a.op, hash(a.basis, h)))
symbolic_group(f::SpinSym) = symbolic_group(f.basis)
symbolic_group(f::SymbolicSpinBasis) = symbolic_id(f)
symbolic_basis(f::SpinSym) = f.basis
change_basis(f::SpinSym, newbasis) = SpinSym(f.op, newbasis, f.exponent)
atomic_id(f::SpinSym) = atomic_id(f.basis)

mat_eltype(::Type{<:SpinSym}) = Float64

function Base.adjoint(x::SpinSym)
    if x.op == :+
        SpinSym(:-, x.basis, x.exponent)
    elseif x.op == :-
        SpinSym(:+, x.basis, x.exponent)
    elseif x.op == :z
        x
    else
        throw(ArgumentError("Invalid spin operator symbol: $(x.op)."))
    end
end
_spin_name(a::SymbolicSpinBasis) = a.field isa Nothing ? a.label : Symbol(a.field.name, a.label)

function _spin_commutator_term(a::SpinSym, b::SpinSym)
    if a.op == :+ && b.op == :z
        return (-1, SpinSym(:+, a.basis))
    elseif a.op == :- && b.op == :z
        return (1, SpinSym(:-, a.basis))
    elseif a.op == :- && b.op == :+
        return (-2, SpinSym(:z, a.basis))
    else
        return nothing
    end
end

_factor_or_empty(x) = isone(x) ? Any[] : [x]

function NonCommutativeProducts.mul_effect(a::S, b::S) where S<:SpinSym
    if _spin_name(a.basis) > _spin_name(b.basis)
        return Swap(1)
    end
    if _spin_name(a.basis) < _spin_name(b.basis)
        return nothing
    end

    if a.op == b.op && a.op in (:+, :-, :z)
        exponent = a.exponent + b.exponent
        if a.op in _spin_z_aliases
            if a.basis.spin == 1 // 2
                div, rem = divrem(exponent, 2)
                return (1 // 4)^div * (rem == 0 ? 1 : SpinSym(a.op, a.basis, rem))
            elseif a.basis.spin == 1
                return iseven(exponent) ? SpinSym(a.op, a.basis, 2) : SpinSym(a.op, a.basis, 1)
            end
        end
        return SpinSym(a.op, a.basis, exponent)
    end

    # Spin-aware identity: S+ S- = J(J+1) - Sz^2 + Sz
    if a.op == :+ && b.op == :- && !(a.basis.spin isa Nothing)
        J = a.basis.spin
        left = SpinSym(a.op, a.basis, a.exponent - 1)
        right = SpinSym(b.op, b.basis, b.exponent - 1)

        left_factors = a.exponent == 1 ? Any[] : [left]
        right_factors = b.exponent == 1 ? Any[] : [right]

        const_term = J * (J + 1)
        term_const = if iszero(const_term)
            0
        elseif length(left_factors) == 0 && length(right_factors) == 0
            const_term
        else
            NCMul(const_term, S[left_factors..., right_factors...])
        end
        szsquare = if J == 1 // 2
            1 // 4
        else
            SpinSym(:z, a.basis, 2)
        end
        term_sz2 = szsquare isa SpinSym ? NCMul(-1 // 1, S[left_factors..., SpinSym(:z, a.basis, 2), right_factors...]) : NCMul(-1 * szsquare // 1, S[left_factors..., right_factors...])
        term_sz = NCMul(1 // 1, S[left_factors..., SpinSym(:z, a.basis), right_factors...])
        return AddTerms((term_const, term_sz2, term_sz))
    end

    if a.basis.spin == 1 // 2 && a.op == :z
        # apply Sz*S+ = S+/2 and related identities for spin-1/2
        if b.op == :+
            return NCMul((1 // 2)^(a.exponent), S[SpinSym(:+, a.basis, b.exponent)])
        elseif b.op == :-
            return NCMul((-1 // 2)^(a.exponent), S[SpinSym(:-, a.basis, b.exponent)])
        end
    end

    comm = _spin_commutator_term(a, b)
    if isnothing(comm)
        return nothing
    end

    comm_coeff, comm_middle_factor = comm

    # For A^N B^M (out-of-order), rewrite only the middle pair:
    # A^(N-1) * (B*A + [A,B]) * B^(M-1)
    left = SpinSym(a.op, a.basis, a.exponent - 1)
    right = SpinSym(b.op, b.basis, b.exponent - 1)
    swapped_factors, comm_factors = if a.exponent == 1 && b.exponent == 1
        S[SpinSym(b.op, b.basis), SpinSym(a.op, a.basis)], S[comm_middle_factor]
    elseif a.exponent == 1
        S[SpinSym(b.op, b.basis), right], S[comm_middle_factor, right]
    elseif b.exponent == 1
        S[left, SpinSym(a.op, a.basis)], S[left, comm_middle_factor]
    else
        S[left, SpinSym(b.op, b.basis), right], S[left, comm_middle_factor, right]
    end

    swapped_term = NCMul(1 // 1, swapped_factors)
    comm_term = NCMul(comm_coeff // 1, comm_factors)
    return AddTerms((swapped_term, comm_term))
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
    newstate = state
    T = typeof(sqrt(J * (J + 1)) * m)
    amplitude = one(T)
    for _ in 1:op.exponent
        m = newstate.m
        if op.op == :z
            # S_z |m⟩ = m |m⟩
            amplitude *= m
        elseif op.op == :+
            # S_+ |m⟩ = √(J(J+1) - m(m+1)) |m+1⟩
            if m < J
                factor = sqrt(J * (J + 1) - m * (m + 1))
                newstate = SpinState(m + 1)
                amplitude *= factor
            else
                return (state, zero(T))
            end
        elseif op.op == :-
            # S_- |m⟩ = √(J(J+1) - m(m-1)) |m-1⟩
            if m > -J
                factor = sqrt(J * (J + 1) - m * (m - 1))
                newstate = SpinState(m - 1)
                amplitude *= factor
            else
                return (state, zero(T))
            end
        else
            throw(ArgumentError("Invalid spin operator symbol: $(op.op)."))
        end
    end
    return (newstate, amplitude)
end

"""
    apply_local_operators(factors::Vector{SpinSym}, state::SpinState, H::SpinSpace) -> (newstate, amplitude)

Apply a sequence of spin operators (product) to a spin state. Operators are applied in reverse order (right-to-left, as in operator composition). Returns (newstate, amplitude) or (state, 0) if any step fails.
"""
function apply_local_operators(op, state::SpinState, space::SpinSpace{J}, precomp) where J
    newstate = state
    amplitude = op.coeff * one(typeof(sqrt(J * (J + 1))))
    # Apply factors in reverse order (from right to left)
    for factor in reverse(op.factors)
        newstate, factor_amp = apply_local_operator(factor, newstate, Val{J}())
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

    # Test with explicit spin
    @spin Shalf 1 // 2
    @test iszero(Shalf[:+]^2)
    @test iszero(Shalf[:-]^2)
    @test Shalf[:+] * Shalf[:-] == Shalf[:z] + 1 // 2

    @spin Sone 1
    Spo = Sone[:+]
    @test iszero(Sone[:+]^3)
    @test Sone[:z]^3 == Sone[:z]

end

@testitem "Spin matrix reps" begin
    @spin s
    H = hilbert_space(s, 1 // 2)
    S_mat = Dict(sym => matrix_representation(s[sym], H) for sym in (:x, :y, :z, :+, :-))
    @test S_mat[:+] == [0 0; 1 0]
    @test S_mat[:-] == [0 1; 0 0]
    @test S_mat[:z] == [-1//2 0; 0 1//2]
    @test S_mat[:x] == [0 1//2; 1//2 0]
    @test S_mat[:y] == [0 im//2; -im//2 0]
end

@testitem "Spin: spin-aware vs not" begin
    for J in (1 // 2, 1, 3 // 2)
        @spin s J
        H = hilbert_space(s)
        @test dim(H) == 2J + 1
        @test iszero(s[:+]^Int(2J + 1))
        @test iszero(s[:-]^Int(2J + 1))
        syms = (:+, :-, :z, :x, :y, :I)
        smat = Dict([sym => matrix_representation(s[sym], H) for sym in syms])
        @test all(smat[sym1] * smat[sym2] ≈ matrix_representation(s[sym1] * s[sym2], H) for sym1 in syms for sym2 in syms)
    end
end
