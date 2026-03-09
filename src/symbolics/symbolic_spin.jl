
struct SymbolicSpinBasis
    name::Symbol
end
Base.hash(x::SymbolicSpinBasis, h::UInt) = hash(x.name, h)

macro spins(x)
    Expr(:block, :($(esc(x)) = SymbolicSpinBasis($(Expr(:quote, x)))),
        :($(esc(x))))
end
Base.:(==)(a::SymbolicSpinBasis, b::SymbolicSpinBasis) = a.name == b.name
Base.getindex(::SymbolicSpinBasis, i) = op -> SpinSym(i, op)

struct SpinSym{L}
    label::L
    op::Symbol
    function SpinSym(label::L, op) where L
        if op in _spin_x_aliases
            return (new{L}(label, :+) + new{L}(label, :-)) / 2
        elseif op in _spin_y_aliases
            return (new{L}(label, :+) - new{L}(label, :-)) / (2im)
        else
            return new{L}(label, _canonical_spin_alias(op))
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

Base.show(io::IO, x::SpinSym) = print(io, "S", x.op, "[$(x.label)]")

Base.:(==)(a::SpinSym, b::SpinSym) = a.op == b.op && a.label == b.label
Base.hash(a::SpinSym, h::UInt) = hash(a.op, hash(a.label, h))

function Base.adjoint(x::SpinSym)
    if x.op == :+
        SpinSym(x.label, :-)
    elseif x.op == :-
        SpinSym(x.label, :+)
    elseif x.op == :z
        x
    else
        throw(ArgumentError("Invalid spin operator symbol: $(x.op)."))
    end
end

function NonCommutativeProducts.mul_effect(a::SpinSym, b::SpinSym)
    if a.label > b.label
        return Swap(1)
    end
    if a.label < b.label
        return nothing
    end

    # Commutation relations: [Sz, S+] = S+, [Sz, S-] = -S-, [S+, S-] = 2Sz
    # Canonical order: :z < :+ < :-
    if a.op == :+ && b.op == :z
        # S+ Sz = Sz S+ - S+
        return AddTerms((Swap(1), -SpinSym(a.label, :+)))
    elseif a.op == :- && b.op == :z
        # S- Sz = Sz S- + S-
        return AddTerms((Swap(1), SpinSym(a.label, :-)))
    elseif a.op == :- && b.op == :+
        # S- S+ = S+ S- - 2Sz
        return AddTerms((Swap(1), -2 * SpinSym(a.label, :z)))
    else
        return nothing
    end
end

@nc SpinSym
NonCommutativeProducts.@commutative FermionSym SpinSym
NonCommutativeProducts.@commutative BosonSym SpinSym
NonCommutativeProducts.@commutative MajoranaSym SpinSym

@testitem "SpinSym" begin
    using Symbolics
    @variables a::Real z::Complex

    @spins S
    @fermions f
    @majoranas γ
    @bosons b

    Sz1 = S[1](:z)
    Sp1 = S[1](:+)
    Sm1 = S[1](:-)
    Sx1 = S[1](:x)
    Sy1 = S[1](:y)

    Sz2 = S[2](:z)
    Sp2 = S[2](:+)
    Sm2 = S[2](:-)

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
    @test Sz1 * b[1] - b[1] * Sz1 == 0
    @test Sp1 * b[1] - b[1] * Sp1 == 0
    @test Sz1 * b[1]' - b[1]' * Sz1 == 0
    @test Sp1 * b[1]' - b[1]' * Sp1 == 0

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