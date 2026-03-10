
struct SymbolicBosonBasis
    name::Symbol
end
Base.hash(x::SymbolicBosonBasis, h::UInt) = hash(x.name, h)

macro bosons(x)
    Expr(:block, :($(esc(x)) = SymbolicBosonBasis($(Expr(:quote, x)))),
        :($(esc(x))))
end
Base.:(==)(a::SymbolicBosonBasis, b::SymbolicBosonBasis) = a.name == b.name
Base.getindex(f::SymbolicBosonBasis, is...) = BosonSym(f, is, -1)
Base.getindex(f::SymbolicBosonBasis, i) = BosonSym(f, i, -1)

struct BosonSym{B,L}
    basis::B
    label::L
    exp::Int
end
Base.adjoint(x::BosonSym) = BosonSym(x.basis, x.label, -x.exp)
Base.iszero(x::BosonSym) = false
function Base.show(io::IO, x::BosonSym)
    print(io, x.basis.name, x.exp > 0 ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
    if abs(x.exp) !== 1
        print(io, "^", abs(x.exp))
    end
end
Base.:(==)(a::BosonSym, b::BosonSym) = a.exp == b.exp && a.label == b.label && a.basis == b.basis
Base.hash(a::BosonSym, h::UInt) = hash(a.exp, hash(a.label, hash(a.basis, h)))
get_symbolic_basis(f::BosonSym) = f.basis

function NonCommutativeProducts.mul_effect(a::BosonSym, b::BosonSym)
    if a.label == b.label && a.basis == b.basis
        if sign(a.exp) == sign(b.exp)
            return BosonSym(a.basis, a.label, a.exp + b.exp)
        else
            if a.exp < 0 && b.exp > 0
                return AddTerms((Swap(1), 1))
            else
                return nothing
            end
        end
    end
    if a.basis.name > b.basis.name
        return Swap(1)
    elseif a.basis.name == b.basis.name
        if a.label > b.label
            return Swap(1)
        else
            return nothing
        end
    end
    throw(ArgumentError("mul_effect undefined for $a * $b"))
end

@nc BosonSym
NonCommutativeProducts.@commutative FermionSym BosonSym
NonCommutativeProducts.@commutative MajoranaSym BosonSym

@testitem "BosonSym" begin
    using Symbolics
    @variables a::Real z::Complex

    @bosons b
    @fermions f
    @majoranas γ

    b1 = b[1]
    b2 = b[2]

    @test 1 * b1 == b1
    @test 1 * b1 + 0 == b1
    @test 1 * b1 + 0 == 1 * b1
    @test hash(b1) == hash(1 * b1) == hash(1 * b1 + 0)

    # Test canonical commutation relations
    @test b1 * b1' - b1' * b1 == 1
    @test b1' * b1 - b1 * b1' == -1

    # Bosons at different sites commute
    @test b1 * b2 - b2 * b1 == 0
    @test b1' * b2 - b2 * b1' == 0
    @test b1' * b2' - b2' * b1' == 0

    # Bosons commute with fermions
    @test b1 * f[1] - f[1] * b1 == 0
    @test b1' * f[1] - f[1] * b1' == 0
    @test b1 * f[1]' - f[1]' * b1 == 0
    @test b1' * f[1]' - f[1]' * b1' == 0

    # Bosons commute with majoranas
    @test b1 * γ[1] - γ[1] * b1 == 0
    @test b1' * γ[1] - γ[1] * b1' == 0

    @test iszero(b1 - b1)
    @test iszero(2 * b1 - 2 * b1)
    @test iszero(0 * b1)
    @test iszero(b1 * 0)
    @test iszero(0 * (b1 + b2))
    @test iszero((b1 + b2) * 0)

    # Test adjoint
    @test b1'' == b1
    @test (b1 * b2)' == b2' * b1'

    # Test algebraic properties
    @test 1 + (b1 + b2) == 1 + b1 + b2 == b1 + b2 + 1 == b1 + 1 + b2
    @test (b1 * b2) * b1' == b1 * (b2 * b1')
end

# _sym_space_match(basis::SymbolicBosonBasis, space::BosonicHilbertSpace) = true
_sym_space_match(basis::SymbolicBosonBasis, space::SymbolicBosonBasis) = false