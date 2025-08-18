
struct SymbolicFermionBasis
    name::Symbol
    universe::UInt64
end
Base.hash(x::SymbolicFermionBasis, h::UInt) = hash(x.name, hash(x.universe, h))

"""
    @fermions a b ...

Create one or more fermion species with the given names. Indexing into fermions species
gives a concrete fermion. Fermions in one `@fermions` block anticommute with each other, 
and commute with fermions in other `@fermions` blocks.

# Examples:
- `@fermions a b` creates two species of fermions that anticommute:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1]' * b[1] + b[1] * a[1]' == 0`
- `@fermions a; @fermions b` creates two species of fermions that commute with each other:
    - `a[1]' * a[1] + a[1] * a[1]' == 1`
    - `a[1] * b[1] - b[1] * a[1] == 0`

See also [`@majoranas`](@ref), [`FermionicHilbertSpaces.eval_in_basis`](@ref).
"""
macro fermions(xs...)
    universe = hash(xs)
    defs = map(xs) do x
        :($(esc(x)) = SymbolicFermionBasis($(Expr(:quote, x)), $universe))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end
Base.:(==)(a::SymbolicFermionBasis, b::SymbolicFermionBasis) = a.name == b.name && a.universe == b.universe
Base.getindex(f::SymbolicFermionBasis, is...) = FermionSym(false, is, f)
Base.getindex(f::SymbolicFermionBasis, i) = FermionSym(false, i, f)

struct FermionSym{L,B} <: AbstractFermionSym
    creation::Bool
    label::L
    basis::B
end
Base.adjoint(x::FermionSym) = FermionSym(!x.creation, x.label, x.basis)
Base.iszero(x::FermionSym) = false
function Base.show(io::IO, x::FermionSym)
    print(io, x.basis.name, x.creation ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::FermionSym, b::FermionSym)
    if a.basis.universe !== b.basis.universe
        a.basis.universe < b.basis.universe
    elseif a.creation == b.creation
        a.basis.name == b.basis.name && return a.label < b.label
        a.basis.name < b.basis.name
    else
        a.creation > b.creation
    end
end
Base.:(==)(a::FermionSym, b::FermionSym) = a.creation == b.creation && a.label == b.label && a.basis == b.basis
Base.hash(a::FermionSym, h::UInt) = hash(a.creation, hash(a.label, hash(a.basis, h)))
struct NormalLabelOrder end
function mul_effect(a::FermionSym, b::FermionSym, ::NormalLabelOrder)
    if a == b
        0
    elseif a < b
        nothing
    elseif a > b
        swap = Swap((-1)^(a.basis.universe == b.basis.universe))
        if a.label == b.label && a.basis == b.basis
            return AddTerms((swap, 1))
        else
            return swap
        end
    else
        throw(ArgumentError("Don't know how to multiply $a * $b"))
    end
end

Base.valtype(::AbstractFermionSym) = Int
Base.valtype(::Type{S}) where {S<:AbstractFermionSym} = Int

@nc_eager FermionSym NormalLabelOrder()

""" 
    eval_in_basis(a, f)

Evaluate an expression with fermions in a basis `f`. 

# Examples
```julia
@fermions a
f = fermions(hilbert_space(1:2))
FermionicHilbertSpaces.eval_in_basis(a[1]'*a[2] + hc, f)
```
"""
eval_in_basis(a::FermionSym, f) = a.creation ? f[a.label]' : f[a.label]


@testitem "SymbolicFermions" begin
    using Symbolics, LinearAlgebra
    @fermions f c
    @fermions b
    @variables a::Real z::Complex
    f1 = f[:a]
    f2 = f[:b]
    f3 = f[1, :↑]

    # Test canonical commutation relations
    @test f1' * f1 + f1 * f1' == 1
    @test iszero(f1 * f2 + f2 * f1)
    @test iszero(f1' * f2 + f2 * f1')

    # c anticommutes with f
    @test iszero(f1' * c[1] + c[1] * f1')
    # b commutes with f
    @test iszero(f1' * b[1] - b[1] * f1')

    @test_nowarn display(f1)
    @test_nowarn display(f3)
    @test_nowarn display(1 * f1)
    @test_nowarn display(2 * f3)
    @test_nowarn display(1 + f1)
    @test_nowarn display(1 + f3)
    @test_nowarn display(1 + a * f2 - 5 * f1 + 2 * z * f1 * f2)

    @test iszero(f1 - f1)
    @test iszero(f1 * f1)
    @test iszero(2 * f1 - 2 * f1)
    @test iszero(0 * f1)
    @test iszero(f1 * 0)
    @test iszero(f1^2)
    @test iszero(0 * (f1 + f2))
    @test iszero((f1 + f2) * 0)
    @test iszero(f1 * f2 * f1)
    f12 = f1 * f2
    @test iszero(f12'' - f12)
    @test iszero(f12 * f12)
    @test iszero(f12' * f12')
    nf1 = f1' * f1
    @test nf1^2 == nf1

    @test 1 + (f1 + f2) == 1 + f1 + f2 == f1 + f2 + 1 == f1 + 1 + f2 == 1 * f1 + f2 + 1 == f1 + 0.5 * f2 + 1 + (0 * f1 + 0.5 * f2) == (0.5 + 0.5 * f1 + 0.2 * f2) + 0.5 + (0.5 * f1 + 0.8 * f2) == (1 + f1' + (1 * f2)')'
    @test iszero((2 * f1) * (2 * f1))
    @test iszero((2 * f1)^2)
    @test (2 * f2) * (2 * f1) == -4 * f1 * f2
    @test f1 == (f1 * (f1 + 1)) == (f1 + 1) * f1
    @test iszero(f1 * (f1 + f2) * f1)
    @test (f1 * (f1 + f2)) == f1 * f2
    @test (2nf1 - 1) * (2nf1 - 1) == 1

    @test (1 * f1) * f2 == f1 * f2
    @test (1 * f1) * (1 * f2) == f1 * f2
    @test f1 * f2 == f1 * (1 * f2) == f1 * f2
    @test f1 - 1 == (1 * f1) - 1 == (0.5 + f1) - 1.5

    for op in [f1, f1 + f2, f1' * f2, f1 * f2 + 1]
        @test op + 1.0I == op + 1.0
        @test op - 1.0I == op - 1.0 == -(1.0I - op) == -(1.0 - op)
    end

    ex = 2 * f1
    @test FermionicHilbertSpaces.head(ex) == (*)
    @test FermionicHilbertSpaces.children(ex) == [2, ex.factors...]
    @test FermionicHilbertSpaces.operation(ex) == (*)
    @test FermionicHilbertSpaces.arguments(ex) == [2, ex.factors...]
    @test FermionicHilbertSpaces.isexpr(ex)
    @test FermionicHilbertSpaces.iscall(ex)
    ex = 2 * f1 + 1
    @test FermionicHilbertSpaces.head(ex) == (+)
    @test FermionicHilbertSpaces.children(ex) == [1, 2 * f1]
    @test FermionicHilbertSpaces.operation(ex) == (+)
    @test FermionicHilbertSpaces.arguments(ex) == [1, 2 * f1]
    @test FermionicHilbertSpaces.isexpr(ex)
    @test FermionicHilbertSpaces.iscall(ex)

    ex = f1
    @test FermionicHilbertSpaces.head(ex) <: FermionicHilbertSpaces.FermionSym
    @test FermionicHilbertSpaces.children(ex) == [false, :a, f]
    @test FermionicHilbertSpaces.operation(ex) == FermionicHilbertSpaces.FermionSym
    @test FermionicHilbertSpaces.arguments(ex) == [false, :a, f]
    @test FermionicHilbertSpaces.isexpr(ex)
    @test FermionicHilbertSpaces.iscall(ex)

    @test substitute(f1, f1 => f2) == f2
    @test substitute(f1', f1' => f2) == f2
    @test substitute(f1', f1 => f2) == f1'
    @test substitute(f1 + f2, f1 => f2) == 2 * f2

    @test substitute(2 * f1, 2 => 3) == 3 * f1
    @test iszero(substitute(a * f1 + 1, a => a^2) - (a^2 * f1 + 1))
    @test iszero(substitute(a * f1 * f2 - f1 + 1 + 0.5 * f2' * f2, f1 => f2) - (a * f2 * f2 - f2 + 1 + 0.5 * f2' * f2))
    @test iszero(substitute(a * f1 + a + a * f1 * f2 * f1', a => 0))

    @test substitute(f[1], 1 => 2) == f[2]
    @test substitute(f[:a]' * f[:b] + 1, :a => :b) == f[:b]' * f[:b] + 1
end

TermInterface.operation(::FermionSym) = FermionSym
TermInterface.arguments(a::FermionSym) = [a.creation, a.label, a.basis]
TermInterface.children(a::FermionSym) = arguments(a)