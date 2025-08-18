struct SymbolicMajoranaBasis
    name::Symbol
    universe::UInt64
end
Base.hash(x::SymbolicMajoranaBasis, h::UInt) = hash(x.name, hash(x.universe, h))

abstract type AbstractMajoranaSym <: AbstractFermionSym end
"""
    @majoranas a b ...

Create one or more Majorana species with the given names. Indexing into Majorana species
gives a concrete Majorana. Majoranas in one `@majoranas` block anticommute with each other,
and commute with Majoranas in other `@majoranas` blocks.

# Examples:
- `@majoranas a b` creates two species of Majoranas that anticommute:
    - `a[1] * a[1] + a[1] * a[1] == 1`
    - `a[1] * b[1] + b[1] * a[1] == 0`
- `@majoranas a; @majoranas b` creates two species of Majoranas that commute with each other:
    - `a[1] * a[1] + a[1] * a[1] == 1`
    - `a[1] * b[1] - b[1] * a[1] == 0`

See also [`@fermions`](@ref), [`FermionicHilbertSpaces.eval_in_basis`](@ref).
"""
macro majoranas(xs...)
    universe = hash(xs)
    defs = map(xs) do x
        :($(esc(x)) = SymbolicMajoranaBasis($(Expr(:quote, x)), $universe))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end
Base.getindex(f::SymbolicMajoranaBasis, is...) = MajoranaSym(is, f)
Base.getindex(f::SymbolicMajoranaBasis, i) = MajoranaSym(i, f)
Base.:(==)(a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis) = a.name == b.name && a.universe == b.universe

struct MajoranaSym{L,B} <: AbstractMajoranaSym
    label::L
    basis::B
end
Base.:(==)(a::MajoranaSym, b::MajoranaSym) = a.label == b.label && a.basis == b.basis
Base.hash(a::MajoranaSym, h::UInt) = hash(a.label, hash(a.basis, h))
Base.adjoint(x::MajoranaSym) = MajoranaSym(x.label, x.basis)
Base.iszero(x::MajoranaSym) = false
function Base.show(io::IO, x::MajoranaSym)
    print(io, x.basis.name)
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::MajoranaSym, b::MajoranaSym)
    if a.basis.universe !== b.basis.universe
        a.basis.universe < b.basis.universe
    elseif a.basis.name == b.basis.name
        a.label < b.label
    else
        a.basis.name < b.basis.name
    end
end
function NonCommutativeProducts.mul_effect(a::MajoranaSym, b::MajoranaSym, ::NormalLabelOrder)
    if a == b
        1
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
eval_in_basis(a::MajoranaSym, f) = f[a.label]

Base.valtype(::S) where {S<:AbstractMajoranaSym} = Complex{Int}
Base.valtype(::Type{S}) where {S<:AbstractMajoranaSym} = Complex{Int}
@nc_eager MajoranaSym NormalLabelOrder()

@testitem "MajoranaSym" begin
    using Symbolics
    @variables a::Real z::Complex

    @majoranas γ f

    @test 1 * γ[1] == γ[1]
    @test 1 * γ[1] + 0 == γ[1]
    @test 1 * γ[1] + 0 == 1 * γ[1]
    @test hash(γ[1]) == hash(1 * γ[1]) == hash(1 * γ[1] + 0)

    #test canonical anticommutation relations
    @test γ[1] * γ[1] == 1
    @test γ[1] * γ[2] == -γ[2] * γ[1]
    @test γ[1] * f[1] + f[1] * γ[1] == 0

    @test γ[1] * γ[2] * γ[1] == -γ[2]
    @test γ[1] * γ[2] * γ[3] == -γ[3] * γ[2] * γ[1]

    f1 = (γ[1] + 1im * γ[2]) / 2
    f2 = (γ[3] + 1im * γ[4]) / 2
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
end
