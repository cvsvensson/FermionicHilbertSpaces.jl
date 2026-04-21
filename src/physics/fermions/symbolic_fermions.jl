struct FermionicGroup{T}
    id::T
end
Base.hash(x::FermionicGroup, h::UInt) = hash(x.id, h)
Base.:(==)(a::FermionicGroup, b::FermionicGroup) = a.id == b.id
symbolic_group(g::FermionicGroup) = g
Base.isless(g1::FermionicGroup, g2::FermionicGroup) = isless(g1.id, g2.id)

struct SymbolicFermionBasis{T<:Tags}
    name::Symbol
    group::FermionicGroup
    tags::T
end
SymbolicFermionBasis(name, group) = SymbolicFermionBasis(name, group, Tags(nothing))
Base.hash(x::SymbolicFermionBasis, h::UInt) = hash(x.tags, hash(x.name, hash(x.group, h)))
symbolic_group(h::SymbolicFermionBasis) = fermionic_group(h)
fermionic_group(b::SymbolicFermionBasis) = b.group
tags(b::SymbolicFermionBasis) = b.tags
add_tag(b::SymbolicFermionBasis, tag) = SymbolicFermionBasis(b.name, b.group, add_tag(b.tags, tag))

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

See also [`@majoranas`](@ref).
"""
macro fermions(xs...)
    group = FermionicGroup(hash(xs))
    defs = map(xs) do x
        :($(esc(x)) = SymbolicFermionBasis($(Expr(:quote, x)), $group))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end
Base.:(==)(a::SymbolicFermionBasis, b::SymbolicFermionBasis) = a.name == b.name && a.group == b.group
Base.getindex(f::SymbolicFermionBasis, is...) = FermionSym(false, is, f)
Base.getindex(f::SymbolicFermionBasis, i) = FermionSym(false, i, f)

struct FermionSym{L,B} <: AbstractFermionSym
    creation::Bool
    label::L
    basis::B
end
Base.adjoint(x::FermionSym) = FermionSym(!x.creation, x.label, x.basis)
Base.iszero(x::FermionSym) = false
symbolic_group(h::FermionSym) = symbolic_group(h.basis)
symbolic_basis(h::FermionSym) = h.basis
change_basis(h::FermionSym, newbasis) = FermionSym(h.creation, h.label, newbasis)
atomic_id(h::FermionSym) = (h.basis, h.label)
label(h::FermionSym) = h.label
group_id(f::FermionSym) = symbolic_group(f)

function Base.show(io::IO, x::FermionSym)
    print(io, _symbolic_name_with_tags(x.basis.name, x.basis), x.creation ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::FermionSym, b::FermionSym)
    if a.basis.group !== b.basis.group
        a.basis.group < b.basis.group
    elseif a.creation == b.creation
        a.basis.name == b.basis.name && return a.label < b.label
        a.basis.name < b.basis.name
    else
        a.creation > b.creation
    end
end
Base.:(==)(a::FermionSym, b::FermionSym) = a.creation == b.creation && a.label == b.label && a.basis == b.basis
Base.hash(a::FermionSym, h::UInt) = hash(a.creation, hash(a.label, hash(a.basis, h)))

hilbert_space(f::FermionSym) = FermionicSpace([_normalize_sym(f)], symbolic_group(f))

function NonCommutativeProducts.mul_effect(a::FermionSym, b::FermionSym)
    if a == b
        0
    elseif a < b
        nothing
    elseif a > b
        swap = Swap((-1)^(a.basis.group == b.basis.group))
        if a.label == b.label && a.basis == b.basis
            return AddTerms((swap, 1))
        else
            return swap
        end
    else
        return nothing
    end
end

mat_eltype(::Type{S}) where {S<:AbstractFermionSym} = Int

@nc FermionSym

@testitem "SymbolicFermions" begin
    using Symbolics, LinearAlgebra
    @fermions f c
    @fermions b
    @variables a::Real z::Complex
    f1 = f[:a]
    f2 = f[:b]
    f3 = f[1, :↑]

    @test 1 * f1 == f1
    @test 1 * f1 + 0 == f1
    @test 1 * f1 + 0 == 1 * f1
    @test hash(f1) == hash(1 * f1) == hash(1 * f1 + 0)

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

end

# """
#     apply_local_operator(op, state, space) -> (new_state, amplitude)

# Apply a local operator (single factor or product) to a state in a single Hilbert space.
# Returns the resulting state and amplitude.

# Type-specific implementations are defined in their respective files (e.g., symbolic_spin.jl).
# """
# function apply_local_operator(op::FermionSym, state::FockNumber, space::AbstractFockHilbertSpace)
#     # Convert single FermionSym to NCMul and use existing machinery
#     ordering = mode_ordering(space)
#     digitpos = getindex(ordering, op.label)
#     dagger = op.creation
#     new_state, amp = togglefermions([digitpos], [dagger], state)
#     return (new_state, amp)
# end
