struct FermionicGroup{T}
    id::T
end
Base.hash(x::FermionicGroup, h::UInt) = hash(x.id, h)
Base.:(==)(a::FermionicGroup, b::FermionicGroup) = a.id == b.id
symbolic_group(g::FermionicGroup) = g
Base.isless(g1::FermionicGroup, g2::FermionicGroup) = isless(g1.id, g2.id)
add_tag(g::FermionicGroup, tag) = FermionicGroup(add_tag(g.id, tag))
tags(g::FermionicGroup) = g.id

struct SymbolicFermionBasis{T<:Tags}
    name::Symbol
    tags::FermionicGroup{T}
end
Base.show(io::IO, x::SymbolicFermionBasis) = print(io, _symbolic_name_with_tags(x.name, x; skip_first=1))
Base.hash(x::SymbolicFermionBasis, h::UInt) = hash(x.name, hash(x.tags, h))
symbolic_group(x::SymbolicFermionBasis) = tags(x)
tags(x::SymbolicFermionBasis) = x.tags
add_tag(x::SymbolicFermionBasis, tag) = SymbolicFermionBasis(x.name, add_tag(x.tags, tag))

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
    group = FermionicGroup(Tags((xs,)))
    defs = map(xs) do x
        :($(esc(x)) = SymbolicFermionBasis($(Expr(:quote, x)), $group))
    end
    fermions = Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
    length(defs) == 1 && return only(defs)
    return fermions
end
Base.:(==)(a::SymbolicFermionBasis, b::SymbolicFermionBasis) = a.name == b.name && a.tags == b.tags
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
with_basis(h::FermionSym, newbasis) = FermionSym(h.creation, h.label, newbasis)
atomic_id(h::FermionSym) = _normalize_sym(h) # names the single-mode FermionicSpace of h
label(h::FermionSym) = h.label
group_id(f::FermionSym) = symbolic_group(f)

function Base.show(io::IO, x::FermionSym)
    print(io, _symbolic_name_with_tags(x.basis.name, x.basis; skip_first=1), x.creation ? "†" : "")
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::FermionSym, b::FermionSym)
    if tags(a.basis) !== tags(b.basis)
        tags(a.basis) < tags(b.basis)
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
        anti_commuting = tags(a.basis) == tags(b.basis)
        swap = Swap((-1)^anti_commuting)
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
    @test sprint(show, f) == "f"
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

@testitem "Deterministic ordering of groups" begin
    using FermionicHilbertSpaces: Tags, tags, add_tag, SymbolicState
    @fermions a
    @fermions b
    @fermions c d
    @majoranas γ

    # the groups are identified by the names given to the macro, not by a hash of them
    @test tags(a).id.tags == ((:a,),)
    @test tags(c).id.tags == ((:c, :d),)
    @test tags(γ).group.id.tags == ((:γ,),)
    a_old = a
    @fermions a
    @test a == a_old

    # groups are in lexicographic order of the names given to the macro
    @test tags(a) < tags(b) < tags(c)
    @test a[2] < b[1] && b[1] < c[1] && !(b[1] < a[2])
    @test a[1]' < b[1]
    @test only(keys((b[1] * a[1]).dict)).factors == [a[1], b[1]]
    @test only(keys((c[1] * b[1]' * a[2]).dict)).factors == [a[2], b[1]', c[1]]

    # the order is total and consistent with ==
    ts = [Tags(nothing), Tags((:left,)), Tags((:right,)), tags(a).id, tags(b).id, tags(c).id,
        add_tag(tags(a).id, :left), add_tag(tags(a).id, :right), add_tag(add_tag(tags(a).id, :left), :right),
        Tags(((:a,),)), Tags(((:a, :b),))]
    for x in ts, y in ts
        @test (x == y) == (!(x < y) && !(y < x))
        @test !(x < y && y < x)
    end
    @test issorted(sort(ts))
    @test sort(ts) == sort(reverse(ts))

    # symbolic states are ordered by group, space, ket and bra
    @boson bb
    Hf = hilbert_space(a, 1:1)
    Hb = hilbert_space(bb, 3)
    vf, vb = Kets(Hf), Kets(Hb)
    @test vf("0") < vf("1")
    @test vf("0")' < vf("0")
    states = [vf("0"), vf("1"), vf("0")', vf("1")', SymbolicState(Hf, vf("1").ket, vf("0").ket),
        vb("0"), vb("2"), vb("1")']
    for x in states, y in states
        @test (x == y) == (!(x < y) && !(y < x))
    end
    @test issorted(sort(states))
end

@testitem "Fermionic triple products vs matrices" begin
    @fermions f c
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(c, 1:1))
    ops = [f[1], f[1]', f[2], f[2]', c[1], c[1]']
    mats = [representation(op, H) for op in ops]
    for i in eachindex(ops), j in eachindex(ops), k in eachindex(ops)
        @test representation(ops[i] * ops[j] * ops[k], H) ≈ mats[i] * mats[j] * mats[k]
    end
end
