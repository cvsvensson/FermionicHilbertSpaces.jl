struct MajoranaGroup
    id::FermionicGroup
end
Base.hash(x::MajoranaGroup, h::UInt) = hash(MajoranaGroup, hash(x.id, h))
Base.:(==)(a::MajoranaGroup, b::MajoranaGroup) = a.id == b.id
symbolic_group(g::MajoranaGroup) = g
Base.isless(g1::MajoranaGroup, g2::MajoranaGroup) = isless(g1.id, g2.id)
tags(g::MajoranaGroup) = tags(g.id)
add_tag(g::MajoranaGroup, tag::Symbol) = MajoranaGroup(add_tag(g.id, tag))
struct SymbolicMajoranaBasis
    name::Symbol
    group::MajoranaGroup
end
Base.hash(x::SymbolicMajoranaBasis, h::UInt) = hash(x.name, hash(x.group, h))
symbolic_group(f::SymbolicMajoranaBasis) = f.group
symbolic_id(f::SymbolicMajoranaBasis) = f.group
change_id(f::SymbolicMajoranaBasis, newid) = SymbolicMajoranaBasis(f.name, newid)
tags(f::SymbolicMajoranaBasis) = tags(symbolic_id(f))
add_tag(f::SymbolicMajoranaBasis, tag::Symbol) = change_id(f, add_tag(symbolic_id(f), tag))
Base.show(io::IO, x::SymbolicMajoranaBasis) = print(io, "SymbolicMajoranaBasis(", x.name, ")")

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

See also [`@fermions`](@ref).
"""
macro majoranas(xs...)
    group = MajoranaGroup(FermionicGroup(hash(xs)))
    defs = map(xs) do x
        :($(esc(x)) = SymbolicMajoranaBasis($(Expr(:quote, x)), $group))
    end
    Expr(:block, defs...,
        :(tuple($(map(x -> esc(x), xs)...))))
end
Base.getindex(f::SymbolicMajoranaBasis, is...) = MajoranaSym(is, f)
Base.getindex(f::SymbolicMajoranaBasis, i) = MajoranaSym(i, f)
Base.:(==)(a::SymbolicMajoranaBasis, b::SymbolicMajoranaBasis) = a.name == b.name && a.group == b.group

struct MajoranaSym{L,B} <: AbstractMajoranaSym
    label::L
    basis::B
end
Base.:(==)(a::MajoranaSym, b::MajoranaSym) = a.label == b.label && a.basis == b.basis
Base.hash(a::MajoranaSym, h::UInt) = hash(a.label, hash(a.basis, h))
Base.adjoint(x::MajoranaSym) = MajoranaSym(x.label, x.basis)
Base.iszero(x::MajoranaSym) = false
symbolic_group(f::AbstractMajoranaSym) = symbolic_group(f.basis)
symbolic_basis(f::AbstractMajoranaSym) = f.basis
change_basis(f::MajoranaSym, newbasis) = MajoranaSym(f.label, newbasis)
function Base.show(io::IO, x::MajoranaSym)
    print(io, _symbolic_name_with_tags(x.basis.name, x.basis))
    if Base.isiterable(typeof(x.label))
        Base.show_delim_array(io, x.label, "[", ",", "]", false)
    else
        print(io, "[", x.label, "]")
    end
end
function Base.isless(a::MajoranaSym, b::MajoranaSym)
    if a.basis.group !== b.basis.group
        a.basis.group < b.basis.group
    elseif a.basis.name == b.basis.name
        a.label < b.label
    else
        a.basis.name < b.basis.name
    end
end
function NonCommutativeProducts.mul_effect(a::MajoranaSym, b::MajoranaSym)
    if a == b
        1
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
        throw(ArgumentError("Don't know how to multiply $a * $b"))
    end
end

mat_eltype(::Type{S}) where {S<:AbstractMajoranaSym} = Complex{Int}
@nc MajoranaSym

@testitem "MajoranaSym" begin
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

struct MajoranaHilbertSpace{B,L,H} <: AbstractGroupedHilbertSpace{B}
    majoranaindices::L
    parent::H
    sym::SymbolicMajoranaBasis
    function MajoranaHilbertSpace(majoranaindices::L, parent::H, sym::SymbolicMajoranaBasis) where {L,H}
        B = statetype(parent)
        new{B,L,H}(majoranaindices, parent, sym)
    end
end

function SectorHilbertSpace(maj_space::MajoranaHilbertSpace, ordered_basis_states::Vector{B}, state_to_index::OrderedDict{B,Int64}, qn_to_states::OrderedDict{Q,Vector{B}}) where {B,Q}
    MajoranaHilbertSpace(maj_space.majoranaindices, SectorHilbertSpace(parent(maj_space), ordered_basis_states, state_to_index, qn_to_states), maj_space.sym)
end
dim(H::MajoranaHilbertSpace) = dim(H.parent)
mode_ordering(H::MajoranaHilbertSpace) = H.majoranaindices
modes(H::MajoranaHilbertSpace) = keys(H.majoranaindices)
Base.:(==)(H1::MajoranaHilbertSpace, H2::MajoranaHilbertSpace) = H1.majoranaindices == H2.majoranaindices && H1.parent == H2.parent && H1.sym == H2.sym
Base.hash(H::MajoranaHilbertSpace, h::UInt) = hash(H.majoranaindices, hash(H.parent, hash(H.sym, h)))
basisstates(m::MajoranaHilbertSpace) = basisstates(m.parent)
basisstate(i, m::MajoranaHilbertSpace) = basisstate(i, m.parent)
Base.parent(H::MajoranaHilbertSpace) = H.parent
nbr_of_modes(H::MajoranaHilbertSpace) = nbr_of_modes(H.parent)
isconstrained(H::MajoranaHilbertSpace) = isconstrained(H.parent)
group_id(H::MajoranaHilbertSpace) = symbolic_group(H.sym)
function atomic_id(H::MajoranaHilbertSpace)
    length(H.majoranaindices) == 2 || throw(ArgumentError("Atomic ID is only defined for MajoranaHilbertSpaces with exactly 2 Majoranas."))
    (H.sym, H.majoranaindices)
end

quantumnumbers(H::MajoranaHilbertSpace) = quantumnumbers(H.parent)
indices(qn::Q, H::MajoranaHilbertSpace{<:Any,<:Any,SectorHilbertSpace{B,P,Q}}) where {B,P,Q} = indices(qn, parent(H))
function sector(qn, H::MajoranaHilbertSpace)
    # get sector from parent, then convert to majorana
    parent_sector = sector(qn, parent(H))
    MajoranaHilbertSpace(H.majoranaindices, parent_sector, H.sym)
end
sector(::Nothing, H::MajoranaHilbertSpace) = H
# indices(Hsub::AbstractHilbertSpace, H::MajoranaHilbertSpace) = indices(Hsub, parent(H))
# indices(::Nothing, H::MajoranaHilbertSpace) = indices(nothing, parent(H))

function combine_into_group(group::MajoranaGroup, spaces)
    fermionic_space = combine_into_group(group.id, map(parent, spaces))
    D = typeof(first(spaces).majoranaindices)
    majoranaindices = D()
    count = 1
    for space in spaces
        l1, p1 = first(space.majoranaindices)
        l2, p2 = last(space.majoranaindices)
        majoranaindices[l1] = count
        majoranaindices[l2] = count + 1
        count += 2
    end
    MajoranaHilbertSpace(majoranaindices, fermionic_space, first(spaces).sym)
end

function state_mapper(H::MajoranaHilbertSpace, Hs)
    state_mapper(parent(H), Hs)
end
_find_position(f::MajoranaHilbertSpace, H::FermionicSpace) = _find_position(f.parent, H)
partial_trace_phase_factor(f1, f2, H::MajoranaHilbertSpace) = partial_trace_phase_factor(f1, f2, H.parent)

function majoranas(H::MajoranaHilbertSpace)
    γ = symbolic_basis(H)
    OrderedDict(l => matrix_representation(γ[l], H) for l in labels(H))
end
symbolic_basis(H::MajoranaHilbertSpace) = H.sym

hilbert_space(y::SymbolicMajoranaBasis, labels, args...; kwargs...) = majorana_hilbert_space(y, labels, args...; kwargs...)
"""
    majorana_hilbert_space(labels, qn)

Represents a hilbert space for majoranas. `labels` must be an even number of unique labels.
"""
function majorana_hilbert_space(y::SymbolicMajoranaBasis, labels, args...; kwargs...)
    iseven(length(labels)) || throw(ArgumentError("Must be an even number of Majoranas to define a Hilbert space."))
    pairs = [(labels[i], labels[i+1]) for i in 1:2:length(labels)-1]
    f = SymbolicFermionBasis(Symbol(y.name, "_fermions",), y.group.id)
    H = hilbert_space(f, pairs, args...; kwargs...)
    majorana_position = OrderedDict(y[label] => n for (n, label) in enumerate(labels))
    MajoranaHilbertSpace(majorana_position, H, y)
end

function subregion(Hsub::MajoranaHilbertSpace, H::MajoranaHilbertSpace)
    parent_subregion = subregion(parent(Hsub), parent(H))
    MajoranaHilbertSpace(Hsub.majoranaindices, parent_subregion, Hsub.sym)
end

partial_trace!(mout, m::AbstractMatrix, H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace, complement::MajoranaHilbertSpace, alg::FullPartialTraceAlg, args...; kwargs...) = partial_trace!(mout, m, H.parent, Hsub.parent, complement.parent, alg, args...; kwargs...)
partial_trace!(mout, m::AbstractMatrix, H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace, complement::MajoranaHilbertSpace, alg::SubsystemPartialTraceAlg, args...; kwargs...) = partial_trace!(mout, m, H.parent, Hsub.parent, complement.parent, alg, args...; kwargs...)

function partial_trace(m::NCMul{C,S,F}, H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace) where {C,S<:AbstractMajoranaSym,F}
    sub_modes = modes(Hsub)
    for f in m.factors
        if f ∉ sub_modes
            return 0 * m
        end
    end
    return m * dim(H) / dim(Hsub)
end
function partial_trace(m::NCAdd{C,NCMul{C2,S,F}}, H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace; kwargs...) where {C,C2,S<:AbstractMajoranaSym,F}
    return sum(partial_trace(term, H, Hsub; kwargs...) for term in NCterms(m); init=0 * m) + m.coeff * dim(H) / dim(Hsub)
end
function partial_trace(m::NCAdd{C,NCMul{C2,S,F}}, Hs::Pair{<:MajoranaHilbertSpace,<:MajoranaHilbertSpace}) where {C,C2,S<:AbstractMajoranaSym,F}
    return partial_trace(m, Hs...)
end

@testitem "Partial trace of symbolic Majoranas" begin
    @majoranas y
    H = majorana_hilbert_space(y, 1:6)
    Hsub = subregion(majorana_hilbert_space(y, 3:4), H)
    op = 1 + 3y[1] + 2y[3] + 4y[1] * y[6] + 3y[4] * y[1] + y[3] * y[4] + y[1] * y[3] * y[4] + y[1] * y[2] * y[6]
    op2 = 3 + 0y[1] # NCterms(op2) is empty
    @test matrix_representation(partial_trace(op, H => Hsub), Hsub) == partial_trace(matrix_representation(op, H), H => Hsub)
    @test matrix_representation(partial_trace(op2, H => Hsub), Hsub) == partial_trace(matrix_representation(op2, H), H => Hsub)
end


state_index(state::AbstractFockState, H::MajoranaHilbertSpace) = state_index(state, H.parent)
_find_position(f::MajoranaSym, H::MajoranaHilbertSpace) = get(H.majoranaindices, f, 0)

function _precomputation_before_operator_application(op::NCMul, space::MajoranaHilbertSpace)
    # find positions of all majoranas in the operator
    majoranapositions = map(f -> _find_position(f, space), op.factors)
    fermionpositions = map(n -> div(n + 1, 2), majoranapositions)
    daggers = map(iseven, majoranapositions)
    return fermionpositions, daggers
end
function apply_local_operators(op::NCMul, f::FockNumber, H::MajoranaHilbertSpace, (fpos, daggers); kwargs...)
    state, amp = togglemajoranas(Iterators.reverse(fpos), Iterators.reverse(daggers), f)
    return (state,), (amp * op.coeff,)
end

function atomic_factors(H::MajoranaHilbertSpace)
    parent_atoms = atomic_factors(H.parent)
    # convert to majoranas
    γ = H.sym
    map(enumerate(parent_atoms)) do (i, atom)
        fsym = only(modes(atom))
        majoranaindices = OrderedDict(γ[fsym.label[1]] => 1, γ[fsym.label[2]] => 2)
        MajoranaHilbertSpace(majoranaindices, atom, H.sym)
    end
end
@testitem "Majorana matrix representations" begin
    using LinearAlgebra
    @majoranas γ
    H = hilbert_space(γ, 1:2)
    Hf = H.parent
    f = H.parent.modes[1].basis

    @test parityoperator(H.parent) == matrix_representation(1im * γ[1] * γ[2], H)
    y1 = matrix_representation(γ[1], H)
    y2 = matrix_representation(γ[2], H)
    @test y1 * y2 == matrix_representation(γ[1] * γ[2], H)

    maj(f) = f.creation ? -1im * f + hc : f + hc
    @test matrix_representation(γ[1], H) == matrix_representation(maj(f[(1, 2)]), Hf)
    @test matrix_representation(γ[2], H) == matrix_representation(maj(f[(1, 2)]'), Hf)
    @test matrix_representation(1, H) == matrix_representation(1, Hf) == matrix_representation(1I, H) == matrix_representation(1I, Hf)
    @test matrix_representation(γ[1] * γ[2], H) == matrix_representation(maj(f[(1, 2)]) * maj(f[(1, 2)]'), Hf)
    @test matrix_representation(1 + γ[1] + 1im * γ[2] + 0.2 * γ[1] * γ[2], H) ==
          matrix_representation(1 + maj(f[(1, 2)]) + 1im * maj(f[(1, 2)]') + 0.2 * maj(f[(1, 2)]) * maj(f[(1, 2)]'), Hf)
end

@testitem "Majorana hilbert space" begin
    import FermionicHilbertSpaces: @majoranas
    @majoranas γ
    H = hilbert_space(γ, 1:4, ParityConservation(1))
    @test dim(H) == 2
    Hsub = subregion(hilbert_space(γ, 1:2), H)
    @test dim(Hsub) == 2
    Hf = H.parent
    Hfsub = subregion(hilbert_space(Hsub.parent.parent.modes[1].basis, [(1, 2)]), Hf)
    m = rand(dim(H), dim(H))
    @test partial_trace(m, H => Hsub) == partial_trace(m, Hf => Hfsub)
    Hsub2 = subregion(hilbert_space(γ, 3:4), H)
    Hfsub2 = subregion(hilbert_space(Hsub.parent.parent.modes[1].basis, [(3, 4)]), Hf)

    Hprod = tensor_product(Hsub, Hsub2)
    @test basisstates(Hprod) == basisstates(tensor_product(Hfsub, Hfsub2))

    m1 = rand(dim(Hsub), dim(Hsub))
    m2 = rand(dim(Hsub2), dim(Hsub2))
    @test tensor_product((m1, m2), (Hsub, Hsub2), Hprod) == embed(m1, Hsub => Hprod) * embed(m2, Hsub2 => Hprod)
end
