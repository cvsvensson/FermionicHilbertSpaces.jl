_left_basis(base) = add_tag(base, :left)
_right_basis(base) = add_tag(base, :right)

function open_system(base, args...; kwargs...)
    left_basis = _left_basis(base)
    right_basis = _right_basis(base)
    Hleft = hilbert_space(left_basis, args...; kwargs...)
    Hright = TransposedSpace(hilbert_space(right_basis, args...; kwargs...))
    Hfull = tensor_product(Hleft, Hright)
    left(op) = add_tag(op, :left)
    right(op) = add_tag(op, :right)
    return Hfull, Hleft, Hright, left, right
end
function open_system(H::AbstractHilbertSpace; kwargs...)
    Hleft = add_tag(H, :left)
    Hright = TransposedSpace(add_tag(H, :right))
    Hlr = tensor_product(Hleft, Hright; kwargs...)
    return Hlr, Hleft, Hright, left, right
end

function _remap_factor_basis(factor, tag)
    basis = symbolic_basis(factor)
    new_basis = add_tag(basis, tag)
    return change_basis(factor, new_basis)
end

function _remap_operator(op::NCMul, tag)
    new_factors = map(f -> _remap_factor_basis(f, tag), op.factors)
    NCMul(op.coeff, new_factors)
end

function _remap_operator(op::NCAdd, tag)
    remapped = op.coeff
    for (term, coeff) in op.dict
        remapped += coeff * _remap_operator(term, tag)
    end
    return remapped
end

add_tag(op::AbstractSym, tag) = _remap_factor_basis(op, tag)
add_tag(op::NCMul, tag) = _remap_operator(op, tag)
add_tag(op::NCAdd, tag) = _remap_operator(op, tag)

struct TransposedSpace{B,H} <: AbstractHilbertSpace{B}
    parent::H
end
TransposedSpace(inner::H) where {H<:AbstractHilbertSpace} = TransposedSpace{statetype(inner),H}(inner)

Base.:(==)(a::TransposedSpace, b::TransposedSpace) = a.parent == b.parent
Base.hash(H::TransposedSpace, h::UInt) = hash(H.parent, h)
basisstates(H::TransposedSpace) = basisstates(H.parent)
basisstate(i::Int, H::TransposedSpace) = basisstate(i, H.parent)
state_index(state, H::TransposedSpace) = state_index(state, H.parent)
dim(H::TransposedSpace) = dim(H.parent)
isconstrained(H::TransposedSpace) = isconstrained(H.parent)
group_id(H::TransposedSpace) = group_id(H.parent)
atomic_id(H::TransposedSpace) = atomic_id(H.parent)
atomic_factors(H::TransposedSpace) = map(TransposedSpace, atomic_factors(H.parent)) # (H,)
add_tag(H::TransposedSpace, tag) = TransposedSpace(add_tag(H.parent, tag))
# Base.keys(H::TransposedSpace) = (atomic_id(H),)
groups(H::TransposedSpace) = groups(H.parent)
_precomputation_before_operator_application(factors, space::TransposedSpace) = _precomputation_before_operator_application(factors, space.parent)
TransposedSpace(H::ProductSpace) = ProductSpace(map(TransposedSpace, factors(H)), map(TransposedSpace, H.atoms))
state_mapper(H::TransposedSpace, Hs) = state_mapper(H.parent, Hs)
combine_states(states, H::TransposedSpace) = combine_states(states, H.parent)


function FermionicSpace(spaces::AbstractVector{F}, group) where {F<:TransposedSpace}
    only(unique(map(group_id, spaces))) == group || throw(ArgumentError("All spaces must belong to the same group"))
    TransposedSpace(FermionicSpace(map(H -> H.parent, spaces), group))
end


function matrix_representation(op, H::TransposedSpace; kwargs...)
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(dim(H))
    end
    op_groups = symbolic_groups(op)
    space_groups = unique(Iterators.map(group_id, factors(H.parent)))
    all(in(space_groups), op_groups) || throw(ArgumentError("Symbolic bases in operator do not match the provided space. Operator groups: $op_groups, expected one of: $space_groups"))
    return transpose(matrix_representation(op, H.parent; kwargs...))
end

function left_operator(op, H::AbstractHilbertSpace, Hfull::AbstractHilbertSpace=tensor_product(H, TransposedSpace(H)); kwargs...)
    Ht = TransposedSpace(H)
    left_matrix = matrix_representation(op, H; kwargs...)
    return generalized_kron((left_matrix, I), (H, Ht), Hfull)
end

function right_operator(op, H::AbstractHilbertSpace, Hfull::AbstractHilbertSpace=tensor_product(H, TransposedSpace(H)); kwargs...)
    Ht = TransposedSpace(H)
    right_matrix = matrix_representation(op, Ht; kwargs...)
    return generalized_kron((I, right_matrix), (H, Ht), Hfull)
end

@testitem "TransposedSpace basics" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: TransposedSpace

    @fermions c
    H = hilbert_space(c, 1:1)
    Ht = TransposedSpace(H)
    op = c[1]' * c[1]
    @test matrix_representation(op, Ht) == transpose(matrix_representation(op, H))

    @boson b
    Hb = hilbert_space(b, 2)
    @test matrix_representation(b' * b, TransposedSpace(Hb)) == transpose(matrix_representation(b' * b, Hb))

    @spin s 1 // 2
    Hs = hilbert_space(s)
    @test matrix_representation(s[:x], TransposedSpace(Hs)) == transpose(matrix_representation(s[:x], Hs))
end

@testitem "open_system symbolic left-right interface" begin
    using FermionicHilbertSpaces: open_system
    @fermions c
    Hfull, Hleft, Hright, left, right = open_system(c, 1:1)
    c_left = left(c)
    c_right = right(c)
    op = c[1]' * c[1]

    M = matrix_representation(left(op) * right(op) + left(op), Hfull)
    @test size(M) == (dim(Hfull), dim(Hfull))
    Mexpected = matrix_representation((c_left[1]' * c_left[1]) * (c_right[1]' * c_right[1]) + (c_left[1]' * c_left[1]), Hfull)
    @test M ≈ Mexpected

    @spin s 1 // 2
    Hs, _, _ = open_system(s)
    s_left = left(s)
    s_right = right(s)
    H = tensor_product(Hfull, Hs)
    op = c[1]' * s[:z]
    M = matrix_representation(left(op) * right(op) + left(op), H)
    Mexpected = matrix_representation((c_left[1]' * s_left[:z]) * (c_right[1]' * s_right[:z]) + (c_left[1]' * s_left[:z]), H)
    @test M ≈ Mexpected
end


@inline function push_inds_amps!((outinds, ininds, amps), inind, newstates, newamps, coeff, space::TransposedSpace; projection=false)
    for n in eachindex(newstates, newamps)
        newstate = newstates[n]
        amp = newamps[n]
        if !iszero(amp)
            outind = state_index(newstate, space)
            if !projection || !ismissing(outind)
                push!(outinds, inind)
                push!(amps, amp * coeff)
                push!(ininds, outind)
            end
        end
    end
end