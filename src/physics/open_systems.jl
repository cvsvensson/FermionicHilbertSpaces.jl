
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
groups(H::TransposedSpace) = groups(H.parent)
factors(H::TransposedSpace) = factors(H.parent)
_precomputation_before_operator_application(factors, space::TransposedSpace) = _precomputation_before_operator_application(factors, space.parent)
TransposedSpace(H::ProductSpace) = ProductSpace(map(TransposedSpace, factors(H)), map(TransposedSpace, H.atoms))
TransposedSpace(H::ConstrainedSpace) = ConstrainedSpace(TransposedSpace(H.parent), H.states, H.state_index)
TransposedSpace(H::SectorHilbertSpace) = SectorHilbertSpace(TransposedSpace(parent(H)), H.ordered_basis_states, H.state_to_index, H.qn_to_states, H.constraint)
state_mapper(H::TransposedSpace, Hs) = state_mapper(H.parent, Hs)
combine_states(states, H::TransposedSpace) = combine_states(states, H.parent)
Base.parent(H::TransposedSpace) = H.parent

function FermionicSpace(spaces::AbstractVector{F}, group) where {F<:TransposedSpace}
    only(unique(map(group_id, spaces))) == group || throw(ArgumentError("All spaces must belong to the same group"))
    TransposedSpace(FermionicSpace(map(H -> H.parent, spaces), group))
end


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

@testitem "Matrix representation of transposed spaces" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: TransposedSpace

    # An operator A*B acts as A*(B*v) on a normal space
    # On a transposed space, we change the order, so
    # A*B acts as B^T*A^T*v
    # Compare to the kron identity vec(X*A*B) = kron(B^T*A^T, I) * vec(X)
    @spin s 1 // 2
    H = hilbert_space(s)
    Ht = TransposedSpace(H)

    A = sum((n + 1im) * s[n] for n in 0:3)
    @test transpose(matrix_representation(A, H)) ≈ matrix_representation(A, Ht)

    B = sum((n + 2)^2 * s[n] for n in 0:3)
    @test matrix_representation(A * B, H) ≈ matrix_representation(A, H) * matrix_representation(B, H)
    @test matrix_representation(A * B, Ht) ≈ matrix_representation(B, Ht) * matrix_representation(A, Ht)
    @test matrix_representation(A * B, Ht) ≈ transpose(matrix_representation(A * B, H))
    # Test on a mixed space
    @fermions f
    Hf = hilbert_space(f, 1:2)
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1]
    B2 = B * f[1]' * f[1] - I
    @test transpose(matrix_representation(A2, Hfull)) ≈ matrix_representation(A2, Hfull_t)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ transpose(matrix_representation(A2 * B2, Hfull))

    @test matrix_representation(A2 * B2, Hfull) ≈ matrix_representation(A2, Hfull) * matrix_representation(B2, Hfull)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ matrix_representation(B2, Hfull_t) * matrix_representation(A2, Hfull_t)

    # Test with constrained spaces
    _Hf = hilbert_space(f, 1:2)
    Hf = constrain_space(_Hf, collect(basisstates(_Hf))[[2, 3]])
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1] + 0.5 * s[:x] * f[1]' * f[2]
    B2 = B * f[2]' * f[1] - I
    @test transpose(matrix_representation(A2, Hfull)) ≈ matrix_representation(A2, Hfull_t)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ transpose(matrix_representation(A2 * B2, Hfull))
    @test matrix_representation(A2 * B2, Hfull) ≈ matrix_representation(A2, Hfull) * matrix_representation(B2, Hfull)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ matrix_representation(B2, Hfull_t) * matrix_representation(A2, Hfull_t)

    # test with sector space
    Hf = hilbert_space(f, 1:2, NumberConservation(1))
    Hfull = tensor_product(H, Hf)
    Hfull_t = TransposedSpace(Hfull)
    A2 = A * f[1]' * f[1] + 0.5 * s[:x] * f[1]' * f[2]
    B2 = B * f[2]' * f[1] - I
    @test transpose(matrix_representation(A2, Hfull)) ≈ matrix_representation(A2, Hfull_t)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ transpose(matrix_representation(A2 * B2, Hfull))
    @test matrix_representation(A2 * B2, Hfull) ≈ matrix_representation(A2, Hfull) * matrix_representation(B2, Hfull)
    @test matrix_representation(A2 * B2, Hfull_t) ≈ matrix_representation(B2, Hfull_t) * matrix_representation(A2, Hfull_t)

    # test with Majorana fermions
    @majoranas m
    Hm = hilbert_space(m, 1:4)
    Hm_t = TransposedSpace(Hm)
    A = m[1] * m[2] + 0.5 * m[3] * m[4]
    @test transpose(matrix_representation(A, Hm)) ≈ matrix_representation(A, Hm_t)
end

@testitem "Vectorization and kron identity" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: open_system

    @spin s 1 // 2
    H = hilbert_space(s)
    Hfull, Hleft, Hright, left, right = open_system(s)

    A = 1.2 * s[:x] - 0.4im * s[:y] + 0.7 * s[:z]
    B = -0.3 * s[:x] + 1.1im * s[:y] + 0.2 * s[:z]

    MA = matrix_representation(A, H)
    MB = matrix_representation(B, H)

    Msuper = matrix_representation(left(A) * right(B), Hfull)
    @test Msuper ≈ kron(transpose(MB), MA)

    Mright = matrix_representation(right(A * B), Hfull)
    @test Mright ≈ kron(transpose(MA * MB), I(dim(H)))
end

function _apply_local_operators(op, state, space::TransposedSpace, precomp)
    newstate, amp = apply_local_operators(op, state, parent(space), precomp; transpose=true)
    return newstate, amp
end

function apply_local_operators(_op::NCMul, state, space, precomp; transpose)
    if !transpose
        return foldr(_op.factors, init=(state, _op.coeff)) do op, (state, amp)
            newstate, _amp = apply_local_operator(op, state, space, precomp)
            return newstate, amp * _amp
        end
    elseif transpose
        return foldl(_op.factors; init=(state, _op.coeff)) do (state, amp), op
            newstate, _amp = apply_local_operator(op', state, space, precomp)
            return newstate, amp * conj(_amp)
        end
    end
end

maximum_particles(space::TransposedSpace) = maximum_particles(parent(space))
_find_position(f::TransposedSpace, H::FermionicSpace) = _find_position(parent(f), H)