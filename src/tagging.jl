# A tag only makes the group of an operator or a space distinct. Operators on different groups commute, so the
# algebra of the tagged operators needs no rules for it.

"""
    TaggedId(id, tags)

The group or atom id `id` of a [`TaggedSym`](@ref) or [`TaggedSpace`](@ref) with its `tags`.
"""
struct TaggedId{I, T <: Tuple}
    id::I
    tags::T
end
Base.:(==)(a::TaggedId, b::TaggedId) = a.id == b.id && a.tags == b.tags
Base.hash(a::TaggedId, h::UInt) = hash(a.tags, hash(a.id, hash(TaggedId, h)))
Base.isless(a::TaggedId, b::TaggedId) = _order_isless((a.id, a.tags), (b.id, b.tags))

"""
    TaggedSym(op, tags)

The operator `op` on a copy of its space told apart by `tags`, which `add_tag` makes for operator types without their
own method. It multiplies with an operator of the same tags as `op` does with that operator's untagged version, and
commutes with all other operators.
"""
struct TaggedSym{O <: AbstractSym, T <: Tuple} <: AbstractSym
    op::O
    tags::T
end
@nc TaggedSym
add_tag(op::AbstractSym, tag) = TaggedSym(op, (tag,))
add_tag(op::TaggedSym, tag) = TaggedSym(op.op, (op.tags..., tag))
symbolic_group(op::TaggedSym) = TaggedId(symbolic_group(op.op), op.tags)
atomic_id(op::TaggedSym) = TaggedId(atomic_id(op.op), op.tags)
Base.adjoint(op::TaggedSym) = TaggedSym(adjoint(op.op), op.tags)
Base.:(==)(a::TaggedSym, b::TaggedSym) = a.op == b.op && a.tags == b.tags
Base.hash(a::TaggedSym, h::UInt) = hash(a.tags, hash(a.op, h))
mat_eltype(::Type{TaggedSym{O, T}}) where {O, T} = mat_eltype(O)

local_mul_effect(a::TaggedSym, b::TaggedSym) = _tag_effect(NonCommutativeProducts.mul_effect(a.op, b.op), a.tags)
_tag_effect(effect::Union{Nothing, Number, Swap}, tags) = effect
_tag_effect(effect::AddTerms, tags) = AddTerms(map(term -> _tag_effect(term, tags), effect.terms))
_tag_effect(effect, tags) = NonCommutativeProducts.ncmap(op -> TaggedSym(op, tags), effect)

"""
    TaggedSpace(H, tags)

The atomic space `H` with its id told apart by `tags`, which `add_tag` makes for atomic spaces without their own
method. It has the basis states of `H`, and a [`TaggedSym`](@ref) with the same tags acts on it as its untagged
operator acts on `H`.
"""
struct TaggedSpace{B, H <: AbstractAtomicHilbertSpace{B}, T <: Tuple} <: AbstractAtomicHilbertSpace{B}
    parent::H
    tags::T
end
add_tag(H::AbstractAtomicHilbertSpace, tag) = TaggedSpace(H, (tag,))
add_tag(H::TaggedSpace, tag) = TaggedSpace(H.parent, (H.tags..., tag))
Base.parent(H::TaggedSpace) = H.parent
atomic_id(H::TaggedSpace) = TaggedId(atomic_id(H.parent), H.tags)
symbolic_group(H::TaggedSpace) = TaggedId(symbolic_group(H.parent), H.tags)
Base.:(==)(a::TaggedSpace, b::TaggedSpace) = a.parent == b.parent && a.tags == b.tags
Base.hash(H::TaggedSpace, h::UInt) = hash(H.tags, hash(H.parent, h))
basisstates(H::TaggedSpace) = basisstates(H.parent)
Base.@propagate_inbounds basisstate(i::Integer, H::TaggedSpace) = basisstate(i, H.parent)
_state_index(state, H::TaggedSpace) = _state_index(state, H.parent)
has_arithmetic_state_index(H::TaggedSpace) = has_arithmetic_state_index(H.parent)
dim(H::TaggedSpace) = dim(H.parent)
interpret_state(state, H::TaggedSpace) = interpret_state(state, H.parent)

_precomputation_before_operator_application(op::TaggedSym, space::TaggedSpace) =
    _precomputation_before_operator_application(op.op, space.parent)
_precomputation_before_operator_application(op::TaggedSym, space::Union{TransposedSpace, AbstractSubspace}) =
    _precomputation_before_operator_application(op, parent(space))
apply_local_operator(op::TaggedSym, state, space::TaggedSpace, precomp) = apply_local_operator(op.op, state, space.parent, precomp)
apply_local_operator(op::TaggedSym, state, space::TaggedSpace, precomp, ::Type{P}) where {P} =
    apply_local_operator(op.op, state, space.parent, precomp, P)

function Base.show(io::IO, op::TaggedSym)
    show(io, op.op)
    return print(io, "(", join(op.tags, ","), ")")
end
function Base.show(io::IO, H::TaggedSpace)
    lines = split(sprint(show, H.parent; context = io), '\n')
    lines[1] *= string(" (", join(H.tags, ","), ")")
    return print(io, join(lines, '\n'))
end

@testitem "Tagging custom operators and spaces" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: AbstractSym, AbstractBasisState, symbolic_group, local_mul_effect,
        apply_local_operator, mat_eltype, add_tag, open_system, dissipator
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc, Swap, AddTerms
    # Shifts on a cycle of length 3 with S(k) * S(m) = S(k + m), and an observable N with N * S(k) = S(k) * N + k * S(k)
    struct Shift <: AbstractSym
        site::Any
        k::Int
    end
    struct Num <: AbstractSym
        site::Any
    end
    struct Phase <: AbstractSym
        site::Any
    end
    @nc Shift Num Phase
    Base.adjoint(s::Shift) = Shift(s.site, -s.k)
    Base.adjoint(n::Num) = n
    Base.adjoint(p::Phase) = p
    symbolic_group(op::Union{Shift, Num, Phase}) = op.site
    local_mul_effect(a::Shift, b::Shift) = Shift(a.site, a.k + b.k)
    local_mul_effect(a::Num, b::Shift) = AddTerms((Swap(1), b.k * b))
    struct Site <: AbstractBasisState
        n::Int
    end
    apply_local_operator(s::Shift, state::Site, space, precomp) = (Site(mod(state.n + s.k, 3)), 1)
    apply_local_operator(::Num, state::Site, space, precomp) = (state, state.n)
    apply_local_operator(::Phase, state::Site, space, precomp) = (state, im^state.n)
    apply_local_operator(s::Shift, n::Int, space, precomp) = (n + s.k, 1)
    mat_eltype(::Type{Shift}) = Int
    mat_eltype(::Type{Num}) = Int
    mat_eltype(::Type{Phase}) = ComplexF64

    l(x) = add_tag(x, :l)
    r(x) = add_tag(x, :r)
    @test l(Shift(1, 1)) * l(Shift(1, 1)) == l(Shift(1, 2))
    @test l(Num(1)) * l(Shift(1, 1)) == l(Shift(1, 1)) * l(Num(1)) + l(Shift(1, 1))
    @test l(Num(1)) * r(Shift(1, 1)) == r(Shift(1, 1)) * l(Num(1))
    @test l(r(Shift(1, 1))) != r(l(Shift(1, 1)))

    # A product of different operator types has the element type of its factors
    Hs = GenericHilbertSpace(1, Site.(0:2))
    M = representation(Phase(1) * Shift(1, 1), Hs, Matrix)
    @test M ≈ Diagonal(im .^ (0:2)) * representation(Shift(1, 1), Hs, Matrix)
    @test representation(l(Phase(1) * Shift(1, 1)), l(Hs), Matrix) == M

    # Kets of a tagged space, and a constrained space with states that are not `AbstractBasisState`s
    @test representation(l(Kets(Hs)(Site(1))), l(Hs)) == representation(Kets(Hs)(Site(1)), Hs)
    Hi = GenericHilbertSpace(3, 0:2)
    Hc = constrain_space(Hi, SectorConstraint(v -> sum(v) <= 1 ? 0 : 1; maps = identity, spaces = (Hi,)))
    MS = representation(Shift(3, 1), Hc, Matrix; projection = true)
    @test representation(l(Shift(3, 1)), l(Hc), Matrix; projection = true) == MS
    Hlr, Hl, Hr, left, right = open_system(Hc)
    ρ = randn(dim(Hc), dim(Hc))
    S = representation(left(Shift(3, 1)) * right(Shift(3, 1)), Hlr; projection = true)
    @test reshape(S * reshape(ρ, (Hl, Hr) => Hlr), Hlr => (Hl, Hr)) ≈ MS * ρ * MS

    for labels in ((1, 2), (:a, :b))
        H = tensor_product([GenericHilbertSpace(k, Site.(0:2)) for k in labels])
        Hlr, Hl, Hr, left, right = open_system(H)
        A = Shift(labels[1], 1) * Num(labels[2]) + 2 * Num(labels[1])
        B = Shift(labels[2], -1) + Shift(labels[1], 1) * Shift(labels[2], 1)
        MA, MB = representation(A, H, Matrix), representation(B, H, Matrix)
        ρ = randn(ComplexF64, dim(H), dim(H))
        v = reshape(ρ, (Hl, Hr) => Hlr)
        S = representation(left(A) * right(B), Hlr)
        @test reshape(S * v, Hlr => (Hl, Hr)) ≈ MA * ρ * MB
        L = representation(-1im * (left(A + A') - right(A + A')) + dissipator(B), Hlr)
        @test abs(tr(reshape(L * v, Hlr => (Hl, Hr)))) < 1.0e-10
    end
end
