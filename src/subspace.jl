"""
    AbstractSubspace{B,P} <: AbstractHilbertSpace{B}

A space spanned by some of the basis states of `parent(H)::P`. A subtype implements

- `parent(H)`: the space that `H` is a subspace of. It is never an `AbstractSubspace`.
- `subspace_states(H)`: the basis states of `H`, in order, each appearing once.
- `subspace_index(H)`: an index with a [`state_position`](@ref) method for the states of `H`,
  such as `default_subspace_index(constraint, parent(H), subspace_states(H))`, built once by
  the constructor.
- `rewrap(H, newparent)`: the space with the same states and index as `H` over `newparent`,
  which has the same basis states and `state_index` as `parent(H)`.

It may also implement `subspace_key(H)`, the values compared by `==` and `hash`, which
defaults to `(parent(H), subspace_states(H))`, and `show_details(io, H)`, printed after the
parent by `show`.

The states are meant to be basis states of `parent(H)`, but this is not checked. Some
spaces store other states over a `FermionicSpace` parent: `hilbert_space(f, labels, states)`
with `FixedNumberFockState`s, `BdG` spaces with `NambuState`s and `SingleParticle` spaces
with `SingleParticleState`s.

A subtype gets `dim`, `basisstates`, `basisstate`, `state_index`, `==`, `hash`, `show`,
`add_tag` and `TransposedSpace`. The functions that describe its modes and apply operators
are forwarded to `parent(H)`, so that matrix representations, tensor products, partial
traces, `subregion` and `constrain_space` work on it.
"""
abstract type AbstractSubspace{B,P} <: AbstractHilbertSpace{B} end

"""
    state_position(index, state, parent)

The position of `state` in the subspace that `index` was built for, or `0` if it is not in it.
This is the only method a subspace index implements. It may use `parent` only through its basis
states and `state_index`, because `add_tag` and `TransposedSpace` keep the index but replace the
parent.
"""
function state_position end

"""
    default_subspace_index(constraint, parent, states)

Build the index (see [`state_position`](@ref)) of the subspace of `parent` with basis `states`,
in order. `constraint` is the constraint that selected `states`: the one given to
`hilbert_space`, `constrain_space`, `tensor_product` or `sector`, or for a `tensor_product` of
sector spaces without a new constraint, the one built from the constraints of the factors. It is
`NoSymmetry()` otherwise, for example for listed states and for `sector` without a constraint.
The generic method returns a lookup table or a `Dict`, and throws an `ArgumentError` on
repeated states.

To specialize, add a method that dispatches on the type of `constraint` only, since restricting
`parent` as well makes extensions ambiguous with each other. When `states` don't fit the custom
index, for example when they are not a whole sector in the expected order, return
`default_subspace_index(NoSymmetry(), parent, states)`.
"""
default_subspace_index(constraint, parent, states) = generic_subspace_index(parent, states)
function generic_subspace_index(parent, states)
    table = index_table(parent, states)
    isnothing(table) ? position_dict(states) : table
end
generic_subspace_index(::AbstractSubspace, states) = throw(nested_subspace_error())
nested_subspace_error() = ArgumentError("The parent of a subspace cannot be a subspace.")

# `space` and `states`, or, when `space` is a subspace, its parent and those of `states` in it.
unnest(space, states) = (space, states)
unnest(space::AbstractSubspace, states) = (parent(space), collect(Iterators.filter(s -> !iszero(state_index(s, space)), states)))

subspace_key(H::AbstractSubspace) = (parent(H), subspace_states(H))
Base.:(==)(H1::AbstractSubspace, H2::AbstractSubspace) = H1 === H2 || (_type_family(typeof(H1)) === _type_family(typeof(H2)) && subspace_key(H1) == subspace_key(H2))
Base.hash(H::AbstractSubspace, h::UInt) = hash(subspace_key(H), h)

dim(H::AbstractSubspace) = length(subspace_states(H))
basisstates(H::AbstractSubspace) = BasisStates(H, subspace_states(H))
Base.@propagate_inbounds basisstate(n::Integer, H::AbstractSubspace) = subspace_states(H)[n]
_state_index(state, H::AbstractSubspace) = state_position(subspace_index(H), _check_state_kind(state, H), parent(H))
isconstrained(H::AbstractSubspace) = true

for f in (:atomic_factors, :groups, :factors, :atom_ids, :group_id, :partial_trace_phase_factor_eltype,
    :partial_transpose_phase_factor_eltype, :fast_path, :mode_ordering)
    @eval $f(H::AbstractSubspace) = $f(parent(H))
end
atomic_factor(H::AbstractSubspace, i) = atomic_factor(parent(H), i)
atom_position(x, H::AbstractSubspace) = atom_position(x, parent(H))
combine_states(substates, H::AbstractSubspace) = combine_states(substates, parent(H))
partial_trace_phase_factor(s1, s2, H::AbstractSubspace) = partial_trace_phase_factor(s1, s2, parent(H))
partial_transpose_phase_factor(s1, s2, H::AbstractSubspace) = partial_transpose_phase_factor(s1, s2, parent(H))
state_mapper(H::AbstractSubspace, Hs) = state_mapper(parent(H), Hs)
_apply_local_operators(ops, state::AbstractBasisState, H::AbstractSubspace, precomp) = _apply_local_operators(ops, state, parent(H), precomp)

# The subspace `H` with `f` applied to its parent and to the spaces its constraint refers to; `f` must commute with taking factors, as tagging and transposing do.
map_parent(f, H::AbstractSubspace) = rewrap(H, f(parent(H)))
add_tag(H::AbstractSubspace, tag) = map_parent(Base.Fix2(add_tag, tag), H)

@testitem "Nested subspaces equal the directly constructed ones" begin
    using FermionicHilbertSpaces: fermionnumber
    @fermions f
    H = hilbert_space(f, 1:3)
    states = collect(basisstates(H))[[8, 2, 5, 3, 1]]

    # a sector space of a constrained space
    Hsector = constrain_space(constrain_space(H, states), NumberConservation())
    @test Hsector == constrain_space(H, NumberConservation(), states)
    @test parent(Hsector) == H
    @test hash(Hsector) == hash(constrain_space(H, NumberConservation(), states))

    # a constrained space of a sector space
    Hconstrained = constrain_space(hilbert_space(f, 1:3, NumberConservation(1:2)), states)
    @test Hconstrained == constrain_space(H, filter(s -> fermionnumber(s) in 1:2, states))
    @test parent(Hconstrained) == H
    @test hash(Hconstrained) == hash(constrain_space(H, filter(s -> fermionnumber(s) in 1:2, states)))
end

@testitem "A subtype implementing only the AbstractSubspace primitives" begin
    using FermionicHilbertSpaces: AbstractSubspace, default_subspace_index
    import FermionicHilbertSpaces: subspace_states, subspace_index, rewrap
    struct EveryOtherState{B,P,I} <: AbstractSubspace{B,P}
        space::P
        states::Vector{B}
        lookup::I
    end
    function EveryOtherState(space)
        states = collect(basisstates(space))[1:2:end]
        EveryOtherState(space, states, default_subspace_index(NoSymmetry(), space, states))
    end
    Base.parent(H::EveryOtherState) = H.space
    subspace_states(H::EveryOtherState) = H.states
    subspace_index(H::EveryOtherState) = H.lookup
    rewrap(H::EveryOtherState, space) = EveryOtherState(space, H.states, H.lookup)

    @fermions f
    H = hilbert_space(f, 1:3)
    Hsub = EveryOtherState(H)
    @test dim(Hsub) == 4
    @test all(state_index(basisstate(i, Hsub), Hsub) == i for i in 1:dim(Hsub))
    op = f[2]' * f[3] + f[3]' * f[2] + 2 * f[2]' * f[2]
    @test matrix_representation(op, Hsub) == matrix_representation(op, H)[1:2:end, 1:2:end]
end

@testitem "Custom subspace indices" begin
    import FermionicHilbertSpaces: state_position, default_subspace_index
    struct LinearSearch{V}
        states::V
    end
    state_position(index::LinearSearch, state, parent) = something(findfirst(isequal(state), index.states), 0)

    @fermions f
    H = hilbert_space(f, 1:4)
    op = sum(f[i]' * f[i+1] + hc for i in 1:3) + 2 * f[1]' * f[1]
    observed(Hsub) = (map(s -> state_index(s, Hsub), basisstates(H)), matrix_representation(op, Hsub; projection=true))
    generic(Hsub) = constrain_space(H, collect(basisstates(Hsub)))

    selected = Ref(0)
    keep(state) = isodd(state.f)
    label(state) = isodd(state.f)
    function default_subspace_index(::Union{FilterConstraint{typeof(keep)},SectorConstraint{<:FilterConstraint{typeof(label)}}}, parent, states)
        selected[] += 1
        LinearSearch(states)
    end
    for build in (() -> constrain_space(H, FilterConstraint(keep)), () -> constrain_space(H, SectorConstraint(label)),
        () -> tensor_product(hilbert_space(f, 1:2), hilbert_space(f, 3:4); constraint=FilterConstraint(keep)))
        before = selected[]
        Hsub = build()
        @test selected[] > before
        @test observed(Hsub) == observed(generic(Hsub))
    end

    wrong(state) = iseven(state.f)
    default_subspace_index(::FilterConstraint{typeof(wrong)}, parent, states) = LinearSearch(reverse(states))
    @test_throws ArgumentError constrain_space(H, FilterConstraint(wrong))
end

@testitem "Subspaces reject duplicate states" begin
    @fermions f
    H = hilbert_space(f, 1:2)
    states = collect(basisstates(H))
    @test_throws ArgumentError constrain_space(H, states[[1, 2, 1]])
    @test_throws ArgumentError constrain_space(H, NumberConservation(), states[[1, 2, 1]])
    @test_throws ArgumentError constrain_space(GenericHilbertSpace(:A, [:a, :b, :c]), [:a, :a])
end
