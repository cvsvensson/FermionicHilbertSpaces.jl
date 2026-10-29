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

`default_subspace_index` throws when its `parent` is an `AbstractSubspace`, unless a more
specific method applies, so a constructor that builds the index as above keeps `parent(H)`
from being a subspace. The package calls
`rewrap` only with `newparent` made from `parent(H)` by `add_tag`, `TransposedSpace` or a
Majorana space, none of which is a subspace.

A subtype gets `dim`, `basisstates`, `basisstate`, `state_index`, `==`, `hash`, `show`,
`add_tag` and `TransposedSpace`. The functions that describe its modes and apply operators
are forwarded to `parent(H)`, so that matrix representations, tensor products, partial
traces, `subregion` and `constrain_space` work on it.
"""
abstract type AbstractSubspace{B,P} <: AbstractHilbertSpace{B} end

"""
    state_position(index, state, parent)

The position of `state` among the states of the subspace over `parent` that `index` was built
for, or `0` if `state` is not one of them. `state` has the type of those states up to type
parameters. A subspace keeps its index when `add_tag` or `TransposedSpace` replaces its parent
by one with the same basis states and `state_index`, so the result may depend on `parent` only
through those.

This is the one method a subspace index implements. `state_index(state, H)` of an
[`AbstractSubspace`](@ref) `H` returns `state_position(subspace_index(H), state, parent(H))`.
"""
function state_position end

"""
    default_subspace_index(constraint, parent, states)

The index, with a [`state_position`](@ref) method, of the subspace with `states`, in order,
over `parent`, where `constraint` describes what selected `states`. Constrained and sector
spaces build their index with it, and `constrain_space`, `sector`, `sectors` and
`tensor_product` accept a function of the same arguments as their `index` keyword instead.
The constructor checks the index on a few of `states`.

`constraint` arrives in the form given by `index_constraint`. A product of constraints is a
`ProductConstraint` of a tuple. A `NumberConservation` or `ParityConservation` lists, as its
`spaces`, the atoms of `parent` it acts on. The constraints of the constrained factors of a
tensor product are included, each acting on its factor. `NoSymmetry()` means that `states`
were listed explicitly or that nothing is known about them. The sectors of a sector space
appear as the constraint itself; a quantum number is dispatchable only through its
constraint type, `NumberConservation`, `ParityConservation`, or `SectorConstraint` through
the type of its function.

Extend it by adding a method that dispatches on the type of `constraint` alone, and query
`parent` and `states` in its body; methods that also restrict `parent` would be ambiguous
with those of other extensions. Such a method checks that its index fits, such as that
`states` are a whole sector in the order the index assumes, and otherwise returns
`default_subspace_index(NoSymmetry(), parent, states)`.

The generic method returns a table over `1:dim(parent)` when `parent` has an arithmetic
`state_index` and is not much larger than `states`, and a `Dict` from state to position
otherwise. It throws an `ArgumentError` if a state appears more than once, or if `parent` is
an `AbstractSubspace`.
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

add_tag(H::AbstractSubspace, tag) = rewrap(H, add_tag(parent(H), tag))

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

    states = collect(basisstates(H))[[9, 2, 12, 5, 3]]
    by_keyword(constraint, parent, states) = LinearSearch(states)
    for constraint in (states, NumberConservation(1:2))
        Hsub = constrain_space(H, constraint; index=by_keyword)
        @test observed(Hsub) == observed(generic(Hsub))
        @test observed(sectors(Hsub; index=by_keyword)[end]) == observed(generic(sectors(Hsub)[end]))
    end
    wrong(constraint, parent, states) = LinearSearch(reverse(states))
    @test_throws ArgumentError constrain_space(H, states; index=wrong)

    selected = Ref(0)
    keep(state) = isodd(state.f)
    label(state) = isodd(state.f)
    function default_subspace_index(::Union{FilterConstraint{typeof(keep)},FilterConstraint{<:Any,<:Tuple{typeof(keep)}},SectorConstraint{<:FilterConstraint{typeof(label)}}}, parent, states)
        selected[] += 1
        LinearSearch(states)
    end
    Hlabel = constrain_space(H, SectorConstraint(label))
    Hkeep = constrain_space(hilbert_space(f, 1:2), FilterConstraint(keep))
    for build in (() -> constrain_space(H, FilterConstraint(keep)), () -> sector(true, Hlabel), () -> tensor_product(Hkeep, hilbert_space(f, 3:4)))
        before = selected[]
        Hsub = build()
        @test selected[] > before
        @test observed(Hsub) == observed(generic(Hsub))
    end
end

@testitem "Subspaces reject duplicate states" begin
    @fermions f
    H = hilbert_space(f, 1:2)
    states = collect(basisstates(H))
    @test_throws ArgumentError constrain_space(H, states[[1, 2, 1]])
    @test_throws ArgumentError constrain_space(H, NumberConservation(), states[[1, 2, 1]])
    @test_throws ArgumentError constrain_space(GenericHilbertSpace(:A, [:a, :b, :c]), [:a, :a])
end
