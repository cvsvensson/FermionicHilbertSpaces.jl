"""
    AbstractSubspace{B,P} <: AbstractHilbertSpace{B}

A space spanned by a subset of the basis states of `parent(H)::P`, which is never itself an
`AbstractSubspace`. A subtype has the fields `parent` and `index`, where
`index = state_positions(parent, subspace_states(H))`, and implements

- `subspace_states(H)`: the basis states of `H`, in order.
- `rewrap(H, newparent, index)`: the space with the same states as `H` over `newparent`.

It may also implement `subspace_key(H)`, the values compared by `==` and `hash`, which
defaults to `(parent(H), subspace_states(H))`, and `_show_details(io, H)`, printed after the
parent by `show`.
"""
abstract type AbstractSubspace{B,P} <: AbstractHilbertSpace{B} end

Base.parent(H::AbstractSubspace) = H.parent
subspace_key(H::AbstractSubspace) = (parent(H), subspace_states(H))
Base.:(==)(H1::AbstractSubspace, H2::AbstractSubspace) = H1 === H2 || (nameof(typeof(H1)) === nameof(typeof(H2)) &&subspace_key(H1) == subspace_key(H2))
Base.hash(H::AbstractSubspace, h::UInt) = hash(subspace_key(H), h)

dim(H::AbstractSubspace) = length(subspace_states(H))
basisstates(H::AbstractSubspace) = BasisStates(H, subspace_states(H))
Base.@propagate_inbounds basisstate(n::Integer, H::AbstractSubspace) = subspace_states(H)[n]
state_index(state, H::AbstractSubspace) = state_position(H.index, state, H)
state_index(state::Union{AbstractString,AbstractChar}, H::AbstractSubspace) = state_index(interpret_state(state, H), H)
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

add_tag(H::AbstractSubspace, tag) = rewrap(H, add_tag(parent(H), tag), H.index)
_wrap(space, H::AbstractSubspace) = rewrap(H, space, state_positions(space, subspace_states(H)))

@testitem "Nested subspaces equal the directly constructed ones" begin
    using FermionicHilbertSpaces: fermionnumber
    @fermions f
    H = hilbert_space(f, 1:3)
    states = collect(basisstates(H))[[8, 2, 5, 3, 1]]

    # a sector space of a constrained space
    Hsector = constrain_space(constrain_space(H, states), NumberConservation())
    @test Hsector == constrain_space(H, NumberConservation(), states)
    @test parent(Hsector) == H

    # a constrained space of a sector space
    Hconstrained = constrain_space(hilbert_space(f, 1:3, NumberConservation(1:2)), states)
    @test Hconstrained == constrain_space(H, filter(s -> fermionnumber(s) in 1:2, states))
    @test parent(Hconstrained) == H
end

@testitem "A subtype implementing only the AbstractSubspace primitives" begin
    using FermionicHilbertSpaces: AbstractSubspace, state_positions, state_index, basisstate
    import FermionicHilbertSpaces: subspace_states, rewrap
    struct EveryOtherState{B,P} <: AbstractSubspace{B,P}
        parent::P
        states::Vector{B}
        index::Union{Vector{Int32},Dict{B,Int}}
    end
    function EveryOtherState(parent)
        states = collect(basisstates(parent))[1:2:end]
        EveryOtherState(parent, states, state_positions(parent, states))
    end
    subspace_states(H::EveryOtherState) = H.states
    rewrap(H::EveryOtherState, parent, index) = EveryOtherState(parent, H.states, index)

    @fermions f
    H = hilbert_space(f, 1:3)
    Hsub = EveryOtherState(H)
    @test dim(Hsub) == 4
    @test all(state_index(basisstate(i, Hsub), Hsub) == i for i in 1:dim(Hsub))
    op = f[2]' * f[3] + f[3]' * f[2] + 2 * f[2]' * f[2]
    @test matrix_representation(op, Hsub) == matrix_representation(op, H)[1:2:end, 1:2:end]
end
