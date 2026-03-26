## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractAtomicHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractProductHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractClusterHilbertSpace{B} <: AbstractProductHilbertSpace{B} end

"""
	basisstates(H)

Return an iterable of basis states for the Hilbert space `H`, in the order used by
matrix representations and indexing utilities.
"""
basisstates

"""
	hilbert_space(args...)

Construct a Hilbert space from symbolic degrees of freedom and labels, optionally
with additional arguments such as constraints.
"""
hilbert_space

factors(H::AbstractAtomicHilbertSpace) = (H,)
clusters(H::AbstractAtomicHilbertSpace) = (H,)
atomic_factors(H::AbstractAtomicHilbertSpace) = (H,)
factors(H::AbstractClusterHilbertSpace) = atomic_factors(H)
clusters(H::AbstractClusterHilbertSpace) = (H,)
atomic_substate(n, f, ::AbstractClusterHilbertSpace) = substate(n, f)
isconstrained(H::AbstractAtomicHilbertSpace) = false

partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1


abstract type AbstractStateMapper end
"""
    state_mapper(H, Hs)

Create a mapper object for decomposing states in `H` into the target subsystems `Hs`.
The returned object must subtype `AbstractStateMapper`.
"""
state_mapper(H, Hs) = throw(MethodError(state_mapper, (H, Hs)))

"""
    split_state(state, mapper)

Split a state using `mapper`.

Contract: returns a tuple with one entry per target subsystem. Each entry is a weighted
collection `((substate, weight), ...)` for that target.
"""
split_state(state, mapper::AbstractStateMapper) = throw(MethodError(split_state, (state, mapper)))

"""
    combine_states(substates, mapper)

Combine subsystem states using `mapper`.

Contract: returns a states and weights `states, weights`.
"""
combine_states(substates, mapper::AbstractStateMapper) = throw(MethodError(combine_states, (substates, mapper)))

"""
    kron_phase_factor(mapper)

Return phase-factor function associated with `mapper` for fermionic tensor products.
"""
kron_phase_factor(mapper::AbstractStateMapper) = throw(MethodError(kron_phase_factor, (mapper,)))

struct AtomicStateMapper <: AbstractStateMapper end
function state_mapper(H::AbstractAtomicHilbertSpace, Hs)
    only(Hs) == H || throw(ArgumentError("For atomic subspaces, the only valid partition is the whole space"))
    AtomicStateMapper()
end
function split_state(state, ::AtomicStateMapper)
    # the return state here is a tuple of tuples because each state is a state in the tuple of subsystem and complement, but the complement is empty
    ((state,),), (1,)
end
function combine_states(states, ::AtomicStateMapper)
    (only(states),), (1,)
end
kron_phase_factor(::AtomicStateMapper) = (f1, f2) -> 1
