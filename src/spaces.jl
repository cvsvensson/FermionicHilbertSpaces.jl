## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractFockHilbertSpace{F<:AbstractFockState} <: AbstractHilbertSpace{F} end
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


abstract type AbstractStateSplitter end
"""
    state_splitter(H, Hs)

Create a splitter object for decomposing states in `H` into the target subsystems `Hs`.
The returned object must subtype `AbstractStateSplitter`.
"""
state_splitter(H, Hs) = throw(MethodError(state_splitter, (H, Hs)))

"""
    split_state(state, splitter)

Split a state using `splitter`.

Contract: returns a tuple with one entry per target subsystem. Each entry is a weighted
collection `((substate, weight), ...)` for that target.
"""
split_state(state, splitter::AbstractStateSplitter) = throw(MethodError(split_state, (state, splitter)))

"""
    combine_states(substates, splitter)

Combine subsystem states using `splitter`.

Contract: returns a weighted collection `((state, weight), ...)`.
"""
combine_states(substates, splitter::AbstractStateSplitter) = throw(MethodError(combine_states, (substates, splitter)))

"""
    kron_phase_factor(splitter)

Return phase-factor function associated with `splitter` for fermionic tensor products.
"""
kron_phase_factor(splitter::AbstractStateSplitter) = throw(MethodError(kron_phase_factor, (splitter,)))

struct AtomicStateSplitter <: AbstractStateSplitter end
function state_splitter(H::AbstractAtomicHilbertSpace, Hs)
    only(Hs) == H || throw(ArgumentError("For atomic subspaces, the only valid partition is the whole space"))
    AtomicStateSplitter()
end
function split_state(state, ::AtomicStateSplitter)
    (((state, 1),),)
end
function combine_states(states, ::AtomicStateSplitter)
    ((only(states), 1),)
end
kron_phase_factor(::AtomicStateSplitter) = (f1, f2) -> 1

