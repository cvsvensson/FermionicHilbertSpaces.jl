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

partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1
