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
