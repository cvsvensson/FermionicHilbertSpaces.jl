
# Misc

## Subregion
When the Hilbert space has a restricted set of fock states, the Hilbert space of the subregion will only include fock states compatible with this restriction. In the example below, the total number of particles 1, and the subregion will have three possible states: (1,0), (0,1), and (0,0).
```@example subregion
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(1))
Hsub = subregion(hilbert_space(f, 1:2), H)
basisstates(Hsub)
``` 

## State mapper interface

Internal tensor/reshape/partial-trace routines use a common mapper protocol:

- `state_mapper(H, Hs)` returns a mapper object.
- `split_state(state, mapper)` returns a tuple with one entry per target subsystem.
- Each tuple entry is a weighted collection `((substate, weight), ...)`.
- `combine_states(substates, mapper)` returns a weighted collection `((state, weight), ...)`.

This package does not require a single concrete container type for weighted collections; callers should treat them as iterable collections of `(state, weight)` outcomes.