
# Misc

## Subregion
The function subregion can be used to extract the Hilbert space of a subregion.
```@example subregion
using FermionicHilbertSpaces
H = hilbert_space(1:4)
Hsub = subregion(1:2, H)
``` 

When the Hilbert space has a restricted set of fock states, the Hilbert space of the subregion will only include fock states compatible with this restriction. In the example below, the total number of particles 1, and the subregion will have three possible states: (1,0), (0,1), and (0,0).
```@example subregion
H = hilbert_space(1:4, number_conservation(; sectors = 1))
Hsub = subregion(1:2, H)
basisstates(Hsub)
``` 