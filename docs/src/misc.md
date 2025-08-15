
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
H = hilbert_space(1:4, NumberConservation(1))
Hsub = subregion(1:2, H)
basisstates(Hsub)
``` 

## No double occupation

The following example demonstrates how to restrict the Hilbert space to only allow single occupation of a site, which can be useful for simulating systems where the on-site coulomb interaction is strong enough to prevent double occupation.

```@example double_occupation
using FermionicHilbertSpaces, LinearAlgebra
N = 2 # number of fermions
space = 1:N 
spin = (:↑,:↓)
# labels = Base.product(space, spin) 
Hs = [hilbert_space([(k, s) for s in spin], NumberConservation(0:1)) for k in space]
H = tensor_product(Hs)
```

```@example double_occupation
size(H,1) == 3^N
```

Can also take product of symmetries
```@example double_occupation
qn2 = prod(IndexConservation(k,0:1) for k in space)
H2 = hilbert_space(keys(H), qn2)
```

This gives the same states but with a different ordering.
```@example double_occupation
sort(basisstates(H2)) == sort(basisstates(H))
```

