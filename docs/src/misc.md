
# Misc

## Single particle hilbert spaces
A quadratic fermionic hamiltonian with number conservation can be written as 
```math
H = E\mathbf{1} + \sum_{ij} c_i^\dagger\, h_{ij}  c_j.
```
The matrix element between two single particle states with modes $n$ and $m$ occupied is
```math
 \bra{m}H\ket{n} = E\delta_{nm} + h_{nm}.
```

One can manually define the hilbert space using only the single particle states as
```@example single_particle_hilbert_space
using FermionicHilbertSpaces, LinearAlgebra
N = 2
H = hilbert_space(1:N, FermionicHilbertSpaces.SingleParticleState.(1:N))
@fermions c
h = rand(N,N)
E = rand()
op = E*I + sum(c[i]'*h[i,j] * c[j] for i in 1:N, j in 1:N)
matrix_representation(op,H) == h + E*I
```
Often, $h_{nm}$ is of interest because diagonalizing it gives information on the quasiparticles in the system.

For convenience, `single_particle_hilbert_space` can be used define the hilbert space which will give only the single particle states, and will remove the contribution of the identity operator when calling `matrix_representation`:
```@example single_particle_hilbert_space
H = single_particle_hilbert_space(1:N)
matrix_representation(op,H) == h
```

## Subregion
The function subregion can be used to extract the Hilbert space of a subregion.
```@example subregion
using FermionicHilbertSpaces
H = hilbert_space(1:4)
Hsub = subregion(1:2, H)
``` 

When the Hilbert space has a restricted set of fock states, the Hilbert space of the subregion will only include fock states compatible with this restriction. In the example below, the total number of particles 1, and the subregion will have three possible states: (1,0), (0,1), and (0,0).
```@example subregion
H = hilbert_space(1:4, FermionConservation(1))
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
Hs = [hilbert_space([(k, s) for s in spin], FermionConservation(0:1)) for k in space]
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

