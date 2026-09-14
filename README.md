# FermionicHilbertSpaces.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
[![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)

This package provides tools for working with operators on hilbert spaces, with automatic handling of fermionic statistics. This includes:
- Tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
- Symbolic algebra and matrix representations of operators on hilbert spaces.

The package includes fermions, bosons and spins, and can be easily extended with custom algebras, hilbert spaces and matrix representations. There is a flexible system to constrain hilbert spaces to subspaces with conserved quantities.

## Quick example
Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.

````julia
using FermionicHilbertSpaces, LinearAlgebra
@fermions f # Defines a symbolic fermion
sym_ham = sum(rand() * f[n]'f[n] for n in 1:4) +
          sum(f[n+1]'f[n] + hc for n in 1:3)

#Get a matrix representation of the hamiltonian on a hilbert space
H = hilbert_space(f, 1:4)
ham = matrix_representation(sym_ham, H)

#Diagonalize to find the ground state
Ψ = eigvecs(Matrix(ham))[:, 1]

#Define a subsystem and partial trace to find the reduced density matrix
Hsub = hilbert_space(f, 1:2)
ρsub = partial_trace(Ψ, H => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ + 1e-16), eigvals(ρsub))
````

````
0.44395495955661934
````

### Conserved quantities
The hamiltonian above conserves the number of fermions, which we can exploit as

````julia
Hcons = hilbert_space(f, 1:4, NumberConservation(2))
````

````
6-dimensional SectorHilbertSpace
Parent: (f[1, 2, 3, 4])
Sectors: [2 (6-dim)]
````

This hilbert space contains only states with two fermions. We can use it just as before to get a matrix representation of the hamiltonian

````julia
ham = matrix_representation(sym_ham, Hcons)
````

````
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 18 stored entries:
 0.422608  1.0        ⋅        ⋅         ⋅         ⋅ 
 1.0       0.772193  1.0      1.0        ⋅         ⋅ 
  ⋅        1.0       1.04807   ⋅        1.0        ⋅ 
  ⋅        1.0        ⋅       0.701631  1.0        ⋅ 
  ⋅         ⋅        1.0      1.0       0.977506  1.0
  ⋅         ⋅         ⋅        ⋅        1.0       1.32709
````

and we can calculate the partial trace as before

````julia
Ψ = eigvecs(Matrix(ham))[:, 1]
ρsub = partial_trace(Ψ, Hcons => Hsub)
````

````
4×4 Matrix{Float64}:
 0.0263519   0.0        0.0       0.0
 0.0         0.523418  -0.430494  0.0
 0.0        -0.430494   0.358676  0.0
 0.0         0.0        0.0       0.0915544
````

