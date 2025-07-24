# FermionicHilbertSpaces.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
[![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)

This package provides tools for working with fermionic hilbert spaces. This includes:
- Fermionic tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
- Operators on the hilbert spaces.

## Quick example
Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.

````julia
using FermionicHilbertSpaces, LinearAlgebra
@fermions f # Defines a symbolic fermion
sym_ham = 0.5 * sum(f[n]'f[n] for n in 1:4) +
          sum(f[n+1]'f[n] + hc for n in 1:3) # Symbolic hamiltonian

#Get a matrix representation of the hamiltonian on a hilbert space
H = hilbert_space(1:4)
ham = matrix_representation(sym_ham, H)

#Diagonalize to find the ground state
Ψ = eigvecs(collect(ham))[:, 1]

#Define a subsystem and partial trace to find the reduced density matrix
Hsub = hilbert_space(1:2)
ρsub = partial_trace(Ψ * Ψ', H => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))
````

````
0.413278627769967
````

## Conserved quantities
The hamiltonian above conserves the number of fermion, which we can exploit.

````julia
Hcons = hilbert_space(1:4, FermionConservation(2))
````

````
6⨯6 SymmetricFockHilbertSpace:
modes: [1, 2, 3, 4]
FermionConservation([2])
````

Hcons contains only states with two fermions. We can use this hilbert space just as before

````julia
ham = matrix_representation(sym_ham, Hcons)
Ψ = eigvecs(collect(ham))[:, 1]
ρsub = partial_trace(Ψ * Ψ', Hcons => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))
````

````
0.41327862776996815
````

