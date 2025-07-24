# FermionicHilbertSpaces.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
[![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)

This package provides tools for working with fermionic hilbert spaces. This includes:
- Fermionic tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
- Operators on the hilbert spaces.

## Example
Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.

````julia
using FermionicHilbertSpaces
@fermions f
ε, t = 0.1, 1
sym_ham = ε * sum(f[n]'f[n] for n in 1:4) +
          t * sum(f[n+1]'f[n] + hc for n in 1:3)
````

````
Sum with 10 terms: 
0.1*f†[1]*f[1] + f†[1]*f[2] + f†[2]*f[1] + ...
````

We now have a symbolic hamiltonian. To represent it as a matrix, let's define a hilbert space with four fermions

````julia
H = hilbert_space(1:4)
````

````
16⨯16 SimpleFockHilbertSpace:
modes: [1, 2, 3, 4]
````

and then do

````julia
ham = matrix_representation(sym_ham, H)
````

````
16×16 SparseArrays.SparseMatrixCSC{Float64, Int64} with 39 stored entries:
⎡⠰⢆⢄⠀⠀⠀⠀⠀⎤
⎢⠀⠑⠱⢆⠑⢄⠀⠀⎥
⎢⠀⠀⠑⢄⠱⢆⢄⠀⎥
⎣⠀⠀⠀⠀⠀⠑⠱⢆⎦
````

Use standard linear algebra to find the ground state

````julia
using LinearAlgebra
Ψ = eigvecs(collect(ham))[:, 1]
ρ = Ψ * Ψ';
````

Define a subsystem and calculate the partial trace to find the reduced density matrix

````julia
Hsub = hilbert_space(1:2)
ρsub = partial_trace(ρ, H => Hsub)
````

````
4×4 Matrix{Float64}:
 0.05   0.0        0.0       0.0
 0.0    0.45      -0.447214  0.0
 0.0   -0.447214   0.45      0.0
 0.0    0.0        0.0       0.05
````

Compute the entanglement entropy

````julia
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))
````

````
0.41327862776996693
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
0.41327862776996765
````

