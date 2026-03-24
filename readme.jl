# Generate the README.md file from the readme.jl file using Literate.jl by: Literate.markdown("readme.jl", "."; name = "README", flavor = Literate.CommonMarkFlavor(), execute=true, credit = false) #src

# # FermionicHilbertSpaces.jl

# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
# [![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)


# This package provides tools for working with fermionic hilbert spaces. This includes:
# - Fermionic tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
# - Operators on the hilbert spaces.

import Random: seed!;#hide
seed!(1);#hide

# ## Quick example
# Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.
using FermionicHilbertSpaces, LinearAlgebra
@fermions f # Defines a symbolic fermion
sym_ham = sum(rand() * f[n]'f[n] for n in 1:4) +
          sum(f[n+1]'f[n] + hc for n in 1:3)

#Get a matrix representation of the hamiltonian on a hilbert space
H = hilbert_space(1:4)
ham = matrix_representation(sym_ham, H)

#Diagonalize to find the ground state
Ψ = eigvecs(collect(ham))[:, 1]

#Define a subsystem and partial trace to find the reduced density matrix
Hsub = hilbert_space(1:2)
ρsub = partial_trace(Ψ * Ψ', H => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))


# ### Conserved quantities
# The hamiltonian above conserves the number of fermions, which we can exploit as
Hcons = hilbert_space(1:4, NumberConservation(2))
# This hilbert space contains only states with two fermions. We can use it just as before to get a matrix representation of the hamiltonian
ham = matrix_representation(sym_ham, Hcons)
# and we can calculate the partial trace as before
Ψ = eigvecs(collect(ham))[:, 1]
ρsub = partial_trace(Ψ * Ψ', Hcons => Hsub)