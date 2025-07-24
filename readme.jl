# Generate the README.md file from the readme.jl file using Literate.jl by: Literate.markdown("readme.jl", "."; name = "README", flavor = Literate.CommonMarkFlavor(), execute=true, credit = false) #src

# # FermionicHilbertSpaces.jl

# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
# [![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)


# This package provides tools for working with fermionic hilbert spaces. This includes:
# - Fermionic tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
# - Operators on the hilbert spaces.


# ## Example
# Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.
using FermionicHilbertSpaces
@fermions f
ε, t = 0.1, 1
sym_ham = ε * sum(f[n]'f[n] for n in 1:4) +
          t * sum(f[n+1]'f[n] + hc for n in 1:3)
# We now have a symbolic hamiltonian. To represent it as a matrix, let's define a hilbert space with four fermions
H = hilbert_space(1:4) 
# and then do
ham = matrix_representation(sym_ham, H)

# Use standard linear algebra to find the ground state
using LinearAlgebra
Ψ = eigvecs(collect(ham))[:, 1]
ρ = Ψ * Ψ';

# Define a subsystem and calculate the partial trace to find the reduced density matrix
Hsub = hilbert_space(1:2) 
ρsub = partial_trace(ρ, H => Hsub)
# Compute the entanglement entropy
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))


# ## Conserved quantities
# The hamiltonian above conserves the number of fermion, which we can exploit.
Hcons = hilbert_space(1:4, FermionConservation(2))
# Hcons contains only states with two fermions. We can use this hilbert space just as before
ham = matrix_representation(sym_ham, Hcons)
Ψ = eigvecs(collect(ham))[:, 1]
ρsub = partial_trace(Ψ * Ψ', Hcons => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))
