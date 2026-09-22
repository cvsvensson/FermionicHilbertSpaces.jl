# Generate the README.md file from the readme.jl file using Literate.jl by: Literate.markdown("readme.jl", "."; name = "README", flavor = Literate.CommonMarkFlavor(), execute=true, credit = false) #src

# # FermionicHilbertSpaces.jl

# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
# [![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)


# This package provides tools for dealing with the symbolic and linear algebra often encountered in quantum mechanics. Fermions, bosons and spins are included, but most functionality is generic and easily extended. This includes:
# - **Symbolic operators:** Define operators algebraically and compute matrix representations on hilbert spaces.
# - **Hilbert-space mappings:** Tensor products, partial traces, reshaping with fermionic signs handled automatically.
# - **Constrained spaces:** Flexible system to constrain hilbert spaces with conserved quantities such as particle number or any custom constraint.

# The goal of this package is that you should be able to define your own physics and never have to think about indexing of vectors and matrices. 

# Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`.


import Random: seed!;#hide
seed!(1);#hide

# ## Quick example
# Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.
using FermionicHilbertSpaces, LinearAlgebra
@fermions f # Defines a symbolic fermion
sym_ham = sum(rand() * f[n]'f[n] for n in 1:4) +
          sum(f[n+1]'f[n] + hc for n in 1:3)

#Get a matrix representation of the hamiltonian on a hilbert space
H = hilbert_space(f, 1:4)
ham = representation(sym_ham, H)

#Diagonalize to find the ground state
Ψ = eigvecs(Matrix(ham))[:, 1]

#Define a subsystem and partial trace to find the reduced density matrix
Hsub = hilbert_space(f, 1:2)
ρsub = partial_trace(Ψ, H => Hsub)
entanglement_entropy = sum(λ -> -λ * log(λ), eigvals(ρsub))


# ### Conserved quantities
# The hamiltonian above conserves the number of fermions, which we can exploit as
Hcons = hilbert_space(f, 1:4, NumberConservation(2))
# This hilbert space contains only states with two fermions. We can use it just as before to get a matrix representation of the hamiltonian
ham = representation(sym_ham, Hcons)
# and we can calculate the partial trace as before
Ψ = eigvecs(Matrix(ham))[:, 1]
ρsub = partial_trace(Ψ, Hcons => Hsub)


# ## Who is this for?
# You may be interested if you
# - deal with fermions and want correct tensor products and partial traces. [More info.](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/fermions)
# - have complicated constraints on your Hilbert space. See examples [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/conservation).
# - want to define your own physics and never think about indexing. Here are some examples of extensions [Floquet example](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/literate_output/floquet_tutorial/), [clock?], [heom]

# You might want to look elsewhere if you
# - have large hilbert spaces that need methods like tensor networks.
# - deal with qubits.
# - deal with non-interacting systems (though there is some functionality for free fermions, see [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/non_interacting/) and [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/literate_output/free_fermions/)).

# This package is delibarately low-level and does not help you with common workflows like calculating eigenstates, doing time evolution, calculating expectation values, etc. But since this package uses standard Julia matrices you can easily use it together with packages that are specialized for such tasks, like Arpack.jl, KrylovKit.jl and DifferentialEquations.jl.
