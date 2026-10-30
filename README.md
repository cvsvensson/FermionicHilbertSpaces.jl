# FermionicHilbertSpaces.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
[![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

FermionicHilbertSpaces.jl turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces, so you never index vectors and matrices by hand.
- **Operator algebra:** operators are written and simplified with ordinary algebra, e.g. `c[1]'c[2] + hc`.
- **Constrained spaces:** matrices can be built on spaces restricted by particle number, parity or any rule on the basis states ([constrained spaces](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/conservation/)).
- **Subsystems with fermionic signs:** tensor products, partial traces and embeddings take a pair of spaces, as in `partial_trace(ρ, H => Hsub)`, and handle fermionic signs automatically ([fermionic conventions](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/fermions/)).
- **Extensible:** custom operator algebras and spaces can be added ([extending](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/extending/)).

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`. See the [documentation](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/) for getting started, tutorials, guides and the API reference.


## Quick example
Let's define a small fermionic system, find the ground state and compute the entanglement entropy of half the system.

````julia
using FermionicHilbertSpaces, LinearAlgebra
@fermions f # Defines a symbolic fermion
sym_ham = sum(rand() * f[n]'f[n] for n in 1:4) +
    sum(f[n+1]'f[n] + hc for n in 1:3)

#Hilbert space of the four modes
H = hilbert_space(f, 1:4)

#Diagonalize to find the ground state
Ψ = eigvecs(Hermitian(representation(sym_ham, H, :dense)))[:, 1]

#Define a subsystem and partial trace to find the reduced density matrix
Hsub = hilbert_space(f, 1:2)
ρsub = partial_trace(Ψ, H => Hsub)
entanglement_entropy = sum(-λ * log(λ) for λ in eigvals(ρsub))
````

````
0.44395495955661934
````

This hamiltonian conserves the number of fermions, which we can exploit. Let's specialize to the 3-particle sector and compute the entanglement entropy of the ground state in that sector.

````julia
Hcons = hilbert_space(f, 1:4, NumberConservation(3))
Hsubcons = subregion(Hsub, Hcons) # Include only substates consistent with Hcons
Ψcons = eigvecs(representation(sym_ham, Hcons, :dense))[:, 1]
ρsub = partial_trace(Ψcons, Hcons => Hsubcons)
sum(-λ * log(λ) for λ in eigvals(ρsub))
````

````
0.6390308969922927
````

## Who is this for?
You may be interested if you
- deal with fermions and want correct tensor products and partial traces. [More info.](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/fermions/)
- have complicated constraints on your Hilbert space. See examples [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/conservation/).
- want to define your own physics and never think about indexing. Examples:  [Clock operators](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/literate_output/clock_operators/), [Floquet](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/literate_output/floquet_tutorial/)

You might want to look elsewhere if you
- have large hilbert spaces that need methods like tensor networks.
- deal with qubits.
- deal with non-interacting systems (though there is some functionality for free fermions, see [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/non_interacting/) and [here](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/literate_output/free_fermions/)).

The documentation has examples of combining this package with other Julia packages for eigenstates and time evolution.

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).
