```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces. The goal of this package is that you should be able to define your own physics and never have to think about indexing of vectors and matrices. The package was originally developed to handle tensor products and partial traces in fermionic systems (which needs special care
[^1]), but now also includes bosons and spins and is easily extensible to other systems. 

Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`.


## Functionality

The package provides
* **Algebra:** operators are written and simplified with ordinary algebra, e.g. `c[1]'c[2] + hc`.
* **Constrained spaces:** matrices can be built on any constrained space, not only the full Fock space.
* **Flexible constraints:** from particle number and parity to any rule on the basis states.
* **Simple interface:** tensor products, partial traces and embeddings take a pair of spaces, as in `partial_trace(ρ, H => Hsub)`.
* **Fermionic signs** are handled automatically.

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

More concretely, with links to documentation and examples:
* Fermions: [Theory on tensor products and partial traces](fermions.md). Example with [Kitaev chain](literate_output/kitaev_chain.md).
* [Bosons tutorial](literate_output/bosons.md), [Spins tutorial](literate_output/spin_chain.md), [Mixed fermion-spin-boson systems](mixed_fermion_spin_boson.md)
* [Conserved quantities and constraints](conservation.md)
* Examples for extending with a custom algebra: [Floquet](literate_output/floquet_tutorial.md), [Clock operators](literate_output/clock_operators.md)
* Symbolic operators can act on [symbolic states](symbolic_states.md), so one can completely avoid instantiating large matrices.
* Dealing with [spatial symmetries](literate_output/translation_symmetry.md) such as translations. 
* [Open systems](literate_output/open_system_lindblad.md) are represented by a tensor product of two copies of the hilbert space. With `reshape` you can convert between vectorized and matrix representations.

# Getting started

Hopping between two sites conserves particle number, so we diagonalize in the one-particle sector, the subspace with exactly one particle. We then `embed` the ground state in the full space and trace out the second site:

```@example start
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
H = hilbert_space(c, 1:2)
Hn = hilbert_space(c, 1:2, NumberConservation(1))
ham = c[1]'c[2] + hc + 0.5 * c[1]'c[1]
E = eigen(Hermitian(representation(ham, Hn, :dense)))
ψ = embed(E.vectors[:, 1], Hn => H)
partial_trace(ψ, H => hilbert_space(c, 1:1))
```

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).

[^1]: Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)