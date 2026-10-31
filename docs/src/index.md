```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces. The goal of this package is that you should be able to define your own physics and never have to think about indexing of vectors and matrices.

Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`.


## Functionality

* **Symbolic operators:** fermions, bosons and spins are combined algebraically, e.g. `f[1]'f[2] + hc`, and simplified with their commutation relations. `representation` turns them into matrices on any Hilbert space.
* **Hilbert spaces:** spaces keep track of their basis states and ordering, so you never index by hand. They can be restricted by particle number, parity or any rule on the basis states.
* **Tensor products and partial traces:** spaces combine and split, as in `partial_trace(ρ, H => Hsub)`, with fermionic signs handled automatically.
* **Extensible:** custom operators and spaces can be added, see [Extending](extending.md).

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

## Getting started

What is the ground-state energy of a half-filled chain of interacting spinless fermions, and how entangled are its two halves? The Hamiltonian conserves particle number, so we diagonalize in the half-filled sector only and trace out the right half:

```@example start
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
N, V = 6, 2.0
ham = sum(c[n+1]'c[n] + hc + V * c[n]'c[n] * c[n+1]'c[n+1] for n in 1:N-1)
H = hilbert_space(c, 1:N, NumberConservation(N ÷ 2))
E, vecs = eigen(Hermitian(representation(ham, H, :dense)))
ρ = partial_trace(vecs[:, 1], H => hilbert_space(c, 1:N÷2))
S = -sum(λ * log(λ) for λ in eigvals(Hermitian(ρ)) if λ > 0)
E[1], S
```

## Where to go next

Tutorials:
* [Interacting Kitaev chain](literate_output/kitaev_chain.md): parity sectors, ground states and the locality of many-body Majoranas.
* [Bosons](literate_output/bosons.md) and [Spins](literate_output/spin_chain.md): truncated bosonic modes and spin chains.
* [Mixed fermion-spin-boson system](mixed_fermion_spin_boson.md): combining different particle types in one space.
* [Open systems](literate_output/open_system_lindblad.md): Lindblad dynamics on the doubled space, with `reshape` between vectors and density matrices.
* [Spatial symmetries](literate_output/translation_symmetry.md): translation-symmetry sectors.

Guides:
* [Constrained spaces and quantum numbers](conservation.md): building sectors and custom constraints.
* [Hilbert space operations](hilbert_space_operations.md): tensor products, embeddings, partial traces and sector blocks.
* [Symbolic states](symbolic_states.md): acting with operators on states without building matrices.
* [Fermionic tensor products and partial traces](fermions.md): the conventions behind the fermionic signs[^1].
* [Non-interacting systems](non_interacting.md): free-fermion methods for quadratic Hamiltonians.
* [Extending](extending.md): adding your own operators, with [clock](literate_output/clock_operators.md) and [Floquet](literate_output/floquet_tutorial.md) examples.

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).

[^1]: Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)