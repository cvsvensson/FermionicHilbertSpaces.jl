```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces, so you can define your own physics and never index vectors and matrices by hand. It started as a tool for tensor products and partial traces of fermions, which need care with signs[^1], and now also covers bosons, spins and algebras you define yourself.

* Operators are written and simplified with ordinary algebra, e.g. `c[1]'c[2] + hc` ([Kitaev chain](literate_output/kitaev_chain.md)).
* Their matrices can be built on any constrained space, not only the full Fock space ([mixed fermion-spin-boson](mixed_fermion_spin_boson.md)).
* Constraints range from particle number and parity to any rule on the basis states ([constrained spaces](conservation.md)).
* Tensor products, partial traces and embeddings take a pair of spaces, as in `partial_trace(ρ, H => Hsub)` ([space operations](hilbert_space_operations.md)).
* Fermionic signs are handled automatically ([fermionic conventions](fermions.md)).

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

## Getting started

Install with `using Pkg; Pkg.add("FermionicHilbertSpaces")`. Hopping between two sites conserves particle number, so we diagonalize in the one-particle sector, the subspace with exactly one particle. We then `embed` the ground state in the full space and trace out the second site:

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

## Where to go

* **Tutorials** work through whole models, from the [interacting Kitaev chain](literate_output/kitaev_chain.md) via [bosons](literate_output/bosons.md), [spins](literate_output/spin_chain.md) and a [mixed fermion-spin-boson system](mixed_fermion_spin_boson.md) to [open systems](literate_output/open_system_lindblad.md) and [spatial symmetries](literate_output/translation_symmetry.md).
* **Guides** explain one tool each: [constraints and quantum numbers](conservation.md), [operations on spaces and arrays](hilbert_space_operations.md), [symbolic states](symbolic_states.md) that avoid building large matrices, the [fermionic tensor product and partial trace](fermions.md), and [non-interacting systems](non_interacting.md).
* **Performance** shows how to build matrices [in parallel](parallelization.md).
* **Extending** shows how to [add your own operators and spaces](extending.md), with [clock operators](literate_output/clock_operators.md) and [Floquet](literate_output/floquet_tutorial.md) as examples.
* **API reference** has [all docstrings](docstrings.md).

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).

[^1]: Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [J. Phys. A: Math. Theor. 54, 393001 (2021)](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087).
