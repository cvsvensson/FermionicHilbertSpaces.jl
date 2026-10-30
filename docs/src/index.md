```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces, so you never index vectors and matrices by hand.

* Operators are written and simplified with ordinary algebra, e.g. `c[1]'c[2] + hc` ([Kitaev chain](literate_output/kitaev_chain.md)).
* Their matrices can be built on any constrained space, not only the full Fock space ([mixed fermion-spin-boson](mixed_fermion_spin_boson.md)).
* Constraints range from particle number and parity to any rule on the basis states ([constrained spaces](conservation.md)).
* Tensor products, partial traces and embeddings take a pair of spaces, as in `partial_trace(ρ, H => Hsub)` ([space operations](hilbert_space_operations.md)).
* Fermionic signs are handled automatically ([fermionic conventions](fermions.md)).

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

## Getting started

Install with `using Pkg; Pkg.add("FermionicHilbertSpaces")`. Hopping between two sites conserves particle number, so we diagonalize in the one-particle sector, `embed` the ground state in the full space and trace out the second site:

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

The [tutorials](literate_output/kitaev_chain.md) work through whole models, starting with the Interacting Kitaev chain.

## Glossary

* **Sector:** the subspace with fixed values of conserved quantities (quantum numbers), such as particle number or parity.
* **Subregion:** the part of a space that lives on a subset of the modes; see [`subregion`](@ref).
* **Factors:** the spaces a product space is built from; [`factors`](@ref) returns them.
* **Symbolic operator vs representation:** an operator like `c[1]'c[2]` is an algebraic expression without a basis; `representation(op, H)` gives its matrix on the space `H`.
* **BdG:** the Bogoliubov-de Gennes space, where a quadratic Hamiltonian acts on particle and hole components; see [Non-interacting systems](non_interacting.md).
* **Parity (superselection):** physical fermionic states have definite fermion parity; [`ParityConservation`](@ref) restricts a space to one parity sector.

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).
