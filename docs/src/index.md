```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces, so you never index vectors and matrices by hand.

It does:
* Symbolic fermion, Majorana, boson and spin operators, simplified by their commutation relations, and custom operator algebras.
* Hilbert spaces that track basis states and ordering: product spaces, subregions, and constraints such as particle number or parity.
* Sparse, dense and lazy matrix representations; tensor products, partial traces, partial transposes and embeddings with correct fermionic signs; open-system superoperators.
* Symbolic states that operators act on without building matrices, and single-particle/BdG spaces for free fermions.

It is not a solver: it has no time evolution, iterative eigensolvers or tensor-network methods. Combine it with LinearAlgebra, KrylovKit, ODE solvers and similar packages.

## Getting started

Install with `using Pkg; Pkg.add("FermionicHilbertSpaces")`. Two spinless fermion sites with hopping and an on-site energy:

```@example start
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
H = hilbert_space(c, 1:2)
ham = c[1]'c[2] + hc + 0.5 * c[1]'c[1]
M = representation(ham, H, :dense)
```
The third argument picks the format: `:sparse` (the default), `:dense` or `:lazy`. The basis states label the rows and columns of `M`:
```@example start
basisstates(H)
```
The Hamiltonian conserves particle number, so we can restrict to one particle. The sector has its own, smaller basis, and `embed` maps its eigenvectors back to the full space:
```@example start
Hn = hilbert_space(c, 1:2, NumberConservation(1))
E = eigen(Hermitian(representation(ham, Hn, :dense)))
basisstates(Hn)
```
```@example start
ψ = embed(E.vectors[:, 1], Hn => H)
expectation(c[1]'c[1], ψ, H)
```
```@example start
partial_trace(ψ, H => hilbert_space(c, 1:1))
```

The [tutorials](literate_output/kitaev_chain.md) work through whole models, starting with the Interacting Kitaev chain. The guides cover [constraints and quantum numbers](conservation.md), [space operations](hilbert_space_operations.md) and [fermionic conventions](fermions.md).

## Glossary

* **Sector:** the subspace with fixed values of conserved quantities (quantum numbers), such as particle number or parity.
* **Constraint:** a rule that restricts a space to some basis states, for example [`NumberConservation`](@ref); see [Constrained spaces](conservation.md).
* **Subregion:** the part of a space that lives on a subset of the modes; see [`subregion`](@ref).
* **Factors:** the spaces a product space is built from; [`factors`](@ref) returns them.
* **Symbolic operator vs representation:** an operator like `c[1]'c[2]` is an algebraic expression without a basis; `representation(op, H)` gives its matrix on the space `H`.
* **BdG:** the Bogoliubov-de Gennes space, where a quadratic Hamiltonian acts on particle and hole components; see [Non-interacting systems](non_interacting.md).
* **Parity (superselection):** physical fermionic states have definite fermion parity; [`ParityConservation`](@ref) restricts a space to one parity sector.

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).
