```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces, so you never index vectors and matrices by hand.

It does:
* Symbolic operators with automatic simplification by their commutation relations.
* Hilbert spaces that track basis states and ordering: product spaces, subregions, and constraints such as particle number or parity.
* Sparse, dense and lazy matrix representations, tensor products, partial traces and embeddings with correct fermionic signs, and open-system superoperators.

It does not solve anything: there are no eigensolvers, time evolution or tensor-network methods. Combine it with LinearAlgebra, KrylovKit, ODE solvers and similar packages.

## Getting started

Install with `using Pkg; Pkg.add("FermionicHilbertSpaces")`. Two spinless fermion sites with hopping and an on-site energy:

```@example start
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
Hs = hilbert_space(c, 1:2)
H = c[1]'c[2] + hc + 0.5 * c[1]'c[1]
M = representation(H, Hs, :dense)
```
The basis states label the rows and columns of `M`:
```@example start
basisstates(Hs)
```
The Hamiltonian conserves particle number, so we can restrict to one particle. The sector has its own, smaller basis, and `embed` maps its eigenvectors back to the full space:
```@example start
Hn = hilbert_space(c, 1:2, NumberConservation(1))
E = eigen(Hermitian(representation(H, Hn, :dense)))
basisstates(Hn)
```
```@example start
ψ = embed(E.vectors[:, 1], Hn => Hs)
expectation(c[1]'c[1], ψ, Hs)
```

The Cheat sheet page lists the common operations. The tutorials work through whole models, starting with the Interacting Kitaev chain.

## Glossary

* **Sector:** the subspace of a constrained space with fixed quantum numbers, for example a fixed particle number.
* **Quantum number:** the value of a conserved quantity, such as particle number or parity, that labels a sector.
* **Constraint:** a rule that restricts a space to some basis states, for example [`NumberConservation`](@ref); see [Constrained spaces](conservation.md).
* **Subregion:** the part of a space that lives on a subset of the modes; see [`subregion`](@ref).
* **Factors / atomic factors:** the spaces a product space is built from; `factors` splits one level and `atomic_factors` down to single modes.
* **Symbolic operator vs representation:** an operator like `c[1]'c[2]` is an algebraic expression without a basis; `representation(op, H)` gives its matrix on the space `H`.
* **BdG:** the Bogoliubov-de Gennes space, where a quadratic Hamiltonian acts on particle and hole components; see [Non-interacting systems](non_interacting.md).
* **Parity (superselection):** physical fermionic states have definite fermion parity; [`ParityConservation`](@ref) restricts a space to one parity sector.

# Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).

# References
```@raw html
<a name="fermion_information_article"></a>
```
[1] Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)