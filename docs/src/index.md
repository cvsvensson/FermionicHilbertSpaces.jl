```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) turns symbolic fermion, boson and spin operators into matrices on Hilbert spaces. The goal of this package is that you should be able to define your own physics and never have to think about indexing of vectors and matrices.

Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`.

The [cheat sheet](cheatsheet.md) lists common tasks and the functions that do them.


## Functionality

* **Symbolic operators:** fermions, bosons and spins are combined algebraically, e.g. `f[1]'f[2] + hc`, and simplified with their commutation relations. `representation` turns them into matrices on any Hilbert space.
* **Hilbert spaces:** spaces keep track of their basis states and ordering, so you never index by hand. They can be restricted by particle number, parity or any rule on the basis states.
* **Tensor products and partial traces:** spaces combine and split, as in `partial_trace(ρ, H => Hsub)`, with fermionic signs handled automatically.
* **Extensible:** custom operators and spaces can be added, see [Extending](extending.md).

It is not a solver: combine it with LinearAlgebra, KrylovKit or ODE solvers for eigenstates and time evolution.

## Getting started

We find the ground state of a half-filled chain of interacting spinless fermions and how entangled its two halves are. The Hamiltonian conserves particle number, so we diagonalize only in the half-filled sector and then trace out the right half:

```@example start
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
N, V = 6, 2.0
ham = sum(c[n+1]'c[n] + hc + V * c[n]'c[n] * c[n+1]'c[n+1] for n in 1:N-1)
H = hilbert_space(c, 1:N, NumberConservation(N ÷ 2))
E, vecs = eigen(Hermitian(representation(ham, H, Matrix)))
ρ = partial_trace(vecs[:, 1], H => hilbert_space(c, 1:N÷2))
S = -sum(λ * log(λ) for λ in eigvals(Hermitian(ρ)) if λ > 0)
E[1], S
```

## How to use this documentation

New here? Start with the [Fermions tutorial](literate_output/fermions_tutorial.md).

- **Cheat sheet**: *recall* common tasks and the functions that do them, on one page.
- **Tutorials**: *learn* the basics step by step, from fermions to mixed systems.
- **How-to guides**: *do* a specific task, such as restricting to a particle-number sector or taking a partial trace, independent of the physical model.
- **Examples**: *see* the package applied to complete physical models, worked from start to finish.
- **Explanation**: *understand* the concepts and design behind the package, such as fermionic tensor products and signs.
- **Performance**: *speed up* matrix construction, with parallelization and GPUs.
- **Extending**: *add* your own operators, basis states and spaces.
- **Reference**: *look up* the public API. Internals for contributors are listed separately.

The documentation structure is inspired by the [Diataxis framework](https://diataxis.fr/).

## Related packages
FermionicHilbertSpaces.jl grew out of [QuantumDots.jl](https://github.com/cvsvensson/QuantumDots.jl). Its symbolic operator algebra is built on [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl).
