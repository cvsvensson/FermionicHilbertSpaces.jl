```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) is a Julia package that provides tools for dealing with the symbolic and linear algebra often encountered in quantum mechanics. The goal of this package is that you should be able to define your own physics and never have to think about indexing of vectors and matrices. The package was originally developed to handle tensor products and partial traces in fermionic systems (which needs special care
[[1]](#fermion_information_article)), but now also includes bosons and spins and is easily extensible to other systems. 

Install the package from the Julia package manager with `Pkg.add("FermionicHilbertSpaces")`.

## Functionality

The package helps with three general areas
* **Symbolic operators:** Fermions, bosons and spins are included and can be used algebraically to build expressions. Their commutation relations are automatically applied to simplify the expression. 
* **Hilbert spaces:** Hilbert space objects keep track of basis states and their ordering, so you never need to think about indexing. Spaces can be combined into product spaces, split into subregions, and constrained with conservation laws.
* **Matrices:** The symbolic operators can be turned into matrix representations on the Hilbert spaces. The Hilbert spaces can then help with tensor products, partial traces and reshapings.

More concretely, with links to documentation and examples:
* Fermions: [Theory on tensor products and partial traces](fermions.md). Example with [Kitaev chain](literate_output/kitaev_chain.md).
* [Bosons tutorial](literate_output/bose_hubbard.md), [Spins tutorial](literate_output/spin_chain.md), [Mixed fermion-spin-boson systems](mixed_fermion_spin_boson.md)
* [Conserved quantities and constraints](conservation.md)
* Examples for extending with a custom algebra: [Floquet](literate_output/floquet_tutorial.md), [Clock operators](literate_output/clock_operators.md)
* Symbolic operators can act on [symbolic states](symbolic_states.md), so one can completely avoid instantiating large matrices.
* Symmetries such as translation symmetry and permutation symmetry can be handled with ...
* [Open systems](literate_output/open_system_lindblad.md) are represented by a tensor product of two copies of the hilbert space. With `reshape` you can convert between vectorized and matrix representations.

# Introduction

The following example demonstrates how to define a fermionic Hilbert space and construct a simple Hamiltonian.
```@example intro
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
labels = [(i, σ) for i in 1:2 for σ in (:↑,:↓)]
H = hilbert_space(c, labels) 
```
We now have a Hilbert space representing 2N fermions. To construct operators on this space we first call `@fermions c` which makes `c` represent symbolic fermions. Indexing into `c` returns an operator representing an annihilation operator e.g. `c[1,:↑]`. The creation operator is given by the adjoint `c[1,:↑]'`. These can be multiplied and added together and is automatically sorted into a unique normal order. Here is a simple Hamiltonian with hopping and Coulomb interaction.
```@example intro
hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc 
coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in 1:2)
ham = hopping + coulomb
```
To get the matrix representation of this operator on the Hilbert space, do
```@example intro
mat = representation(ham, H)
```

## Tensor product and partial trace
You can take tensor products of Hilbert spaces and operators on them, as well as calculate partial traces in a way that is consistent with fermionic anticommutation relations, see [Fermions](fermions.md). 
```@example intro
H1 = hilbert_space(c, 1:2)
H2 = hilbert_space(c, 3:4)
H = tensor_product(H1, H2)
c1 = representation(c[1], H1)
c3 = representation(c[3], H2)
c1c3 = representation(c[1] * c[3], H)
# Use embed to embed operators into a larger space
embed(c1, H1 => H) * embed(c3, H2 => H) ≈ c1c3 #true 
# Or call tensor_product to combine operators from different spaces
tensor_product((c1, c3), (H1, H2) => H) ≈ c1c3 
```
Let's partial trace to sites 1 and 3. Let's get a Hilbert space for those sites by using the function `subregion`, and then we can partial trace to that space with `partial_trace`.
```@example intro
Hsub = subregion([c[1], c[3]], H)
dim(Hsub) / dim(H) * partial_trace(c1c3, H => Hsub) ≈ representation(c[1] * c[3], Hsub)
```

## Conserved quantum numbers
Hilbert spaces can be constrained with custom constraints, see [Conserved quantites](conservation.md), which is useful when dealing with conservation laws. Particle number conservation is built into the package. Defining
```@example intro
H = hilbert_space(c, labels, NumberConservation())
```
gives a hilbert space which consists of 5 sectors with different numbers of particles. The matrix representation of the hamiltonian is block diagonal in this basis. We can get the representation of the hamiltonian in a specific sector, e.g. the 3-particle sector
```@example intro
representation(ham, sector(3, H))
```
We can restrict the sectors from the outset, which is useful if the full space is too large. For example, let's take the 0, 1 and 2 particle sectors of a 100-mode system
```@example intro
H = hilbert_space(c, 1:100, NumberConservation([0,1,2]))
representation(sum(c[k]'c[k] for k in 1:100), H)
```


Sectors and constraints are respected in tensor products and subregions.
```@example intro
H1 = hilbert_space(c, 1:10, NumberConservation([0,1]))
H2 = hilbert_space(c, 11:15, NumberConservation([3]))
H12 = tensor_product(H1, H2)
```
We see that the tensor product keeps track of the sectors of each factor. Let's take the subregion of modes 9-12. The function `subregion` finds the allowed states
```@example intro
Hsub = subregion([c[k] for k in 9:12], H12)
```
and we can sort it into sectors with definite particle numbers with
```julia
constrain_space(Hsub, NumberConservation())
```

# References
```@raw html
<a name="fermion_information_article"></a>
```
[1] Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)