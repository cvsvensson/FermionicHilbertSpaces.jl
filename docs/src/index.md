```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces.jl

[FermionicHilbertSpaces.jl](https://github.com/cvsvensson/FermionicHilbertSpaces.jl) is a Julia package for defining fermionic Hilbert spaces and operators. The central features are fermionic tensor products and partial traces, which differ from the standard tensor product since fermions anticommute. 
[[1]](#fermion_information_article) 


# Introduction

The following example demonstrates how to define a fermionic Hilbert space and construct a simple Hamiltonian.
```@example intro
using FermionicHilbertSpaces, LinearAlgebra
N = 2 # number of fermions
spatial_labels = 1:N 
internal_labels = (:↑,:↓)
labels = Base.product(spatial_labels, internal_labels) 
H = hilbert_space(labels) 
```
We now have a Hilbert space representing 2N fermions. To construct operators on this space we first call `@fermions c` which makes `c` represent symbolic fermions. Indexing into `c` returns an operator representing an annihilation operator e.g. `c[1,:↑]`. The creation operator is given by the adjoint `c[1,:↑]'`. These can be multiplied and added together. Here is a simple Hamiltonian with hopping and Coulomb interaction.
```@example intro
@fermions c
hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc 
coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in spatial_labels)
ham = hopping + coulomb
```
To get the matrix representation of this operator on the Hilbert space, do
```@example intro
mat = matrix_representation(ham, H)
```

## Tensor product and partial trace
This package also includes functionality for combining Hilbert spaces and operators on them, and taking partial traces, in a way that is consistent with fermionic anticommutation relations. 
```@example intro
H1 = hilbert_space(1:2)
H2 = hilbert_space(3:4)
H = tensor_product(H1, H2)
c1 = matrix_representation(c[1], H1)
c3 = matrix_representation(c[3], H2)
c1c3 = matrix_representation(c[1] * c[3], H)
# Use embed to embed operators into a larger space
embed(c1, H1 => H) * embed(c3, H2 => H) ≈ c1c3 #true 
# Or call tensor_product to combine operators from different spaces
tensor_product((c1, c3), (H1, H2) => H) ≈ c1c3 
```
Let's partial trace to sites 1 and 3. Let's get a Hilbert space for those sites by using the function `subregion`, and then we can partial trace to that space with `partial_trace`.
```@example intro
Hsub = subregion([1,3], H)
size(Hsub, 1) / size(H, 1) * partial_trace(c1c3, H => Hsub) ≈ matrix_representation(c[1] * c[3], Hsub)
```

## Conserved quantum numbers
This package also includes some functionality for working with conserved quantum numbers. If we have for example number conservation, we might want to get a block structure of the hamiltonian. Here's how one can do that:
```@example intro
H = hilbert_space(labels, NumberConservation())
matrix_representation(ham, H)
```
This has a block structure corresponding to the different sectors. To only look at some sectors, for example the sectors with 0, 2 and 4 particles, use
```@example intro
H = hilbert_space(labels, NumberConservation([0, 2, 4]))
matrix_representation(ham, H)
```

Those sectors have even fermionic parity, which can alternatively be specified with `ParityConservation`.
```@example intro
H = hilbert_space(labels, ParityConservation(1))
matrix_representation(ham, H)
```

# References
```@raw html
<a name="fermion_information_article"></a>
```
[1] Szalay, Szilárd, et al. "Fermionic systems for quantum information people." [Journal of Physics A: Mathematical and Theoretical 54.39 (2021): 393001](https://doi.org/10.1088/1751-8121/ac0646), [arXiv:2006.03087](https://arxiv.org/abs/2006.03087)