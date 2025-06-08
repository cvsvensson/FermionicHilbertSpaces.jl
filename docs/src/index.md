```@meta
CurrentModule = FermionicHilbertSpaces
```

# FermionicHilbertSpaces

Documentation for [FermionicHilbertSpaces](https://github.com/cvsvensson/FermionicHilbertSpaces.jl).

## Installation 
```julia
using Pkg; Pkg.add(url="https://github.com/cvsvensson/FermionicHilbertSpaces.jl")
```
or by adding a registry to your julia environment and then installing the package
```julia
using Pkg; Pkg.Registry.add(RegistrySpec(url = "https://github.com/williamesamuelson/PackageRegistry"))
Pkg.add("FermionicHilbertSpaces")
```

## Introduction

The following example demonstrates how to define a fermionic Hilbert space, create fermionic operators, and construct a simple Hamiltonian:

```@example intro
using FermionicHilbertSpaces
N = 2 # number of fermions
spatial_labels = 1:N 
internal_labels = (:↑,:↓)
labels = Base.product(spatial_labels, internal_labels) 
H = hilbert_space(labels) 
```

```@example intro
c = fermions(H) # fermionic annihilation operators
```

### Define a simple Hamiltonian from the fermionic operators

```@example intro
H_hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc 
H_coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in spatial_labels)
H_total
```

### Defining a symbolic hamiltonian

```@example intro
@fermions f 
matrix_representation(f[1,:↑]'*f[1,:↑], H)
```

### Tensor product and partial trace

```@example intro
H1 = hilbert_space(1:2)
H2 = hilbert_space(3:4)
H = tensor_product(H1, H2)
c1,c2,c = fermions(H1), fermions(H2), fermions(H)
c1c3 = tensor_product([c1[1], c2[3]], [H1, H2] => H)
c[1]*c[3] == c1c3
```

```@example intro
partial_trace(tensor_product([c1[1], I/4], [H1, H2] => H), H => H1) == c1[1] 
```

### Subspace
```@example intro
H1 == FermionicHilbertSpaces.subspace([1,2], H)
``` 

### Conserved quantum numbers
```@example intro
H = hilbert_space([1,2], ParityConservation())
```

# Functions
```@index
```
# Docstrings
```@autodocs
Modules = [FermionicHilbertSpaces]
```
