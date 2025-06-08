# FermionicHilbertSpaces

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvsvensson.github.io/FermionicHilbertSpaces.jl/dev/)
[![Build Status](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cvsvensson/FermionicHilbertSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cvsvensson/FermionicHilbertSpaces.jl)


This package provides tools for working with fermionic hilbert spaces. This includes:
- Fermionic tensor products and partial traces mapping between different hilbert spaces, taking into account the fermionic properties.
- Operators on the hilbert spaces.

## Introduction
Let's analyze a small fermionic system. We first define a basis
```julia
using FermionicHilbertSpaces
N = 2 # number of fermions
spatial_labels = 1:N 
internal_labels = (:↑,:↓)
labels = Base.product(spatial_labels, internal_labels) 
H = hilbert_space(labels) 
#= 16⨯16 SimpleFockHilbertSpace:
modes: [(1, :↑), (2, :↑), (1, :↓), (2, :↓)]=#
c = fermions(H) #fermionic annihilation operators
```

Indexing into `c` returns sparse representations of the fermionic operators, so that one can write down Hamiltonians in a natural way:
```julia
H_hopping = c[1,:↑]'c[2,:↑] + c[1,:↓]'c[2,:↓] + hc 
H_coulomb = sum(c[n,:↑]'c[n,:↑]c[n,:↓]'c[n,:↓] for n in spatial_labels)
H = H_hopping + H_coulomb
#= 16×16 SparseArrays.SparseMatrixCSC{Int64, Int64} with 23 stored entries:
⎡⠠⠂⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠰⢂⠑⢄⠀⠀⎥
⎢⠀⠀⠑⢄⠠⢆⠀⠀⎥
⎣⠀⠀⠀⠀⠀⠀⠰⢆⎦ =#
```

## Tensor product and Partial trace
```julia
using FermionicHilbertSpaces, LinearAlgebra
H1 = hilbert_space([1, 2])
H2 = hilbert_space([3, 4])
H = tensor_product(H1, H2)

c1 = fermions(H1)
c2 = fermions(H2)
c = fermions(H)

c1c3 = tensor_product([c1[1], c2[3]], [H1, H2] => H)
c[1]*c[3] == c1c3 # true

partial_trace(tensor_product([c1[1], I/4],[H1,H2] => H), H => H1) == c1[1] # true
```