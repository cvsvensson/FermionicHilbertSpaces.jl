```@meta
CurrentModule = FermionicHilbertSpaces
```

# Hilbert space and array operations

This page shows how to build composite Hilbert spaces, inspect their sectors, and move vectors and matrices between a space and its subsystems. The fermionic conventions behind these operations are explained in [Fermionic tensor products, partial traces](fermions.md).

## [Pair arguments](@id pair_convention)

Wherever a function takes a `Pair` of Hilbert spaces, it reads `input => output`: the left side is the space the argument lives on, the right side the space of the result.

```julia
partial_trace(m, H => Hsub)              # m on H, result on Hsub
embed(m, Hsub => H)                      # m on Hsub, result on H
partial_transpose(m, H => Hout, Hsub)    # m on H, result on Hout
tensor_product(ms, Hs => H)              # ms on the spaces Hs, result on H
reshape(t, H => (H1, H2))                # t has one axis on H, result has axes on H1 and H2
```

## Spaces

### Tensor products and subregions

`tensor_product(H1, H2; constraint)` builds the composite space, optionally restricted by a [constraint](conservation.md), and `factors(H)` returns the spaces it is built from. `subregion(Hs, H)` returns the part of `H` on the modes `Hs`; on a constrained `H` it contains only the compatible states:

```@example hilbert_space_ops
using FermionicHilbertSpaces, LinearAlgebra
@fermions f
H = tensor_product(hilbert_space(f, 1:2), hilbert_space(f, 3:4); constraint=NumberConservation(1))
Hsub = subregion([f[1], f[2]], H)
dim(Hsub)
```

Build subsystems with `subregion` rather than from scratch, so that their basis, such as the truncation of a boson, matches `H`.

### Sectors

A space built with a grouping constraint such as `NumberConservation()` is split into sectors, one per quantum number. `quantumnumbers(H)` lists the quantum numbers, `sectors(H)` the sectors as spaces, `sector(qn, H)` one of them, and `indices(Hsec, H)` the positions of a sector's basis states in `H`. Here we find the ground state in each sector and embed it in the full space:

```@example hilbert_space_ops
H = hilbert_space(f, 1:4, NumberConservation())
symham = sum(f[n]'f[n] for n in 1:4)
ground_states = stack(sectors(H)) do Hsec
    embed(eigvecs(representation(symham, Hsec, :dense))[:, 1], Hsec => H)
end
ground_states' * representation(symham, H) * ground_states
```

`dim(H)`, `basisstates(H)`, `basisstate(i, H)` and `state_index(state, H)` give the dimension, the basis states and the conversion between a basis state and its index.

### Maps between sectors

`representation(op, Axes(Hout, Hin))` gives the matrix of `op` from `Hin` to `Hout`, for an operator that changes a quantum number. Annihilating a fermion maps the two-particle states to the one-particle states:

```@example hilbert_space_ops
using FermionicHilbertSpaces: Axes
H2 = hilbert_space(f, 1:4, NumberConservation(2))
H1 = hilbert_space(f, 1:4, NumberConservation(1))
size(representation(f[1], Axes(H1, H2)))
```

## Arrays

The operations below include the fermionic phase factors implied by the mode order; `phase_factors=false` turns them off.

### Embedding and partial trace

`embed(m, Hsub => H)` puts a matrix on `Hsub` into the larger space `H`, and `partial_trace(m, H => Hsub)` reduces a density matrix or state vector on `H` to `Hsub`. The two are adjoint, so a local observable can be evaluated on either space:

```@example hilbert_space_ops
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
A = representation(f[2]' * f[4], Hsub)
ρ = rand(dim(H), dim(H))
tr(embed(A, Hsub => H) * ρ) ≈ tr(A * partial_trace(ρ, H => Hsub))
```

`expectation(op, state, H)` gives the expectation value of a symbolic operator in a state vector or density matrix on `H`.

### Partial transpose

`partial_transpose(m, H, Hsub)` transposes the subsystem `Hsub` of an operator or state on `H`. For fermions it is the fermionic partial transpose, whose singular values give the negativity:

```@example hilbert_space_ops
H = hilbert_space(f, 1:2)
state = representation((f[1]' + f[2]') * Ket("0", H)) / √2
rho_pt = partial_transpose(state, H, hilbert_space(f, [1]))
(sum(svdvals(rho_pt)) - 1) / 2
```

### [Tensor products of arrays](@id tensor_product_arrays)

`tensor_product(ms, Hs => H)` is the ordered product of the embeddings of `ms`: `tensor_product((A, B), (HA, HB) => H)` equals `embed(A, HA => H) * embed(B, HB => H)`, so swapping two odd fermionic operators flips the sign. Given vectors, it builds the product state. `generalized_kron` is the fermionic tensor product of [Fermionic tensor products](fermions.md), which does not depend on the order of `ms` and can differ from `tensor_product` when the subsystems interleave:

```@example hilbert_space_ops
H = hilbert_space(f, 1:4)
H13 = hilbert_space(f, [1, 3])
H24 = hilbert_space(f, [2, 4])
A = representation(f[1]' * f[3], H13)
B = representation(f[2]' * f[4], H24)
AB = representation(f[1]' * f[3] * f[2]' * f[4], H)
tensor_product((A, B), (H13, H24) => H) ≈ AB, generalized_kron((A, B), (H13, H24) => H) ≈ -AB
```

Use `tensor_product` for product states, local observables and anything that should equal a product of embedded operators.

### Reusable operations

Called with only spaces, `partial_trace(H => Hsub)`, `embed(Hsub => H)`, `partial_transpose(H, Hsub)` and `tensor_product(Hs => H)` return an operation that is applied as `op(m)`, or in place as `op(out, m)`. `sparse(op)` gives it as a sparse superoperator. This pays off when the same operation is applied many times.

### Reshaping

`reshape(t, H => (H1, H2))` splits an axis on `H` into axes on its subsystems, and `(H1, H2) => H` combines them. The main use is converting between vectorized and matrix forms of a density matrix, as in [Open systems](literate_output/open_system_lindblad.md):

```@example hilbert_space_ops
@fermions fl
@fermions fr
Hl = hilbert_space(fl, 1:1)
Hr = hilbert_space(fr, 1:1)
Hlr = tensor_product(Hl, Hr)
v = representation("00", Hlr) + 2 * representation("10", Hlr)
rho = reshape(v, Hlr => (Hl, Hr))
reshape(rho, (Hl, Hr) => Hlr) == v
```
