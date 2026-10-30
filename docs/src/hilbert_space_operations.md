```@meta
CurrentModule = FermionicHilbertSpaces
```

# Hilbert space and array operations

This page shows how to build composite Hilbert spaces out of smaller ones, inspect their structure, and move arrays (vectors, matrices, density matrices) between a space and its subsystems. See [Fermionic tensor products, partial traces](fermions.md) for the conventions and phase factors behind `tensor_product`, `embed`, `partial_trace` and `partial_transpose`.

## [Pair arguments](@id pair_convention)

Wherever a function takes a `Pair` of Hilbert spaces, the pair reads
`input => output`: the left side is the space the argument lives on, and the right
side is the space of the result. The same order is used for every function:

```julia
partial_trace(m, H => Hsub)              # m on H, result on Hsub
embed(m, Hsub => H)                      # m on Hsub, result on H
partial_transpose(m, H => Hout, Hsub)    # m on H, result on Hout
generalized_kron(ms, Hs => H)            # ms on the spaces Hs, result on H
tensor_product(ms, Hs => H)              # ms on the spaces Hs, result on H
reshape(t, H => (H1, H2))                # t has one axis on H, result has axes on H1 and H2
```

Called without the array, the same pair returns a callable operation, for example
`partial_trace(H => Hsub)` or `partial_transpose(H => Hout, Hsub)`.

## Composing and querying Hilbert spaces

`tensor_product(spaces...; constraint=NoSymmetry())` builds the composite space spanned by `spaces`; an optional `constraint` restricts it to some basis states or groups them into sectors, see [Constrained Hilbert spaces and conserved quantum numbers](conservation.md). `subregion(Hs, H)` returns the part of `H` spanned by the spaces (or symbolic operators) in `Hs`; when `H` is constrained, the subregion only contains the compatible states, and the modes of the result are ordered as in `H`. The unexported `complementary_subsystem(H, Hsub)` returns the complementary subregion. It is used internally by `partial_trace` and `embed`, and precomputing it and passing it as `complement` can save time.

```@example hilbert_space_ops
using FermionicHilbertSpaces, LinearAlgebra
import FermionicHilbertSpaces: complementary_subsystem, atomic_factors
@fermions f
@boson b
H1 = hilbert_space(f, 1:2)
H2 = hilbert_space(f, 3:4)
tensor_product(H1, H2; constraint=NumberConservation(1));

H = hilbert_space(f, 1:4, NumberConservation(1))
Hsub = subregion([f[k] for k in 1:2], H)
basisstates(Hsub)  # the three states of modes 1:2 compatible with one particle in total
```

`factors(H)` splits a product space one level, so spaces that are grouped together stay intact (as of v0.10 only fermions group together), while `atomic_factors` splits all the way down to single modes. Atoms are ordered group by group, so `tensor_product(f1, b1, f2)` and `tensor_product(f1, f2, b1)` give the same space; only the order within a fermionic group matters.

```@example hilbert_space_ops
Hf = hilbert_space(f, 1:2, NumberConservation(1))
Hfb = tensor_product(Hf, hilbert_space(b, 5))
length.((factors(Hfb), factors(Hf), atomic_factors(Hfb)))
```

Symbols and spaces passed to `subregion` are looked up by name only, so `subregion([b[1]], H)` returns the bosonic mode as it appears in `H`, whatever its truncation. `partial_trace(m, H => Hsub)` and `embed` also look up atoms by name, so an `Hsub` built with a different truncation loses states: `partial_trace` throws an error if they carry nonzero weight (unless `skipmissing=true`), and `embed` gives zero on the states of `H` whose part on `Hsub` is lost. Use `subregion` to get an `Hsub` whose basis matches `H`.

A space built with a grouping constraint such as `NumberConservation()` is a `SectorHilbertSpace`. `quantumnumbers(H)` lists the quantum-number labels, `sectors(H)` the sectors as Hilbert spaces, `sector(qn, H)` one of them, and `indices(qn_or_Hsub, H)` the basis-state indices of a sector in the ordering of `H`. A space without sector structure behaves as a single sector: `quantumnumbers(H) == (nothing,)` and `indices(nothing, H) == 1:dim(H)`. A typical use is to find the ground state in each sector and `embed` it into the full space:

```@example hilbert_space_ops
H = hilbert_space(f, 1:4, NumberConservation())
symham = sum(f[n]'f[n] for n in 1:4)
ground_states = stack(sectors(H)) do Hsec
    embed(eigvecs(representation(symham, Hsec, :dense))[:, 1], Hsec => H)
end
ground_states' * representation(symham, H) * ground_states  # the full Hamiltonian projected onto the ground states
```

`representation(op, Axes(Hout, Hin))` returns the matrix of `op` as a map from `Hin` to `Hout`, with rows on `Hout` and columns on `Hin`, like `axes(m)`. A bare space `H` means the same space on every axis, and a ket or bra has the single axis `Axes(H)`. The two spaces must have the same groups, for example two particle-number sectors of the same modes. Unlike the [Pair arguments](@ref pair_convention), where `H => Hout` maps one array to another, `Axes` names the spaces of a single array, and it is public but not exported. Annihilating a fermion maps the two-particle states of four modes to the one-particle states:

```@example hilbert_space_ops
using FermionicHilbertSpaces: Axes
H2 = hilbert_space(f, 1:4, NumberConservation(2))
H1 = hilbert_space(f, 1:4, NumberConservation(1))
size(representation(f[1], Axes(H1, H2)))
```

The lowest-level accessors are `dim(H)`, `basisstates(H)` (in matrix-representation order), and `basisstate(ind, H)` / `state_index(state, H)`, which convert between a basis state and its integer index (`state_index` returns `0` if the state is not in `H`). See the [API reference](docstrings.md) for full docstrings.

## Operations on arrays

These functions respect fermionic anticommutation: embedding and tracing out include the Jordan-Wigner phase factors implied by the mode ordering, and `partial_transpose` uses the fermionic partial-time-reversal convention, unless disabled with `phase_factors=false`.

`embed(m, Hsub => H)` puts a matrix on `Hsub` into the larger space `H`, `partial_trace(m, H => Hsub)` reduces a matrix on `H` (or a state vector, which is first converted to a density matrix) to `Hsub`, and `expectation(op, state, H)` gives the expectation value of an operator in a normalized state vector or density matrix on `H`.

```@example hilbert_space_ops
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
m = representation(f[2]' * f[4], Hsub)
M = embed(m, Hsub => H)
partial_trace(M, H => Hsub) ≈ m
```

Two algorithms are available for the partial trace, `SubsystemPartialTraceAlg` and `FullPartialTraceAlg`; by default the cheaper one is picked based on the sizes of `Hsub`, `H` and the complement, and the `alg` keyword forces one. Both give the same result and throw in the same cases. When a state of `H` has no counterpart in `Hsub`, as for a truncated subsystem, the trace drops that state: by default `partial_trace` throws an `ArgumentError` if `m` has a nonzero element in its row or column, and `skipmissing=true` drops them silently. Combinations of a subsystem state and a complement state that are not in `H` drop nothing from `m`. Likewise `embed` returns the compression of `m ⊗ I` onto `H`, the adjoint of the partial trace, and by default throws if `m ⊗ I` maps a state of `H` outside `H` with nonzero amplitude, as `representation` does for an operator that does not preserve the constraint of `H`.

`partial_transpose(m, H, Hsub)` transposes the subsystem `Hsub` of an operator `m` on `H`; a pure state vector or vectorized density matrix is accepted too. For fermions this is the fermionic partial transpose (partial time reversal), for bosons and spins the ordinary one. It need not preserve particle-number or parity constraints: if a transposed state is missing from `Hout`, an `ArgumentError` is thrown, so pass a larger output space such as the unconstrained space on the same modes as `partial_transpose(m, H => Hout, Hsub)`, or use `skipmissing=true` to drop those terms. The unexported `logarithmic_negativity(rho, H, Hsub)` computes the logarithm of the trace norm of the partial transpose from its singular values, since the fermionic result is not generally Hermitian.

```@example hilbert_space_ops
H = hilbert_space(f, 1:2)
Hsub = hilbert_space(f, [1])
state = representation((f[1]' + f[2]') * Ket("0", H)) / √2
rho_pt = partial_transpose(state, H, Hsub)
negativity = (sum(svdvals(rho_pt)) - 1) / 2
negativity > 0  # the PPT criterion detects the entanglement
```

`tensor_product(ms, Hs => H)` computes the *ordered* product of the fermionic embeddings of the matrices or vectors in `ms`, each living on the corresponding space in `Hs`. The order of `ms` matters: `tensor_product((A, B), (HA, HB) => H)` equals `embed(A, HA=>H) * embed(B, HB=>H)`, so swapping `A` and `B` gives a minus sign when both are fermionic and odd. For vectors it builds a product state from its factors, which is only well defined for fermions if each state has definite fermion parity; with `ms[k] = ψₖ|0⟩` the result is `ψ₁ψ₂⋯|0⟩`. `generalized_kron(ms, Hs => H)` returns the same product state for vectors, but for matrices it computes the fermionic tensor product of [Fermionic tensor products](fermions.md), which does not depend on the order of `ms`. It equals `tensor_product` when each space in `Hs` covers consecutive modes of `H` and `Hs` is listed in that order; on other partitions it is not a product of embeddings, and `generalized_kron` of density matrices need not be a density matrix. Use `tensor_product` for product states, local observables, and anything that should equal a product of embedded operators.

```@example hilbert_space_ops
H1 = hilbert_space(f, 1:1)
H2 = hilbert_space(f, 2:3)
H = tensor_product(H1, H2)
f12 = tensor_product((representation(f[1], H1), representation(f[2], H2)), (H1, H2) => H)
v12 = tensor_product((representation("0", H1), representation("10", H2)), (H1, H2) => H);

H = hilbert_space(f, 1:4)
H13 = hilbert_space(f, [1, 3])
H24 = hilbert_space(f, [2, 4])
A = representation(f[1]' * f[3], H13)
B = representation(f[2]' * f[4], H24)
AB = representation(f[1]' * f[3] * f[2]' * f[4], H)
tensor_product((A, B), (H13, H24) => H) ≈ AB, generalized_kron((A, B), (H13, H24) => H) ≈ -AB
```

Called with only spaces, `partial_trace(H => Hsub)`, `embed(Hsub => H)`, `partial_transpose(H, Hsub)` and `tensor_product(Hs => H)` / `generalized_kron(Hs => H)` return callable operation objects. They are applied as `op(m)` or in place as `op(out, m)`, `sparse(op)` gives the superoperator as a sparse matrix, and they pay off when the same operation is applied many times. The maps `partial_trace(H => Hsub)` and `embed(Hsub => H)` skip the same states as the eager functions but do not check their input: they are the linear maps the eager functions compute when nothing is dropped.

`reshape(t, mappings...; repeat=false, skipmissing=false)` splits or combines the axes of `t` according to Hilbert-space mappings, applied left to right, each consuming the next consecutive group of input axes. The forms are `H => (H1, H2, ...)` (split one axis), `(H1, H2, ...) => H` (combine consecutive axes), `(H1, H2, ...) => (K1, K2, ...)` (repartition) and `H => H` (keep). With one mapping per axis group, `reshape(A, H1 => (H1a, H1b), (H2a, H2b) => H2, H3 => H3)` splits the first axis, combines the next two and keeps the last. A single mapping is expanded automatically: it is applied once if it consumes all axes of `t`, and to both halves if `t` has twice as many axes as it consumes, which suits square operator matrices (`reshape(m, H => (H1, H2))`); for asymmetric operators write `reshape(m, Hout => (H1, H2), Hin => (K1, K2))`, and `repeat=true` applies one mapping to every consecutive compatible axis group. Given only mappings, `reshape(H => (H1, H2))` returns a callable that precomputes data for repeated use. A component of the array on a basis state missing in the output space makes `reshape` throw an `ArgumentError`, unless `skipmissing=true`. The most common use is converting density matrices between vectorized and matrix form:

```@example hilbert_space_ops
@fermions fl
@fermions fr
Hl = hilbert_space(fl, 1:1)
Hr = hilbert_space(fr, 1:1)
Hlr = tensor_product(Hl, Hr)
vlr = representation("00", Hlr) + 2 * representation("10", Hlr)
rho = reshape(vlr, Hlr => (Hl, Hr))
Hlrconstrained = constrain_space(Hlr, NumberConservation([0, 1]))
reshape(rho, (Hl, Hr) => Hlrconstrained);
```

For worked examples, see [Open systems](literate_output/open_system_lindblad.md).
