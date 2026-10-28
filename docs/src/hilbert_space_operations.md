```@meta
CurrentModule = FermionicHilbertSpaces
```

# Hilbert space and array operations

This page documents the operations used to build composite Hilbert spaces out of
smaller ones, to inspect their structure, and to move arrays (vectors, matrices,
density matrices) between a space and its subsystems. See [Fermionic tensor products,
partial traces](fermions.md) for the mathematical background
and phase-factor conventions behind `tensor_product`, `embed`, `partial_trace`, and
`partial_transpose`; this page focuses on syntax and usage.

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

### Building composite spaces with `tensor_product`

```julia
tensor_product(spaces...; constraint=NoSymmetry())
tensor_product(spaces; constraint=NoSymmetry())
```

Constructs the composite Hilbert space spanned by `spaces`.

```@example hilbert_space_ops
using FermionicHilbertSpaces
@fermions f
H1 = hilbert_space(f, 1:2)
H2 = hilbert_space(f, 3:4)
H = tensor_product(H1, H2)
```

An optional `constraint` restricts the resulting space to a subset of basis
states, or groups it into sectors — see [Constrained Hilbert spaces and conserved
quantum numbers](conservation.md) for the available constraint types.

```@example hilbert_space_ops
Hn = tensor_product(H1, H2; constraint=NumberConservation(1))
```

### Extracting subsystems with `subregion`

```julia
subregion(Hs, H::AbstractHilbertSpace)
```

Returns the subsystem of `H` spanned by the factor spaces (or symbolic operators)
in `Hs`. When `H` has a restricted set of basis states (e.g. from a constraint),
the returned subregion only contains the fock states that are compatible with
that restriction.

```@example hilbert_space_ops
H = hilbert_space(f, 1:4, NumberConservation(1))
Hsub = subregion(hilbert_space(f, 1:2), H)
basisstates(Hsub)
```

Here the total particle number is fixed to 1, so the subregion on modes `1:2` has
exactly three compatible states: `(1,0)`, `(0,1)`, and `(0,0)`.

In some cases `subregion` can also be called with symbolic operators instead of Hilbert spaces. Their order does not matter: the modes of the result are ordered as in `H`.
```@example hilbert_space_ops
Hsub = subregion([f[k] for k in 1:2], H)
```

In many cases one needs the complementary subsystem as well. It is available with `complementary_subsystem(H, Hsub)` (but unexported), which returns the subregion of `H` that is complementary to `Hsub`.
```@example hilbert_space_ops
FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
basisstates(Hsub)
```
The complement is used when performing partial trace or embed, and it can be a useful optimization to precompute it and pass it explicitly in some cases.

### Decomposing a space into factors

`factors(H)` returns the composite spaces that make up a product space 
```@example hilbert_space_ops
@boson b 
Hf = hilbert_space(f, 1:2, NumberConservation(1))
Hb = hilbert_space(b, 5)
Hfb = tensor_product(Hf, Hb)
factors(Hfb)
```
Note that the fermionic space was kept intact! However, calling factors in the fermionic space directly will return the individual mode spaces
```@example hilbert_space_ops
factors(Hf)
```

`factors` splits the space one level, so spaces grouped together will be preserved (as of v0.10, only Fermions group together). To split it all the way down to the individual mode spaces, use 
```@example hilbert_space_ops
FermionicHilbertSpaces.atomic_factors(Hfb)
```
The atoms are always ordered group by group, so `tensor_product(f1, b1, f2)` and `tensor_product(f1, f2, b1)` give the same space. Only the order within a fermionic group matters.

Internally, `FermionicHilbertSpaces.groups(H)` returns the spaces that line up with the slots of a basis state: for a product space these are the same as `factors(H)`, while any other space is a single group, `groups(H) == (H,)`.

When you pass symbols or spaces to `subregion`, they are looked up by name only, so `subregion([b[1]], H)` returns the bosonic mode as it appears in `H`, whatever its truncation. `partial_trace(m, H => Hsub)` and `embed` also look up atoms by name, so an `Hsub` built with a different truncation loses states: `partial_trace` throws an error if they carry nonzero weight, unless `skipmissing=true`, and `embed` gives zero on the states of `H` whose part on `Hsub` is lost. Use `subregion` to get an `Hsub` whose basis matches `H`.

### Sectors and quantum numbers

When a space is built with a constraint that groups states into sectors (such as
`NumberConservation`), the result is a `SectorHilbertSpace`. Its sector structure
can be inspected and navigated with:

```julia
quantumnumbers(H)       # the quantum-number label of each sector
sectors(H)              # the sectors themselves, as Hilbert spaces
sector(qn, H)           # the sector corresponding to a given quantum number
indices(qn_or_Hsub, H)  # basis-state indices of a sector, in H's ordering
```

```@example hilbert_space_ops
@fermions f
H = hilbert_space(f, 1:4, NumberConservation())
map((qn, Hsec) -> qn => basisstates(Hsec), quantumnumbers(H), sectors(H))
```

This pattern is useful to calculate ground states in each sector and extend each vector to the full space
```@example hilbert_space_ops
using LinearAlgebra, SparseArrays
symham = sum(f[n]'f[n] for n in 1:4)
ground_states = stack(sectors(H)) do Hsec
    ham = representation(symham, Hsec, :dense)
    v = zeros(dim(H))
    v[indices(Hsec, H)] = eigvecs(ham)[:, 1]
    v
end
# Let's project the full hamiltonian onto the ground states
ground_states' * representation(symham, H) * ground_states 
```

`indices` also accepts a sector Hilbert space instead of a quantum number, e.g.
`indices(H1, H)`. Spaces without sector structure behave as a single sector:
`quantumnumbers(H) == (nothing,)` and `indices(nothing, H) == 1:dim(H)`.

### Basic introspection

The lowest-level accessors work on any Hilbert space and are used throughout the
package to build the operations above:

- `dim(H)` — the Hilbert-space dimension (number of basis states).
- `basisstates(H)` — an iterable of the basis states, in matrix-representation order.
- `basisstate(ind, H)` / `state_index(state, H)` — convert between a basis state and
  its integer index (`state_index` returns `0` if the state is not in `H`).

See the [Functions](docstrings.md) page for full docstrings.

## Operations on arrays

The functions below move vectors and matrices between a Hilbert space and its
subsystems. They respect fermionic anticommutation: embedding and tracing out
operators include the Jordan-Wigner phase factors implied by the mode ordering,
and `partial_transpose` uses the fermionic partial-time-reversal convention,
unless disabled with `phase_factors=false`.

### `embed`: putting a subsystem operator into a larger space

```julia
embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)
```

Embeds a matrix `m` acting on `Hsub` into the larger space `H`.

```@example hilbert_space_ops
@fermions f
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
m = representation(f[2]' * f[4], Hsub)
M = embed(m, Hsub => H)
```

### `partial_trace`: reducing to a subsystem

```julia
partial_trace(m, H => Hsub; complement=complementary_subsystem(H, Hsub), kwargs...)
```

Computes the partial trace of `m` (a matrix on `H`, or a state vector which is
first converted to a density matrix) down to `Hsub`. This is the adjoint
operation of `embed`.

```@example hilbert_space_ops
Msub = partial_trace(M, H => Hsub)
```

Two algorithms are available, `SubsystemPartialTraceAlg` and
`FullPartialTraceAlg`; by default `partial_trace` picks whichever is cheaper based
on the sizes of `Hsub`, `H`, and the complementary subsystem, but you can force
one with the `alg` keyword. Both give the same result and throw in the same cases.

When a state of `H` has no counterpart in `Hsub`, as for a truncated subsystem, the trace
drops that state. By default `partial_trace` then throws an `ArgumentError` if `m` has a
nonzero element in its row or column; `skipmissing=true` drops them silently.
Combinations of a subsystem state and a complement state that are not in `H` drop nothing
from `m`.

`embed(m, Hsub => H)` returns the compression of `m ⊗ I` onto `H`, the adjoint of the partial
trace. By default it throws if `m ⊗ I` maps a state of `H` outside `H` with nonzero amplitude,
as `matrix_representation` does for an operator that does not preserve the constraint of `H`.

The maps `partial_trace(H => Hsub)` and `embed(Hsub => H)` skip the same states, but do not
check their input: they are the linear maps that the eager functions compute when nothing is dropped.

### `partial_transpose`: transposing a subsystem

```julia
partial_transpose(m, H, Hsub;
  complement=complementary_subsystem(H, Hsub), phase_factors=true, skipmissing=false)
partial_transpose(m, H => Hout, Hsub; kwargs...)  # result on Hout
partial_transpose(H => Hout, Hsub; kwargs...)   # callable map
```

Transposes the subsystem `Hsub` of an operator `m` on `H`. A pure state vector is
also accepted and treated as its density matrix, as is a vectorized density
matrix. For fermions, the default operation is the fermionic partial
transpose, also known as partial time reversal. For bosons and spins it is the ordinary partial transpose.

```@example hilbert_space_ops
using LinearAlgebra
@fermions f
H = hilbert_space(f, 1:2)
Hsub = hilbert_space(f, [1])
state = representation((f[1]' + f[2]') * Ket("0", H)) / √2
rho_pt = partial_transpose(state, H, Hsub)
negativity = (sum(svdvals(rho_pt)) - 1)/2 
negativity > 0 # PPT criteria for entanglement
```

The partial transpose need not preserve particle-number or parity constraints.
If a transposed state is missing from `Hout`, the default behavior is to throw an
`ArgumentError`. Pass a larger output space, such as the unconstrained space on the
same modes, as `H => Hout`; `skipmissing=true` instead drops terms whose output states
are absent. Set `phase_factors=false` to disable the fermionic phase factors and
compute the ordinary subsystem transpose.

`logarithmic_negativity(rho, H, Hsub; kwargs...)` computes the logarithm of the
trace norm of the partial transpose, using singular values (the fermionic result
is not generally Hermitian). This function is not exported.

### `tensor_product` and `generalized_kron` on arrays

```julia
tensor_product(ms, Hs, H::AbstractHilbertSpace; kwargs...)
tensor_product(ms, Hs => H; kwargs...)
```

Computes the *ordered* product of the fermionic embeddings of the matrices or
vectors in `ms`, each living on the corresponding space in `Hs`, into `H`. The
order of `ms` is significant: `tensor_product((A, B), (HA, HB) => H)` corresponds
to `embed(A, HA=>H) * embed(B, HB=>H)`, so swapping `A` and `B` picks up a minus
sign whenever both are fermionic and odd.

```@example hilbert_space_ops
@fermions f
H1 = hilbert_space(f, 1:1)
H2 = hilbert_space(f, 2:3)
H = tensor_product(H1, H2)
f1 = representation(f[1], H1) 
f2 = representation(f[2], H2)
f12 = tensor_product((f1, f2), (H1, H2) => H)
```
It can be used for vectors as well to build a product state from its factors,
but note that this is only well-defined for fermions if each state respects 
super-selection: it must not be a superposition of states with different fermion parity.
If each `ms[k] = ψₖ|0⟩` with `ψₖ` an operator on the modes of `Hs[k]`, the result is
`ψ₁ψ₂⋯|0⟩`. Swapping two odd states flips the sign.
```@example hilbert_space_ops
v1 = representation("0", H1)
v2 = representation("10", H2)
v12 = tensor_product((v1, v2), (H1, H2) => H)
```

`generalized_kron(ms, Hs, H=tensor_product(Hs); kwargs...)` returns the
same product state as `tensor_product` for vectors. For matrices it computes the fermionic tensor
product instead (see [Fermionic tensor products](fermions.md)), which does not depend
on the order of `ms`. It equals `tensor_product` when each space in `Hs` covers
consecutive modes of `H` and `Hs` is listed in that order. On other partitions it is
not a product of embeddings, and `generalized_kron` of density matrices need not be a
density matrix:

```@example hilbert_space_ops
H = hilbert_space(f, 1:4)
H13 = hilbert_space(f, [1, 3])
H24 = hilbert_space(f, [2, 4])
A = matrix_representation(f[1]' * f[3], H13)
B = matrix_representation(f[2]' * f[4], H24)
AB = matrix_representation(f[1]' * f[3] * f[2]' * f[4], H)
tensor_product((A, B), (H13, H24) => H) ≈ AB, generalized_kron((A, B), (H13, H24) => H) ≈ -AB
```

Use `tensor_product` for product states, local observables, and anything that
should equal a product of embedded operators. Both functions also accept the
curried form `tensor_product(Hs => H)` / `generalized_kron(Hs => H)`, returning a
callable that can be applied to different collections of `ms`.

### Callable maps and sparse conversion

`partial_trace(H => Hsub; kwargs...)`, `embed(Hsub => H; kwargs...)`, and
`partial_transpose(H, Hsub; kwargs...)` (or `partial_transpose(H => Hout, Hsub; kwargs...)`), called with only spaces, return callable
operation objects instead of immediately acting on an array:

```@example hilbert_space_ops
@fermions f
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
emb = embed(Hsub => H)
pt = partial_trace(H => Hsub)
ptt = partial_transpose(H, Hsub)
```

These objects can be applied directly, `op(m)` or in-place `op(out, m)`. Use
`sparse(op)` to obtain superoperator as a sparse matrix. Callable
maps are useful when the same operation is applied many times.

### `reshape`: splitting and combining array axes

```julia
reshape(t::AbstractArray, mappings...; repeat=false, skipmissing=false)
```

Reshapes the array `t` by splitting or combining its axes according to
Hilbert-space mappings. Mappings are applied left-to-right, and each mapping
consumes the next consecutive group of input axes.

Supported mapping forms:

- `H => (H1, H2, ...)`: split one axis into several
- `(H1, H2, ...) => H`: combine several consecutive axes into one
- `(H1, H2, ...) => (K1, K2, ...)`: repartition several axes into several
- `H => H`: keep one axis unchanged

The most common usage of this functionality is probably to convert density matrices
between vectorized and matrix forms
```@example hilbert_space_ops
@fermions fl
@fermions fr
Hl = hilbert_space(fl, 1:1)
Hr = hilbert_space(fr, 1:1)
Hlr = tensor_product(Hl, Hr)
vlr = representation("00", Hlr) + 2*representation("10", Hlr)
rho = reshape(vlr, Hlr => (Hl, Hr))
```

This works with constraints as well
```@example hilbert_space_ops
Hlrconstrained = constrain_space(Hlr, NumberConservation([0, 1]))
reshape(rho, (Hl, Hr) => Hlrconstrained)
```
If the array has a nonzero component on a basis state that is missing in the
output space, `reshape` throws an `ArgumentError`. Pass `skipmissing=true` to
drop such components instead.

#### Multiple mappings

Provide one mapping per axis group:

```julia
reshape(A, H1 => (H1a, H1b), (H2a, H2b) => H2, H3 => H3)
```

This splits the first axis, combines the next two, and keeps the last unchanged.

#### Single-mapping shorthand

If exactly one mapping is provided, it is expanded automatically:

- If it consumes all axes of `t`, it is applied once.
- If `t` has twice as many axes as the mapping consumes, the mapping is applied to
  the first half and again to the second half.

This is convenient for square operator matrices:

```julia
reshape(v, H => (H1, H2))
reshape(m, H => (H1, H2))
reshape(T, (H1, H2) => H)
```

For asymmetric operators, specify both sides explicitly:

```julia
reshape(m, Hout => (H1, H2), Hin => (K1, K2))
```

#### Repeating one mapping

With `repeat=true`, one mapping is applied to every consecutive compatible axis
group:

```julia
reshape(A, H => (H1, H2); repeat=true)
```

#### Pair syntax and curried form

Mappings use `Pair` syntax and can be passed directly:

```julia
reshape(m, H => (H1, H2))
reshape(T, (H1, H2) => H)
```

`reshape` also accepts mappings first and returns a callable object:

```julia
to_tensor = reshape(H => (H1, H2))
to_matrix = reshape((H1, H2) => H)
```

This returns a callable reshape map which precomputes some data so that it can be
applied to multiple arrays efficiently.

For worked examples, see [Open systems](literate_output/open_system_lindblad.md).

### The state mapper interface (internals)

Internally, `tensor_product`, `reshape` and `partial_trace` all decompose and
recombine basis states through a common mapper protocol:

- `state_mapper(H, Hs)` returns a mapper object.
- `split_state(state, mapper)` returns a tuple with one entry per target subsystem.
- Each tuple entry is a weighted collection `((substate, weight), ...)`.
- `combine_states(substates, mapper)` returns a weighted collection `((state,
  weight), ...)`.

This package does not require a single concrete container type for weighted
collections; callers should treat them as iterable collections of `(state,
weight)` outcomes. This protocol is only relevant if you are implementing a new
kind of Hilbert space.
