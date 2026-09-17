```@meta
CurrentModule = FermionicHilbertSpaces
```

# Hilbert space and array operations

This page documents the operations used to build composite Hilbert spaces out of
smaller ones, to inspect their structure, and to move arrays (vectors, matrices,
density matrices) between a space and its subsystems. See [Fermionic tensor products
and partial traces](fermions.md) for the mathematical background and phase-factor
conventions behind `tensor_product`, `embed` and `partial_trace`; this page focuses
on syntax and usage.

## Composing and querying Hilbert spaces

### Building composite spaces with `tensor_product`

```julia
tensor_product(spaces...; constraint=NoSymmetry())
tensor_product(spaces; constraint=NoSymmetry())
```

Constructs the composite Hilbert space spanned by `spaces`. Spaces that share the
same underlying mode (e.g. the same fermionic mode) are merged rather than
duplicated, so `tensor_product` is safe to use even when the inputs overlap in
their atomic factors.

```@example ops_tensorproduct
using FermionicHilbertSpaces
@fermions f
H1 = hilbert_space(f, 1:2)
H2 = hilbert_space(f, 3:4)
H = tensor_product(H1, H2)
dim(H)
```

An optional `constraint` restricts the resulting space to a subset of basis
states, or groups it into sectors — see [Constrained Hilbert spaces and conserved
quantum numbers](conservation.md) for the available constraint types
(`NumberConservation`, `ParityConservation`, custom constraints, ...).

```@example ops_tensorproduct
Hn = tensor_product(H1, H2; constraint=NumberConservation(1))
dim(Hn)
```

!!! note
    The order in which spaces are combined matters for fermionic systems: it fixes
    the Jordan-Wigner mode ordering used by `embed`, `tensor_product` on arrays, and
    `partial_trace`. See [Fermionic tensor products and partial traces](fermions.md).

### Extracting subsystems with `subregion`

```julia
subregion(Hs, H::AbstractHilbertSpace)
```

Returns the subsystem of `H` spanned by the factor spaces (or symbolic operators)
in `Hs`. When `H` has a restricted set of basis states (e.g. from a constraint),
the returned subregion only contains the fock states that are compatible with
that restriction.

```@example ops_subregion
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(1))
Hsub = subregion(hilbert_space(f, 1:2), H)
basisstates(Hsub)
```

Here the total particle number is fixed to 1, so the subregion on modes `1:2` has
exactly three compatible states: `(1,0)`, `(0,1)`, and `(0,0)`.

### Decomposing a space into factors

`factors(H)` and `groups(H)` return the composite spaces that make up a product
space (for example the two `hilbert_space(f, 1:2)` and `hilbert_space(f, 3:4)`
factors of `H` above). For an atomic (non-product) space, both return `(H,)`.

### Sectors and quantum numbers

When a space is built with a constraint that groups states into sectors (such as
`NumberConservation`), the result is a `SectorHilbertSpace`. Its sector structure
can be inspected and navigated with:

```julia
quantumnumbers(H)      # the quantum-number label of each sector
sectors(H)              # the sectors themselves, as Hilbert spaces
sector(qn, H)           # the sector corresponding to a given quantum number
indices(qn_or_Hsub, H)  # basis-state indices of a sector, in H's ordering
```

```@example ops_sectors
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4, NumberConservation())
collect(quantumnumbers(H))
```

```@example ops_sectors
H1 = sector(1, H)
basisstates(H1)
```

```@example ops_sectors
indices(1, H) # same states, as indices into basisstates(H)
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
- `keys(H)` — the mode labels of `H`.

See the [Functions](docstrings.md) page for full docstrings.

## Operations on arrays

The functions below move vectors and matrices between a Hilbert space and its
subsystems. All of them respect fermionic anticommutation: embedding or tracing
out an operator picks up the Jordan-Wigner phase factors implied by the mode
ordering, unless disabled with `phase_factors=false`.

### `embed`: putting a subsystem operator into a larger space

```julia
embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)
```

Embeds a matrix `m` acting on `Hsub` into the larger space `H`.

```@example ops_embed
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
m = representation(f[2]' * f[4], Hsub)
M = embed(m, Hsub => H)
size(M)
```

### `partial_trace`: reducing to a subsystem

```julia
partial_trace(m, H => Hsub; complement=complementary_subsystem(H, Hsub), kwargs...)
```

Computes the partial trace of `m` (a matrix on `H`, or a state vector which is
first converted to a density matrix) down to `Hsub`. This is the adjoint
operation of `embed`.

```@example ops_embed
Msub = partial_trace(M, H => Hsub)
Msub ≈ m
```

Two algorithms are available, `SubsystemPartialTraceAlg` and
`FullPartialTraceAlg`; by default `partial_trace` picks whichever is cheaper based
on the sizes of `Hsub`, `H`, and the complementary subsystem, but you can force
one with the `alg` keyword. The `skipmissing` keyword controls what happens when
a combination of a substate and a complement state does not occur in `H` — this
can happen for constrained spaces, and defaults to the safer choice for each
algorithm.

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

```@example ops_tp_arrays
using FermionicHilbertSpaces
@fermions f
H1 = hilbert_space(f, 1:1)
H2 = hilbert_space(f, 2:3)
H = tensor_product(H1, H2)
v1 = representation(f[1], H1) * ones(dim(H1))
v2 = representation(f[2], H2) * ones(dim(H2))
v = tensor_product((v1, v2), (H1, H2) => H)
length(v)
```

`generalized_kron(ms, Hs, H=tensor_product(Hs); kwargs...)` computes the same kind
of combination, but using the mode ordering of `H` itself to decide the relative
order of the embeddings, rather than the order in which `ms` are given — this
matches plain `kron` when there are no fermionic phase factors. Both functions
also accept the curried form `tensor_product(Hs => H)` / `generalized_kron(Hs =>
H)`, returning a callable that can be applied to different collections of `ms`.

### Callable maps and sparse conversion

`partial_trace(H => Hsub; kwargs...)` and `embed(Hsub => H; kwargs...)`, called
with only a `Pair` of spaces, return callable operation objects (`PartialTraceMap`
and `EmbedMap`) instead of immediately acting on a matrix:

```@example ops_maps
using FermionicHilbertSpaces, SparseArrays
@fermions f
H = hilbert_space(f, 1:4)
Hsub = hilbert_space(f, [2, 4])
pt = partial_trace(H => Hsub)
emb = embed(Hsub => H)
pt' == emb # partial_trace and embed are mutually adjoint
```

These objects can be applied directly, `op(m)` or in-place `op(out, m)`, and
converted to an explicit sparse matrix acting on vectorized inputs with
`sparse(op)`. This is useful when the same embedding or partial trace is applied
many times, e.g. when building Liouvillian superoperators.

### `reshape`: splitting and combining array axes

```julia
reshape(t::AbstractArray, mappings...; repeat=false)
```

Reshapes the array `t` by splitting or combining its axes according to
Hilbert-space mappings. Mappings are applied left-to-right, and each mapping
consumes the next consecutive group of input axes.

Supported mapping forms:

- `H => (H1, H2, ...)`: split one axis into several
- `(H1, H2, ...) => H`: combine several consecutive axes into one
- `(H1, H2, ...) => (K1, K2, ...)`: repartition several axes into several
- `H => H`: keep one axis unchanged

Direct many-to-many mappings are equivalent to explicit combine-then-split
composition:

```julia
reshape(A, (H1, H2) => (K1, K2))
# equivalent to reshape(reshape(A, (H1, H2) => Hmid), Hmid => (K1, K2))
```

where `Hmid` is the canonical combined space of the input tuple.

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
