```@meta
CurrentModule = FermionicHilbertSpaces
```

# Extending and performance hooks

This page lists the places where code outside the package can plug in new operators, states
and spaces, and the options that select a faster way to compute something. Each entry links
to the full docstring.

Most of the methods below are not exported. Add methods to them with
`import FermionicHilbertSpaces: name` or by qualifying the name. They are less stable than
the exported API and may change in a minor release.

## Custom operators

A new kind of symbolic operator is a subtype of [`AbstractSym`](@ref). The tutorials
[Custom operators: Clock](literate_output/clock_operators.md) and
[Custom operators: Floquet](literate_output/floquet_tutorial.md) build one step by step.

Algebra, from [NonCommutativeProducts.jl](https://github.com/cvsvensson/NonCommutativeProducts.jl):

- `@nc MyOp` makes `+` and `*` build symbolic sums and products of `MyOp`.
- `mul_effect(a::MyOp, b::MyOp)` gives the rule for the product `a * b`: `nothing` keeps it,
  `Swap(c)` rewrites it as `c * b * a`, and `AddTerms((Swap(c), r, ...))` adds further terms.
- `@commutative MyOp AbstractSym` declares that `MyOp` commutes with all other operators.

Connection to Hilbert spaces:

- [`apply_local_operator`](@ref)`(op, state, space, precomp) -> (newstate, amplitude)` defines
  how `op` acts on a basis state. This is required for matrix representations.
- [`symbolic_group`](@ref)`(op)` returns the label of the space `op` acts on. This is required.
- [`mat_eltype`](@ref)`(::Type{<:MyOp})` gives the element type of the matrix elements. The
  default is `Float64`, so operators with complex matrix elements need a method.

For speed:

- [`_precomputation_before_operator_application`](@ref)`(op, space)` computes data once per
  operator factor, which `apply_local_operator` then receives as `precomp` for every state.
- [`apply_local_operators`](@ref)`(op::NCMul, state, space, precomps; transpose)` applies a
  whole product at once. A method for products of one operator type avoids applying the
  factors one by one.

## Custom basis states and spaces

The simplest custom space is a [`GenericHilbertSpace`](@ref)`(label, states)`, which works with
any type of state. Its optional third argument `state_index` can be a function that computes
the index of a state, which avoids the default dictionary lookup.

- [`AbstractBasisState`](@ref) is the supertype of basis-state types. Subtyping it lets strings
  name states, and lets constrained spaces return index `0` for states of other types.
- [`interpret_state`](@ref)`(input, ::Type{MyState})` parses labels such as strings into states,
  for [`Kets`](@ref), `basisstate` and `state_index`.
- [`internal_rep`](@ref)`(state, space, Int)` and `physical_rep(n, ::Type{MyState})` convert
  states to integers and back. When all groups of a product space have them, operators on the
  product space act on integers, which is faster.
- [`has_arithmetic_state_index`](@ref)`(H)` returns `true` for a space type whose
  [`state_index`](@ref) is computed from the state alone. Constrained spaces over such a
  space then find states with a table instead of a dictionary.
- [`AbstractSubspace`](@ref) is the supertype of spaces spanned by some basis states of a
  parent space. A subtype implements `parent`, `subspace_states`, `subspace_index` and
  `rewrap`, and gets tensor products, partial traces and matrix representations from its parent.
  [`SubspaceIndex`](@ref) builds the index.

To restrict a space of custom states, use [`SectorConstraint`](@ref) or
[`FilterConstraint`](@ref) with a `maps` function from states to values; see
[Constrained Hilbert spaces and conserved quantum numbers](conservation.md).

## Matrix representations

The third argument of [`representation`](@ref) selects the format:

- `:sparse` or [`EagerSparseRepr`](@ref)`{T}()`: a sparse matrix, the default.
- `:dense` or [`EagerDenseRepr`](@ref)`{T}()`: a dense matrix.
- `:lazy` or [`LazyRepr`](@ref)`(input)`: a matrix-free operator, a [`LazyOperator`](@ref) wrapped
  as a SciMLOperators operator. It needs no storage for the matrix. `input` selects dense or
  sparse vectors.
- [`PartitionedSparseRepr`](@ref): a distributed sparse matrix, with PartitionedArrays.jl.
- A type such as `ComplexF64` or `Matrix{Float32}` fixes the element type.

The `chunking` keyword parallelizes the construction, and the products of lazy operators, with
OhMyThreads.jl: [`NoChunking`](@ref) (the default), [`TermChunking`](@ref)`(scheduler)` and
[`StateChunking`](@ref)`(scheduler)`. See [Parallelization](@ref) for how to choose a scheduler.

Before many products of a lazy operator `L` with vectors like `v`, call
`SciMLOperators.cache_operator(L, v)` once, so that the work buffers are allocated once
rather than on every product.

## Partial traces and embeddings

- The `alg` keyword of [`partial_trace`](@ref) selects [`SubsystemPartialTraceAlg`](@ref) or
  [`FullPartialTraceAlg`](@ref). By default the cheaper one is estimated from the dimensions
  of the spaces and, for a sparse input, its number of stored elements.
- `partial_trace(H => Hsub)` and `embed(Hsub => H)` build a map once, which is faster when the
  same partial trace or embedding is applied to many inputs.
