```@meta
CurrentModule = FermionicHilbertSpaces
```

# Extending and performance hooks

This page lists the places where code outside the package can plug in new operators, states
and spaces. Each entry links to the full docstring.

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
  This is required; without it, products such as `X * X` throw a `MethodError`.
- `@commutative MyOp AbstractSym` declares that `MyOp` commutes with all other operators.
  This is required for products with the operators of the package.

Connection to Hilbert spaces:

- [`apply_local_operator`](@ref)`(op, state, space, precomp) -> (newstate, amplitude)` defines
  how `op` acts on a basis state. This is required for matrix representations.
- [`symbolic_group`](@ref)`(op)` returns the label of the space `op` acts on. This is required.
- `Base.adjoint(op)` returns the adjoint operator. This is required; transposed spaces, as
  used for open systems, apply `adjoint(op)`.
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
  space then find states with a table instead of a dictionary. The table relies on
  `state_index` returning the exact index in `1:dim(H)` for each basis state and `0` for every
  other state of the same type.
- [`AbstractSubspace`](@ref) is the supertype of spaces spanned by some basis states of a
  parent space. A subtype implements `parent`, `subspace_states`, `subspace_index` and
  `rewrap`, and gets tensor products, partial traces and matrix representations from its parent.
  [`default_subspace_index`](@ref) builds the index.
- [`default_subspace_index`](@ref)`(constraint, parent, states)` builds the lookup that
  `state_index` uses on constrained and sector spaces. A method that dispatches on the type of
  `constraint` only, not on `parent`, selects a custom index. It returns
  `default_subspace_index(NoSymmetry(), parent, states)` when `states` don't fit it.
- [`state_position`](@ref)`(index, state, parent)` is the one method a custom index
  implements: the position of `state` in the subspace, or `0`. It may use `parent` only through
  its basis states and `state_index`. Constructors check the index on the first, middle and
  last state and throw an `ArgumentError` on a mismatch.

To restrict a space of custom states, use [`SectorConstraint`](@ref) or
[`FilterConstraint`](@ref). Their required argument is the reducer, which is called on each
basis state. With `maps` and `spaces` together, it is called instead on the values that `maps`
gives for the parts of the state in `spaces`. See
[Constrained Hilbert spaces and conserved quantum numbers](conservation.md).
