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
- [`local_mul_effect`](@ref)`(a::MyOp, b::MyOp)` gives the rule for the product `a * b` of
  operators on the same group: `nothing` keeps it, `Swap(c)` rewrites it as `c * b * a`,
  `AddTerms((Swap(c), r, ...))` adds further terms, and a number, operator or sum replaces it.
  Without a rule, products stay as written, so operators that are equal in your algebra can
  compare unequal. For two different types on the same group, rewrite one argument order in
  terms of the other.
- Operators on different groups commute and are ordered by group, so no rule is needed for them.
  Groups of different types are ordered by type name; numbers and strings are compared by value,
  and tuples element by element. Unequal groups whose types share a name need an `isless` that is
  a strict total order consistent with `==`; without one, `*` throws a `MethodError`.

Connection to Hilbert spaces:

- [`apply_local_operator`](@ref)`(op, state, space, precomp) -> (newstate, amplitude)` defines
  how `op` acts on a basis state. This is required for matrix representations.
- [`symbolic_group`](@ref)`(op)` returns the label of the space `op` acts on. This is required.
- `Base.adjoint(op)` returns the adjoint operator. This is required; transposed spaces, as
  used for open systems, apply `adjoint(op)`.
- `add_tag(op::MyOp, tag)` is needed for [`open_system`](@ref). It returns `op` rebuilt with
  its group label replaced by [`tag_label`](@ref)`(label, tag)`, so the field holding the label
  must accept that value. A custom space type other than `GenericHilbertSpace` needs the same method.
- [`mat_eltype`](@ref)`(::Type{<:MyOp})` gives the element type of the matrix elements. The
  default is `Float64`, so operators with complex matrix elements need a method.
- `apply_local_operator(op, state, space, precomp, ::Type{T})` is optional and computes the
  amplitude in the float type `T` of the matrix. Without it, the amplitude of the method above
  is converted to `T`, which adds no precision and leaves any `Float64` arithmetic of that method
  in GPU kernels.

For speed:

- [`_precomputation_before_operator_application`](@ref)`(op, space)` computes data once per
  operator factor, which `apply_local_operator` then receives as `precomp` for every state.
- [`apply_local_operators`](@ref)`(op::NCMul, state, space, precomps; transpose)` applies a
  whole product at once. A method for products of one operator type avoids applying the
  factors one by one.

## Custom basis states and spaces

`FermionicHilbertSpaces.groups(H)` returns the spaces that line up with the slots of a basis state: `factors(H)` for a product space, and `(H,)` for any other space. See [Hilbert space and array operations](hilbert_space_operations.md).

The simplest custom space is a [`GenericHilbertSpace`](@ref)`(label, states)`, which works with
any type of state. Its optional third argument `state_index` can be a function that computes
the index of a state, which avoids the default dictionary lookup.

- [`AbstractBasisState`](@ref) is the supertype of basis-state types. Subtyping it lets strings
  name states, and lets constrained spaces return index `0` for states of other types.
- [`interpret_state`](@ref)`(input, ::Type{MyState})` parses labels such as strings into states,
  for [`Kets`](@ref), `basisstate` and `state_index`.
- [`internal_rep`](@ref)`(state, space, Int)` and [`physical_rep`](@ref)`(n, ::Type{MyState})` convert
  states to integers and back. When all groups of a product space have them, operators on the
  product space act on integers.
- [`has_arithmetic_state_index`](@ref)`(H)` returns `true` for a space type whose
  [`state_index`](@ref) is computed from the state alone. Constrained spaces over such a
  space then find states with a table instead of a dictionary. The table relies on
  `state_index` returning the exact index in `1:dim(H)` for each basis state and `0` for every
  other state of the same type.
- [`AbstractSubspace`](@ref) is the supertype of spaces spanned by some basis states of a
  parent space. A subtype implements `parent`, [`subspace_states`](@ref), [`subspace_index`](@ref) and
  [`rewrap`](@ref), and gets tensor products, partial traces and matrix representations from its parent.
  [`default_subspace_index`](@ref) builds the index.
- [`default_subspace_index`](@ref)`(constraint, parent, states)` builds the lookup that
  `state_index` uses on constrained and sector spaces. A method that dispatches on the type of
  `constraint` only, not on `parent`, selects a custom index. It returns
  `default_subspace_index(NoSymmetry(), parent, states)` when `states` don't fit it.
- [`state_position`](@ref)`(index, state, parent)` is the one method a custom index
  implements: the position of `state` in the subspace, or `0`. It may use `parent` only through
  its basis states and `state_index`. Constructors check the index on the first, middle and
  last state and throw an `ArgumentError` on a mismatch.

A custom operator that is diagonal in the basis, with integer or rational values, can be the
quantity of a [constraint](@ref "Writing a constraint"), like the Floquet number in
[Custom operators: Floquet](literate_output/floquet_tutorial.md).
Otherwise, to restrict a space of custom states, use [`SectorConstraint`](@ref) or
[`FilterConstraint`](@ref). Their required argument is the reducer, which is called on each
basis state. With `maps` and `spaces` together, it is called instead on the values that `maps`
gives for the parts of the state in `spaces`. See
[Constrained Hilbert spaces and conserved quantum numbers](conservation.md).

Internally, `tensor_product`, `reshape` and `partial_trace` decompose and recombine basis states through a common mapper protocol. [`state_mapper`](@ref)`(H, Hs)` returns a mapper object, [`split_state`](@ref)`(state, mapper)` returns a pair `(outcomes, weights)`, where each outcome is a tuple with one substate per target subsystem, and [`combine_states`](@ref)`(substates, mapper)` returns a pair `(states, weights)`. The collections can have any concrete type; treat them as iterables.
