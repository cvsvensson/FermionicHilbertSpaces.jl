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

A symbolic operator is an [`Op`](@ref)`(kind, mode)`: a kind, which says what the operator is, acting on a mode of
a field. A new algebra defines the kind. The tutorials
[Custom operators: Clock](literate_output/clock_operators.md) and
[Custom operators: Floquet](literate_output/floquet_tutorial.md) build one step by step.

These names are not exported, so import them; `import`, not `using`, also lets you add methods to the functions among them:
`import FermionicHilbertSpaces: @field, Op, apply_local_operator, local_mul_effect, mat_eltype` and
`import FermionicHilbertSpaces.NonCommutativeProducts: Swap, AddTerms`.

Required:

- A kind `MyKind`: a plain struct. It should be isbits, and one concrete type for the whole algebra, with a label
  such as `:X` or `:Z` in a field and not in a type parameter: products of operators of one type are several
  times faster, and GPU kernels need isbits kinds.
- `Base.adjoint(kind)` returns the kind of the adjoint operator.
- [`apply_local_operator`](@ref)`(kind, state, space, precomp) -> (newstate, amplitude)` defines how an operator
  of that kind acts on a basis state. The amplitude is zero for a state that the operator annihilates or maps
  out of a truncated basis. The method goes on the kind, not on `Op{MyKind}`.
- [`@field`](@ref) `c::MyKind` creates the field. `c[l]` is the mode labelled `l`, `c[l][k]` is
  `Op(MyKind(k), c[l])`, and
  `hilbert_space(c, labels, states)` has one space per mode, whose basis order is the order of `states`. A field names one kind type: an operator of
  another kind on the same mode is `Op(OtherKind(...), c[l])`, or the field is typed by a common abstract
  supertype whose constructor picks the kind, as in
  [Custom operators: Floquet](literate_output/floquet_tutorial.md).

Optional:

- [`local_mul_effect`](@ref)`(a, b)` gives the rule for the product `a * b` of operators on the same group:
  `nothing` keeps it, `Swap(c)` rewrites it as `c * b * a`, `AddTerms((Swap(c), r, ...))` adds further terms, and
  a number, operator or sum replaces it. A rule that returns `nothing`, a number or a `Swap` is defined
  on the kinds, `local_mul_effect(a::MyKind, b::MyKind)`. A rule that returns operators or needs the mode is
  defined on `Op{MyKind}`: there `a.kind` and `a.mode` are the parts of an operator, and
  `Op(MyKind(...), a.mode)` is another operator on the same mode. Without a rule, products stay as written, so
  operators that are equal in your algebra can compare unequal. Rules are applied before a basis is truncated,
  so a simplified product can differ from the product of the truncated matrices at the edge of the basis, as
  for bosons.
- Operators on different groups commute and are ordered by group, so no rule is needed for them.
- [`mat_eltype`](@ref)`(::Type{MyKind})` gives the element type of the matrix elements. The default is
  `Float64`, so operators with complex matrix elements need a method.
- `Base.show(io, op::Op{MyKind})` prints an operator; the default is `c[l][kind]`.
- [`mode_operator`](@ref)`(::Type{MyKind})` returns the kind of the single operator of a mode, which makes
  `c[l]` that operator.
- `apply_local_operator(kind, state, space, precomp, ::Type{T})` computes the amplitude in the float type `T`
  of the matrix. Without it, the amplitude of the method above is converted to `T`, which adds no precision and
  leaves any `Float64` arithmetic of that method in GPU kernels.

For speed:

- [`_precomputation_before_operator_application`](@ref)`(op::Op{MyKind}, space)` computes data once per
  operator factor, such as data that depends on the mode, which `apply_local_operator` then receives as
  `precomp` for every state.
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
  `default_subspace_index(NoConstraint(), parent, states)` when `states` don't fit it.
- [`state_position`](@ref)`(index, state, parent)` is the one method a custom index
  implements: the position of `state` in the subspace, or `0`. It may use `parent` only through
  its basis states and `state_index`. Constructors check the index on the first, middle and
  last state and throw an `ArgumentError` on a mismatch.

A custom operator that is diagonal in the basis, with integer or rational values, can be the
quantity of a [constraint](@ref "Writing a constraint"), like the Floquet number in
[Custom operators: Floquet](literate_output/floquet_tutorial.md).
Otherwise, a function `f` of the basis state can be the quantity, as in
`constrain_space(H, f => 2)`. On a product space the basis state is a [`ProductState`](@ref)
with parts `state.states`. To call `f` instead on the values that `maps` gives for the parts of
the state in `spaces`, use [`FermionicHilbertSpaces.SectorConstraint`](@ref) or
[`FermionicHilbertSpaces.FilterConstraint`](@ref).

Internally, `tensor_product`, `reshape` and `partial_trace` decompose and recombine basis states through a common mapper protocol. [`state_mapper`](@ref)`(H, Hs)` returns a mapper object, [`split_state`](@ref)`(state, mapper)` returns a pair `(outcomes, weights)`, where each outcome is a tuple with one substate per target subsystem, and [`combine_states`](@ref)`(substates, mapper)` returns a pair `(states, weights)`. The collections can have any concrete type; treat them as iterables.
