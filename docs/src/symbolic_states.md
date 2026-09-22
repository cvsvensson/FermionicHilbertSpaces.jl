```@meta
CurrentModule = FermionicHilbertSpaces
```

# Symbolic bras and kets

Besides symbolic operators (`c[1]`, `c[1]'`, ...), this package lets you write basis states symbolically as bras and kets, multiply them with operators and with each other, and turn the resulting expressions into vectors or matrices.

## Creating symbolic states

`Kets(H)` returns a callable object that parses a basis-state label into a symbolic ket for the Hilbert space `H`.

```@example states
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:2)
vf = Kets(H)
vf("10")
```

The string is a Fock bitstring: one `'0'`/`'1'` character per mode, in the same order as the labels used to build `H`. A bra is obtained with `'`:

```@example states
vf("10")'
```

Other space types parse their labels differently. Bosons use an occupation-number string, e.g. `vb("3")` for a boson space `vb = Kets(hilbert_space(b, 1))`. For a `ProductSpace`, pass one label per factor, either as separate arguments or as a tuple: `vfb("0", "2")` for a fermion factor and a boson factor combined with `tensor_product`.

## Algebra

Multiplying a bra and a ket gives their overlap (`0` if the states differ):

```@example states
vf("10")' * vf("10")
```

```@example states
vf("10")' * vf("11")
```

Multiplying a ket and a bra instead gives an operator (an outer product, or "ketbra"):

```@example states
vf("10") * vf("01")'
```

Symbolic operators act on kets from the left and on bras from the right, picking up the same fermionic signs as usual:

```@example states
f[2]' * vf("10")
```

```@example states
vf("10")' * f[2]
```

`ket * ket` and `bra * bra` within the same space are not meaningful and throw an `ArgumentError`.

## Numeric vectors and matrices

`representation` (or the more specific `vector_representation`/`representation`) turns a symbolic ket, bra or ketbra into a concrete vector or matrix on `H`:

```@example states
representation(vf("10"), H)
```

```@example states
representation(vf("10") * vf("01")', H)
```

This also works for sums of kets/bras/ketbras with coefficients, and for the other representation types (`:sparse`, `:dense`, `:lazy`).

## Product spaces and mixed operator/state expressions

States belonging to different factor spaces of a `tensor_product` combine into a joint state, matching the state you would get from `Kets` on the combined space directly:

```@example states
@boson b
Hb = hilbert_space(b, 3)
Hfb = tensor_product(H, Hb)
vb = Kets(Hb)
vfb = Kets(Hfb)
vf("10") * vb("2") == vfb("10", "2")
```

Operators and states from different factors can be freely interleaved in a product; they are automatically reordered (with correct signs) into operators acting first, then states combined per factor:

```@example states
f[1]' * vf("10") * b' * vb("2") == f[1]' * b' * vfb("10", "2")
```

## Simple example

Put together, here is a two-site fermion example that builds a ket, computes an overlap, and constructs a projector onto that state:

```@example simple
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:2)
vf = Kets(H)

ket = vf("10")
overlap = ket' * ket        # 1, since the state is normalized

projector = ket * ket'      # a rank-1 projector operator
representation(projector, H)
```
