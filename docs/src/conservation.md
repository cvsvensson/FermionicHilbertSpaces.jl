
# Conserved quantum numbers
The basis states of a Hilbert space can be organized into sectors with different quantum numbers. Only quantum numbers which are diagonal in the fock state basis are supported. 

Use `hilbert_space(labels, qn)`, where `qn` can be e.g.
- `ParityConservation()`: Puts all states with odd parity first, then all states with even parity.
- `ParityConservation(p::Int)`: Only contains states with parity `p` (-1 for odd and 1 for even).
- `number_conservation()`: Sorts basis states by the number of fermions.
- `number_conservation(sectors::Union{Int,Vector{Int}})`: Only contains states with the number(s) in the list `sectors`.
- `number_conservation(sectors, weight_function)`: 'weight_function' is a function that takes a label and returns an integer weight that represents the contribution of that fermion to the total number. The returned states will have total weighted number of fermions contained in `sectors`.
- `number_conservation(weight_function)`: As above, but allowing all sectors.
- Products of the above quantum numbers, which sorts states according to each factor in the product.

Using `number_conservation` with small sectors avoids the exponentially large Hilbert space.

## Spin
This package does not know anything about spin, but one can treat spin just as an extra label as follows:
```@example spin
using FermionicHilbertSpaces
labels = Base.product(1:4, (:↑,:↓))
H = hilbert_space(labels)
```
If spin is conserved, one can use 
```@example spin
H = hilbert_space(labels, number_conservation(label -> :↑ in label) * number_conservation(label -> :↓ in label))
```
to sort states according to the number of fermions with spin up and down. However, this package can't help to sort states into sectors with different total angular momentum, because that requires taking superpositions of different fock states.

To pick out the sector with 2 fermions with spin up and 0 fermions with spin down, one can extract it from the hilbert space defined above using `sector`, or construct it directly
```@example spin
FermionicHilbertSpaces.sector((2,0), H)
hilbert_space(labels, number_conservation(2, label -> :↑ in label) * number_conservation(0, label -> :↓ in label))
```

## No double occupation

The following example demonstrates how to restrict the Hilbert space to only allow single occupation of a site, which can be useful for simulating systems where the on-site coulomb interaction is strong enough to prevent double occupation.

```@example double_occupation
using FermionicHilbertSpaces, LinearAlgebra
N = 2 # number of fermions
space = 1:N 
spin = (:↑,:↓)
Hs = [hilbert_space([(k, s) for s in spin], number_conservation(0:1) for k in space]
H = tensor_product(Hs)
```

```@example double_occupation
dim(H) == 3^N
```

Can also take product of symmetries
```@example double_occupation
qn2 = prod(number_conservation(0:1, label -> k in label) for k in space)
H2 = hilbert_space(keys(H), qn2)
```

This gives the same states but with a different ordering.
```@example double_occupation
sort(basisstates(H2)) == sort(basisstates(H))
```

