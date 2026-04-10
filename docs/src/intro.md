

## Algebra
This package includes operators algebras for fermions, bosons and spins. One can also define custom algebras. 

Let's look at fermions. We can define a fermionic field with
```@example intro
using FermionicHilbertSpaces, LinearAlgebra
@fermions f
```
This defines a fermionic field which can be index into to get fermionic operators.
```@example intro
f[1] # annihilation operator
f[1]' # creation operator
# indices can be anything and you can use several indices (which is equivalent to using a tuple)
f[1, :↑, π] == f[(1, :↑, π)]
```
These operators can be multiplied and added together, and they will be automatically normal ordered, taking into account the anticommutation relations. For example
```@example intro
f[1] * f[1]' 
```
Creation operator come first, sorted by their labels, and then annihilation operators, sorted by their labels. We have defined `hc` to act as the hermitian conjugate.
```@example intro
f[2]f[1]f[1]'f[2]' + hc
```

These work together with bosons and spins, so you can mix them together in the same expression
```@example intro
@boson b
@spin s 1//2
op = (f[0] + b + s[:x])^2
```
To see all the terms in the expression, you can access the .dict field
```@example intro
op.dict
```

## Matrix representations on hilbert spaces
In order to get a matrix representation, we must first associate the operators with a hilbert space. 
```@example intro
Hf = hilbert_space(f, 1:2) # a hilbert space associated with the fermionic field f and the labels 1 and 2. 
```
We can see the basis states of the hilbert space with
```@example intro
collect(basisstates(Hf))
```
We can get a matrix representation of an operator by
```@example intro
matrix_representation(f[1]'f[1] + f[2]'*f[2] + hc, Hf)
```
The same goes for bosons and spins
```@example intro
Hb = hilbert_space(b, 3)  # a three-dimensional hilbert space associated with the bosonic mode b
Hs = hilbert_space(s)     # a two-dimensional hilbert space associated with the spin s
matrix_representation(b'b, Hb) 
matrix_representation(s[:z], Hs)
```
```@example intro
H = tensor_product(Hf, Hb, Hs) # the tensor product of the three hilbert spaces
mat = matrix_representation(f[1] + b + s[:x], H);
```

### Partial traces and tensor products
This package was developed with the intention of making it easy to work with tensor products and partial traces of fermionic systems. The standard approach is to use the standard tensor product combined with appropriate Jordan-Wigner strings. Here, we follow the approach of Szalay.. and define a fermionic tensor product $\widetilde{\otimes}$ which ensures that fermionic operators from different spaces anti-commute. 

Let $H = H_1 \widetilde{\otimes} H_2$. Define the embedding of an operator $M_1$ in $H_1$ as $\mathscr{i}(M_1) = M_1 \widetilde{\otimes} \mathbb{1}_2$. 
```@example intro
    H1 = hilbert_space(f, 1:1)
    H2 = hilbert_space(f, 2:2)
    H = hilbert_space(f, 1:2)
    mat1 = matrix_representation(f[1], H1)
    mat = embed(mat1, H1 => H) 
```

The partial trace can be defined as the adjoint operation to the embedding. I.e. it satisfies 
```math
\mathrm{Tr}[A \, \mathrm{Tr}_{H\rightarrow H_1}[B]] = \mathrm{Tr}[\mathscr{i}(A)B]
```
for all operators $A$ on $H_1$ and $B$ on $H$. If the embedding has fermionic factors, so will the partial trace. Let's verify this identity
```@example intro
B = rand(dim(H), dim(H)) # some random operator on H
A = rand(dim(H1), dim(H1)) # some random operator on H1
Breduced = partial_trace(B, H => H1)
Aembedded = embed(A, H1 => H)
tr(A*Breduced), tr(Aembedded*B)
```

If we didn't explicitly define the subregion we wanted, we could use the function `subregion` to do so
```@example intro
H = tensor_product(Hf, Hb, Hs)
subregion([f[1], b], H)
```
