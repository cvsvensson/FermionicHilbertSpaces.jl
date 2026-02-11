
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

## Small subspaces in large systems
For N fermions, the full hilbert space is exponentially large in N. However, due to conservation laws, we may be interested in only a small subspace. If you use a quantum number which consists of only products of number conservations, this package attempts to find the subspaces without enumerating the full hilbert space.

As an example, consider the Hubbard model on N sites 
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + U \sum_i n_{i,↑} n_{i,↓}.
```
which conserves the number of spin up and spin down fermions separately. Let's define a function to get the hamiltonian with symbolic fermions 
```@example hubbard
using FermionicHilbertSpaces
function hubbard_hamiltonian(N, t, U)
    @fermions c
    spins = (:↑,:↓)
    sum(-t * c[i,σ]' * c[i+1,σ] + hc for σ in spins for i in 1:N-1) + sum(U * c[i,:↑]'c[i,:↑] * c[i,:↓]'c[i,:↓] for i in 1:N)
end
```
Let's find the matrix representation of the hamiltonian in the sector with `N_up` spin up fermions and `N_down` spin down fermions. To find this subspace we do
```@example hubbard
N = 20
labels = Base.product(1:N, (:↑,:↓))
N_up = 2
N_down = 1
qn_spin = number_conservation(N_up, label -> :↑ in label) * number_conservation(N_down, label -> :↓ in label)
H = hilbert_space(labels, qn_spin)
```
The full hilbert space is of size `4^20 ≈ 10^12`, but the sector with 2 spin up and 1 spin down fermion is only of size `3800` and is generated without constructing the full hilbert space. Finally, we can get the matrix representation of the hamiltonian in this sector as
```@example hubbard
symham = hubbard_hamiltonian(N, 1.0, 4.0)
ham = matrix_representation(symham, H)
```

### No double occupation
When the onsite Coulomb interaction is very strong, there is a large energy penalty for double occupation of a site. In that case, we can restrict the Hilbert space to not allow double occupation of any site. Consider the site `k`, which has two labels `(k, :↑)` and `(k, :↓)`. We can use `number_conservation(0:1, label -> label[1] == k)` which says that the sum of occupation numbers of all labels where the first element of the label equals `k` is contained in the set `0:1`. To impose this for all sites, we take the product over all sites.
```@example hubbard
qn_no_double_occ = prod(number_conservation(0:1, label -> label[1] == k) for k in 1:N)
qn = qn_spin * qn_no_double_occ
H_ndo = hilbert_space(labels, qn)
```
This quantum number is a product of number conservations, so the sector is constructed without enumerating the full Hilbert space.

The matrix representation of the hamiltonian in this sector can be constructed as before, but now we need to specify `projection = true` as the symbolic hamiltonian maps states in the subspace to states outside the subspace. The keyword `projection = true` says to ignore those terms.
```@example hubbard
symham = hubbard_hamiltonian(N, 1, 0)
ham_ndo = matrix_representation(symham, H_ndo; projection = true)
```

## More advanced use of conserved quantities: Fractionalized hilbert space
The ``t-J_z`` model is
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + J_z \sum_{i} S^z_i S^z_{i+1},
```
where ``S^z_i = n_{i,↑} - n_{i,↓}`` and double occupation is forbidden. This model features a fractionalized hilbert space where the Hilbert space splits into exponentially many dynamically disconnected sectors, see [[1910.06341]](https://arxiv.org/abs/1910.06341). We implement the hamiltonian as
```@example hubbard
function tjz(N,t,Jz)
    @fermions c
    spins = (:↑,:↓)
    Sz(i) = c[i,:↑]'c[i,:↑] - c[i,:↓]'c[i,:↓]
    -t*sum(c[i,σ]'c[i+1,σ] + hc for σ in spins for i in 1:N-1) + Jz*sum(Sz(i)Sz(i+1) for i in 1:N-1)
end
```

To construct the hilbert space for this model, we first use the same conservation as above to restrict to no double occupation and conserve spin. This gives a conserved quantum numbers which is a product of number conservations and so it is efficient in generating states.
```@example hubbard
N = 12
N_up = 4
N_down = 1
labels = Base.product(1:N, (:↑,:↓))
qn_spin = number_conservation(N_up, label -> :↑ in label) * number_conservation(N_down, label -> :↓ in label)
qn = qn_spin * qn_no_double_occ
H = hilbert_space(labels, qn)
symham = tjz(N, 1, 1/4)
ham = matrix_representation(symham, H; projection = true)
```
This space fragments into more sectors, which are labelled by the order of occupied spins. By defining a function that maps states to the spin order, we can split the hilbert space into those fragments. The function can be defined as
```@example hubbard
function spin_order(state, H)
    N = length(keys(H)) ÷ 2
    occupations = map(1:N) do n
        if FermionicHilbertSpaces.occupation(state, (n, :↑), H)
            :↑
        elseif FermionicHilbertSpaces.occupation(state, (n, :↓), H)
            :↓
        else
            :hole
        end
    end
    filter(x -> x != :hole, occupations)
end
```
We can then construct the fractionalized hilbert space as
```@example hubbard
qnfrac = Base.Fix2(spin_order, H)
Hfrac = FermionicHilbertSpaces.symmetrize(H, qnfrac)
Hfrac.symmetry.qntofockstates
```
This splits the space of dimension 3960 into 5 fragments each of size 792. 

Let's find the half-chain entanglement entropy of the ground state in each sector. We can iterate over sectors by calling `sectors(H)`, and we can use `subregion` to find the hilbert space of a subsystem. 
```@example hubbard
using Arpack, LinearAlgebra
using FermionicHilbertSpaces: sectors
left_labels = Base.product(1:(N÷2), (:↑,:↓))
entropy(ρ) = sum(-λ * log(λ) for λ in eigvals(Hermitian(ρ)) if λ > 1e-12)
map(sectors(Hfrac)) do Hsec
    ham = matrix_representation(symham, Hsec; projection = true)
    vals, vecs = eigs(ham, nev = 1)
    Hleft = subregion(left_labels, Hsec)
    rholeft = partial_trace(vecs[:,1]*vecs[:,1]', Hsec => Hleft)
    entropy(rholeft)
end
```
