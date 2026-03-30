
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
@fermions f
Hup = hilbert_space(f, [(i, :↑) for i in 1:4])
Hdn = hilbert_space(f, [(i, :↓) for i in 1:4])
H = tensor_product(Hup, Hdn)
```
If spin is conserved, one can use 
```@example spin
Hblocks = constrain_space(H, NumberConservation(Hup)*NumberConservation(Hdn))
```
to sort states according to the number of fermions with spin up and down. However, this package can't help to sort states into sectors with different total angular momentum, because that requires taking superpositions of different fock states.

To pick out the sector with 2 fermions with spin up and 0 fermions with spin down, one can extract it from the hilbert space defined above using `sector` as
```@example spin
sector((2,0), Hblocks)
```
or more efficiently by only constructing that specific sector in the first place
```@example spin
constrain_space(H, NumberConservation(2, Hup)*NumberConservation(0, Hdn))
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
function hubbard_hamiltonian(c, N, t, U)
    spins = (:↑,:↓)
    sum(-t * c[i,σ]' * c[i+1,σ] + hc for σ in spins for i in 1:N-1) + sum(U * c[i,:↑]'c[i,:↑] * c[i,:↓]'c[i,:↓] for i in 1:N)
end
```
Let's find the matrix representation of the hamiltonian in the sector with `N_up` spin up fermions and `N_down` spin down fermions. To find this subspace we do
```@example hubbard
@fermions f
N = 20
Nup = 2
Ndn = 1
Hup = hilbert_space(f,  [(i, :↑) for i in 1:N])
Hdn = hilbert_space(f, [(i, :↓) for i in 1:N])
constraint = NumberConservation(Nup, Hup)*NumberConservation(Ndn, Hdn)
H = tensor_product((Hup, Hdn); constraint)
```
The full hilbert space is of size `4^20 ≈ 10^12`, but the sector with 2 spin up and 1 spin down fermion is only of size `3800` and is generated without constructing the full hilbert space. Finally, we can get the matrix representation of the hamiltonian in this sector as
```@example hubbard
symham = hubbard_hamiltonian(f, N, 1.0, 4.0)
ham = matrix_representation(symham, H)
```

### No double occupation
When the onsite Coulomb interaction is very strong, there is a large energy penalty for double occupation of a site. In that case, we can restrict the Hilbert space to not allow double occupation of any site. Consider the site `k`, which has two labels `(k, :↑)` and `(k, :↓)`. We can use `number_conservation(0:1, label -> label[1] == k)` which says that the sum of occupation numbers of all labels where the first element of the label equals `k` is contained in the set `0:1`. To impose this for all sites, we take the product over all sites.
```@example hubbard
no_double_occ = prod(NumberConservation(0:1, [f[(k, s)] for s in (:↑,:↓)]) for k in 1:N)
H_ndo = constrain_space(H, constraint * no_double_occ)
```
This quantum number is a product of number conservations, so the sector is constructed without enumerating the full Hilbert space.

The matrix representation of the hamiltonian in this sector can be constructed as before, but now we need to specify `projection = true` as the symbolic hamiltonian maps states in the subspace to states outside the subspace. The keyword `projection = true` says to ignore those terms.
```@example hubbard
symham = hubbard_hamiltonian(f, N, 1, 0)
ham_ndo = matrix_representation(symham, H_ndo; projection = true)
```

## More advanced use of conserved quantities: Fractionalized hilbert space
The ``t-J_z`` model is
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + J_z \sum_{i} S^z_i S^z_{i+1},
```
where ``S^z_i = n_{i,↑} - n_{i,↓}`` and double occupation is forbidden. This model features a fractionalized hilbert space where the Hilbert space splits into exponentially many dynamically disconnected sectors, see [[1910.06341]](https://arxiv.org/abs/1910.06341). We implement the hamiltonian as
```@example hubbard
function tjz(c, N,t,Jz)
    spins = (:↑,:↓)
    Sz(i) = c[i,:↑]'c[i,:↑] - c[i,:↓]'c[i,:↓]
    -t*sum(c[i,σ]'c[i+1,σ] + hc for σ in spins for i in 1:N-1) + Jz*sum(Sz(i)Sz(i+1) for i in 1:N-1)
end
```
To construct the hilbert space in a specific sector, we need to go beyond simple particle number constraints. Each sector is defined by a spin ordering, e.g. '[:↑, :↑, :↓, :↑, :↓]'. Each state in this sector has three spin up electrons occupied, two spin down electrons occupied, and they come in that specific order with possible holes between them. 

This package can help construct these small sectors while avoiding the exponentially large hilbert space, by 
```@example hubbard
function valid_state(partials, depth::Int, spaces, expected_order)
    # partials is a list of 2N focknumbers, but only the ones up to 'depth' are assigned.
    # spaces is a list of 2N fermionic modes
    exc_count = 0
    prev_site = 0
    for n in 1:depth
        if count_ones(partials[n]) == 1
            exc_count += 1
            site, spin = spaces[n].label
            exc_count > length(expected_order) && return false # too many excitations
            site == prev_site && return false # two excitations on the same site (this uses the fact that we ordered the spaces by site)
            prev_site = site
            expected_order[exc_count] == spin || return false # check if the spin of the excitation matches the expected order
        end
    end
    depth == length(spaces) && return exc_count == length(expected_order)
    return true
end
using FermionicHilbertSpaces: BranchConstraint
spin_order_constraint(order) = BranchConstraint((p,d,s) -> valid_state(p,d,s,order))
```

```@example hubbard
N = 16
labels = [(i, s) for i in 1:N for s in (:↑,:↓)]
Hfull = hilbert_space(f,  labels)
```
This space is too large to deal with directly, but we can constrain it to the sector with spin order '[:↑, :↑, :↓, :↑, :↓]' as
```@example hubbard
Hfrac = hilbert_space(f, labels, spin_order_constraint([:↑, :↑, :↓, :↑, :↓]))
```
which has only dimension 4368. We can then construct hamiltonian in this sector by
```@example hubbard
symham = tjz(f, N, 1, 1/4)
ham = matrix_representation(symham, Hfrac; projection = true)
```
We can use `subregion` to find the hilbert space of a subsystem, taking into account the constraint. The full hilbert space of the left half of the system is
```@example hubbard
Hsub_full = hilbert_space(f,  [(i, s) for i in 1:div(N,2) for s in (:↑,:↓)])
```
but taking into account the constraint, we have
```@example hubbard
Hsub_frac = subregion(Hsub_full, Hfrac)
```
and now one can do partial traces `partial_trace(mat, Hfrac => Hsub_frac)`.