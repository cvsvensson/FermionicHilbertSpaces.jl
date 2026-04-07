# Constrained Hilbert spaces and conserved quantum numbers
The basis states of a Hilbert space can be restricted to a subset of states and organized into sectors with different quantum numbers. 

You can apply a constraint either when constructing a space,
`hilbert_space(f, labels, constraint)`,
when doing tensor products
`tensor_product((H1, H2, ...); constraint)`,
or afterwards with
`constrain_space(H, constraint)`.

We define two common types of constraints for conserved quantum numbers:
- **NumberConservation** — Conserves a (possibly weighted) particle number
- **ParityConservation** — Conserves parity

As well as some types to help with constructing custom constraints:
- **FilterConstraint** — Flexible filtering using a boolean test on states (iterates over all states)
- **BlockConstraint** — Groups states into sectors using a custom rule (iterates over all states)
- **AdditiveConstraint** — Generalization of number conservation, which can be used for more complicated additive quantum numbers (can be used by iterating over all states, or by branch pruning)
- **BranchConstraint** — Efficient generation of states using branch pruning

One can also combine constraints by taking products.

### NumberConservation and ParityConservation
- `NumberConservation(total=missing, subspaces=missing, weights=missing)`:
    Conserves a (possibly weighted) particle number.
    - `total` can be a single integer or a collection of allowed values.
    - `subspaces` can restrict the count to selected subspaces (or modes).
    - `weights` lets each selected subspace contribute with a different integer weight.
    Examples: `NumberConservation()`, `NumberConservation(2)`, `NumberConservation(0:1, Hup)`, `NumberConservation(-1:1, [Hl, Hr], [1, -1])`.

```@example constraints 
using FermionicHilbertSpaces
@fermions f
Hf = hilbert_space(f, 1:3, NumberConservation(2)) # Single sector with 2 particles
```

```@example constraints 
@boson b
Hb = hilbert_space(b, 10)
H = tensor_product(Hf, Hb; constraint = NumberConservation(0:1, [Hf,Hb] , [1,-1])) # the number of fermions in Hf minus the number of bosons in Hb must be 0 or 1
basisstates(H)
```
- `ParityConservation(parities=[-1, 1], subspaces=missing)`:
    Conserves fermionic parity.
    - Allowed parities are `-1` (odd) and `1` (even).
    - `ParityConservation()` keeps both sectors (odd first, then even).
    - `ParityConservation([1])` or `ParityConservation(1)` keeps only even parity.
    - `subspaces` lets you enforce parity on a specific part of a tensor-product space.

```@example constraints 
constrain_space(H, ParityConservation(1, [Hb])) # keep only even parity of the bosonic part
```

### Filter and Block
- `FilterConstraint(...)`:
    A flexible filtering constraint.
    - `FilterConstraint(reducer)` applies `reducer(state)` and keeps states where it returns `true`.
    - `FilterConstraint(subspaces, subspace_functions, reducer)` first applies the subspace_functions to the states in each subspace, then passes them to the `reducer` function.
    Use this for custom selection rules that are easiest to express as a boolean test on complete states.

- `BlockConstraint(...)`:
    Like `FilterConstraint`, but for grouping states into sectors.
    - The reducer should return a sector label/key (or `missing` to discard a state).
    - States with the same returned key are collected into the same block.
    Use this when you want an explicit block structure from a custom rule.

```@example constraints
@bosons b
H = hilbert_space(b, 1:2, 3)
using FermionicHilbertSpaces: FilterConstraint, BlockConstraint, particle_number
constraint = FilterConstraint(H.clusters, particle_number, issorted)
basisstates(constrain_space(H, constraint)) # keep only states with sorted particle numbers
```

```@example constraints
constraint = BlockConstraint(H.clusters, particle_number, numbers -> issorted(numbers) ? first(numbers) : missing) #  keep only states with sorted particle numbers, organize them into blocks according to the number of particles in the first mode
constrain_space(H, constraint).qn_to_states
```

### Product

Constraints can be combined by multiplying them:
```@example constraints
Hs = [hilbert_space(f,2k-1:2k) for k in 1:3]
constraint = prod(NumberConservation(0:1, H) for H in Hs) * NumberConservation(2) # each mode can be occupied by at most one fermion, and the total number of fermions must be 2
H = tensor_product(Hs; constraint)
```

### BranchConstraint
- `BranchConstraint(f)`:
    For efficient constrained generation using branch pruning.
    - `f(partial_state, depth, spaces) -> Bool` is evaluated during state construction.
    - Returning `false` prunes the branch immediately.
    When branches can be pruned early, this avoids the exponentially large full Hilbert space.


## Examples

### Spin
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
sector([2,0], Hblocks)
```
or more efficiently by only constructing that specific sector in the first place
```@example spin
constrain_space(H, NumberConservation(2, Hup)*NumberConservation(0, Hdn))
```

### Hubbard model
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
Hup = hilbert_space(f, [(i, :↑) for i in 1:N], NumberConservation(Nup))
Hdn = hilbert_space(f, [(i, :↓) for i in 1:N], NumberConservation(Ndn))
H = tensor_product((Hup, Hdn))
```
The full hilbert space is of size `4^20 ≈ 10^12`, but the sector with 2 spin up and 1 spin down fermion is only of size `3800` and is generated without constructing the full hilbert space. Finally, we can get the matrix representation of the hamiltonian in this sector as
```@example hubbard
symham = hubbard_hamiltonian(f, N, 1.0, 4.0)
ham = matrix_representation(symham, H)
```

#### No double occupation
When the onsite Coulomb interaction is very strong, there is a large energy penalty for double occupation of a site. In that case, we can restrict the Hilbert space to not allow double occupation of any site. Consider the site `k`, which has two labels `(k, :↑)` and `(k, :↓)`. We can use `number_conservation(0:1, label -> label[1] == k)` which says that the sum of occupation numbers of all labels where the first element of the label equals `k` is contained in the set `0:1`. To impose this for all sites, we take the product over all sites.
```@example hubbard
no_double_occ = prod(NumberConservation(0:1, [f[(k, :↑)], f[(k, :↓)]]) for k in 1:N)
H_ndo = constrain_space(H, no_double_occ)
```
This quantum number is a product of number conservations, so the sector is constructed without enumerating the full Hilbert space.

The matrix representation of the hamiltonian in this sector can be constructed as before, but now we need to specify `projection = true` as the symbolic hamiltonian maps states in the subspace to states outside the subspace. The keyword `projection = true` says to ignore those terms.
```@example hubbard
symham = hubbard_hamiltonian(f, N, 1, 0)
ham_ndo = matrix_representation(symham, H_ndo; projection = true)
```

### Fractionalized hilbert space with BranchConstraint
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
            site, spin = only(spaces[n].modes).label
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