# Constrained Hilbert spaces and conserved quantum numbers
Conservation laws let you work in a small part of an exponentially large Hilbert space. This page shows how a constraint selects basis states and groups them into sectors of fixed quantum numbers, lists the available constraints, and ends with worked examples where the full space would be far too large to build.

## Constraints and sectors
A constraint selects basis states, and can split them into sectors labelled by quantum numbers. It is given when building a space, either to `hilbert_space` or as the `constraint` keyword of `tensor_product`, and `constrain_space` applies one to an existing space:
```@example constraints
using FermionicHilbertSpaces
@fermions f
N = sum(f[k]'f[k] for k in 1:4)
Hleft = hilbert_space(f, 1:2)
Hright = hilbert_space(f, 3:4)
Hone = tensor_product(Hleft, Hright; constraint = N => 1)
Hfew = constrain_space(hilbert_space(f, 1:4), N => 0:1)
H = hilbert_space(f, 1:4, N)
```

The sector labels of a space are its quantum numbers. A sector is a space of its own, and an operator that conserves the quantity can be represented directly on it:
```@example constraints
quantumnumbers(H)
H2 = sector(2, H)
hopping = sum(f[k]'f[k+1] + hc for k in 1:3)
representation(hopping, H2)
```

### Writing a constraint
A constraint is `quantity => f => g => ... => allowed`, where everything after `quantity` is optional:
- `quantity` is a diagonal operator, such as `N`, or a function of the basis state.
- `f`, `g`, … are functions applied to the value of the quantity in turn.
- `allowed` is a single value or a collection of values.

With `v` the value after the functions, the end of the constraint decides what it does:

| the constraint ends in | example | keeps the states where | labels sectors by |
|:--- |:--- |:--- |:--- |
| a value `a` | `N => 2` | `v == a` | none |
| a collection `c` | `N => 0:2` | `v ∈ c` | `v` |
| a quantity or function giving `true` or `false` | `N => isodd` | `v` is `true` | none |
| a quantity or function giving anything else | `N`, `N => parity` | always | `v` |

A state where a function returns `missing` is discarded. To allow a single tuple or vector, wrap it, as in `=> [(0, 1)]`. A constraint that allows no value of its operator throws an error. Odd particle numbers, the sectors of even and odd particle number, and a ``\mathbb{Z}_3`` charge:
```@example constraints
Hodd = hilbert_space(f, 1:4, N => isodd)
parity(v) = mod(v, 2)
Hparity = hilbert_space(f, 1:4, N => parity => 0:1)
Q = sum(k * f[k]'f[k] for k in 1:4)
charge(v) = mod(v, 3)
Hcharge = hilbert_space(f, 1:4, Q => charge => 0)
dim(Hodd), quantumnumbers(Hparity), dim(Hcharge)
```
[`ParityConservation`](@ref) is a shorthand for `N => (v -> (-1)^v) => [-1, 1]` with `N` the total number of fermions.

Several constraints go in a tuple or a vector, and a sector label then has one entry per quantity. Labels are integers when every term of the operator takes integer values, and rationals otherwise. Coefficients must be integers or rationals, so that the values are exact. A spin chain split into sectors of total ``S^z``, and the Rydberg blockade, which forbids neighbouring excitations:
```@example constraints
@spins S 1 // 2
Sz = sum(S[i][:z] for i in 1:6)
Hspin = hilbert_space(S, 1:6, Sz)
up(i) = S[i][:z] + 1 // 2
blockade = [up(i) * up(i + 1) => 0 for i in 1:9]
Hpxp = hilbert_space(S, 1:10, blockade)
quantumnumbers(Hspin), dim(Hpxp)
```

A function of the basis state works the same way:
```@example constraints
nup(state) = count(s -> s.m > 0, state.states)
dim(hilbert_space(S, 1:6, nup => 2)), dim(hilbert_space(S, 1:6, s -> nup(s) <= 1))
```
Applied to a space with sectors, a constraint written this way keeps those sectors, and its own labels follow theirs.

### How a constrained space is built
The states are built one factor at a time, as a tree. Each term of an operator is assigned to the factors it acts on, so after each factor the value so far is known, along with the range that the remaining factors can still add. A branch that cannot reach an allowed value is skipped. For a particle number, a magnetization or a blockade, every branch that is kept leads to a kept state, so the cost grows with the size of the result rather than the full space. A term that acts on several factors is tabulated on their product, which may have at most 4096 states.

A function of the basis state cannot be split this way, so it is evaluated on every state of the full space, which must be small enough to build. `constrain_space` on a space that is already constrained also checks its states one by one.

## Lower-level constraints
#### NumberConservation
Conserves the particle number, or a weighted sum of the particle numbers of selected spaces. The allowed total is one value or a collection; leaving it out keeps all sectors.
```@example constraints
@boson b
Hf = hilbert_space(f, 1:3)
Hb = hilbert_space(b, 10)
# fermions in Hf minus bosons in Hb is 0 or 1
imbalance = NumberConservation(0:1, [Hf, Hb], [1, -1])
tensor_product(Hf, Hb; constraint = imbalance)
```

#### ParityConservation
Conserves fermion parity, `1` for even and `-1` for odd states; without arguments it keeps both sectors.
```@example constraints
quantumnumbers(hilbert_space(f, 1:4, ParityConservation()))
```

#### Products of constraints
Multiplying constraints imposes all of them, and the quantum number of a sector has one entry per factor. Here each pair of modes holds at most one fermion, and there are two fermions in total:
```@example constraints
pairs = [hilbert_space(f, 2k-1:2k) for k in 1:3]
pair_constraint = prod(NumberConservation(0:1, Hk) for Hk in pairs) * NumberConservation(2)
tensor_product(pairs; constraint = pair_constraint)
```

#### AdditiveConstraint
Conserves a sum of contributions from the factors of a tensor product, given by `maps` applied to the state of each factor. It works for any space, also one without symbolic operators:
```@example constraints
spins = [hilbert_space(S[k]) for k in 1:4]
magnetization = AdditiveConstraint(-2:2; maps = s -> s.m)
Hmag = tensor_product(spins; constraint = magnetization)
quantumnumbers(Hmag)
```

#### FilterConstraint
Keeps the states for which a function returns `true`. With `maps` and `spaces`, the function receives the value of each map on the state of its space. Here the particle numbers of two bosonic modes must be sorted:
```@example constraints
import FermionicHilbertSpaces: particle_number
@bosons d
Hbb = hilbert_space(d, 1:2, 3)
sorted = FilterConstraint(issorted; maps = particle_number, spaces = factors(Hbb))
basisstates(constrain_space(Hbb, sorted))
```

#### SectorConstraint
Like `FilterConstraint`, but the function returns a sector label, or `missing` to discard the state. Here the two modes must hold different numbers of bosons, and the sectors are labelled by the total:
```@example constraints
distinct = SectorConstraint(n -> allunique(n) ? sum(n) : missing;
    maps = particle_number, spaces = factors(Hbb))
Hdistinct = constrain_space(Hbb, distinct)
map((qn, Hsec) -> qn => basisstates(Hsec), quantumnumbers(Hdistinct), sectors(Hdistinct))
```

#### BranchConstraint
Builds the states one factor at a time and calls `f(partial_state, depth, spaces)` after each factor is assigned. Returning `false` discards every state that starts this way, so a sector much smaller than the full space is found without enumerating it. Here no two neighbouring modes are both occupied:
```@example constraints
no_neighbours = BranchConstraint((p, depth, spaces) -> depth == 1 || particle_number(p[depth]) + particle_number(p[depth-1]) < 2)
dim(hilbert_space(f, 1:20, no_neighbours))
```

## Examples

#### Spinful fermions
The package does not special-case spinful fermions: spin is an extra label. If the numbers of spin up and spin down fermions are separately conserved, the space divides into sectors labelled by both numbers:
```@example spin
using FermionicHilbertSpaces
@fermions f
labels = [(i, σ) for i in 1:4 for σ in (:↑, :↓)]
Nup = sum(f[i, :↑]'f[i, :↑] for i in 1:4)
Ndn = sum(f[i, :↓]'f[i, :↓] for i in 1:4)
H = hilbert_space(f, labels, (Nup, Ndn))
```
The sector with two spin up and no spin down fermions can be picked out of this space, or built directly, which is cheaper since it never enumerates the full space:
```@example spin
two_up_none_down = (Nup => 2, Ndn => 0)
@assert Set(basisstates(sector((2, 0), H))) == Set(basisstates(hilbert_space(f, labels, two_up_none_down)))
sector((2, 0), H)
```
A state found on a sector, such as the ground state of the hopping there, is placed in the full space by `embed`, and `indices` gives the positions of the sector's states in it:
```@example spin
using LinearAlgebra
H20 = sector((2, 0), H)
hopping = sum(f[i, σ]'f[i+1, σ] + hc for i in 1:3 for σ in (:↑, :↓))
ψ = eigvecs(Hermitian(representation(hopping, H20, Matrix)))[:, 1]
@assert embed(ψ, H20 => H)[indices((2, 0), H)] ≈ ψ
indices((2, 0), H)
```

#### Hubbard model
The Hubbard model
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + U \sum_i n_{i,↑} n_{i,↓}
```
conserves the number of spin up and spin down fermions separately. On 20 sites the full space has dimension ``4^{20} \approx 10^{12}``, but the sector with two spin up fermions and one spin down fermion is small, and is generated directly:
```@example hubbard
using FermionicHilbertSpaces
function hubbard_hamiltonian(c, N, t, U)
    spins = (:↑,:↓)
    sum(-t * c[i,σ]' * c[i+1,σ] + hc for σ in spins for i in 1:N-1) +
    sum(U * c[i,:↑]'c[i,:↑] * c[i,:↓]'c[i,:↓] for i in 1:N)
end

@fermions f
N = 20
Hup = hilbert_space(f, [(i, :↑) for i in 1:N], NumberConservation(2))
Hdn = hilbert_space(f, [(i, :↓) for i in 1:N], NumberConservation(1))
H = tensor_product(Hup, Hdn)
ham = representation(hubbard_hamiltonian(f, N, 1.0, 4.0), H);
size(ham)
```

##### No double occupation
For strong interactions, double occupation can be forbidden by constraining the number of doubly occupied sites to zero. The hamiltonian maps some states of this subspace outside it, and `projection = true` drops those terms:
```@example hubbard
D = sum(f[k, :↑]'f[k, :↑] * f[k, :↓]'f[k, :↓] for k in 1:N)
H_ndo = constrain_space(H, D => 0)
ham_ndo = representation(hubbard_hamiltonian(f, N, 1, 0), H_ndo; projection = true);
size(ham_ndo)
```

#### Tavis–Cummings model
``N`` spin-1/2 emitters coupled to a cavity mode,
```math
H = \omega a^\dagger a + \omega_0 \sum_i S^z_i + g \sum_i \left(a^\dagger S^-_i + a S^+_i\right),
```
conserve the number of excitations, the boson number plus ``\sum_i (S^z_i + 1/2)``:
```@example tavis_cummings
using FermionicHilbertSpaces
N = 3
nmax = 4
@spins S 1//2
@boson a
spaces = [hilbert_space(a, nmax + 1); [hilbert_space(S[i]) for i in 1:N]]
excitations = a'a + sum(S[i][:z] + 1//2 for i in 1:N)
H = tensor_product(spaces; constraint = excitations)
```
The Hamiltonian is block diagonal in these sectors, one per excitation number.

#### Fractionalized Hilbert space of the t-Jz model
The ``t-J_z`` model
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + J_z \sum_{i} S^z_i S^z_{i+1},
```
with ``S^z_i = n_{i,↑} - n_{i,↓}`` and no double occupation, has a Hilbert space that splits into exponentially many dynamically disconnected sectors[^1]. Each sector is labelled by a spin pattern such as ↑↑↓↑↓: its states have the spins of the occupied sites in exactly that order, with holes in between. A `BranchConstraint` builds one such sector without the full space:
```@example hubbard
import FermionicHilbertSpaces: particle_number
function tjz(c, N, t, Jz)
    spins = (:↑,:↓)
    Sz(i) = c[i,:↑]'c[i,:↑] - c[i,:↓]'c[i,:↓]
    -t*sum(c[i,σ]'c[i+1,σ] + hc for σ in spins for i in 1:N-1) + Jz*sum(Sz(i)Sz(i+1) for i in 1:N-1)
end

# partials holds the fock numbers of the 2N modes, of which the first `depth` are assigned
function valid_state(partials, depth, spaces, expected_order)
    exc_count = 0
    prev_site = 0
    for n in 1:depth
        if particle_number(partials[n]) == 1
            exc_count += 1
            site, spin = only(spaces[n].modes).label
            exc_count > length(expected_order) && return false
            site == prev_site && return false # the spaces are ordered by site
            prev_site = site
            expected_order[exc_count] == spin || return false
        end
    end
    depth == length(spaces) && return exc_count == length(expected_order)
    return true
end
spin_order_constraint(order) = BranchConstraint((p, d, s) -> valid_state(p, d, s, order))

N = 16
labels = [(i, s) for i in 1:N for s in (:↑,:↓)]
Hfrac = hilbert_space(f, labels, spin_order_constraint([:↑, :↑, :↓, :↑, :↓]))
ham = representation(tjz(f, N, 1, 1/4), Hfrac; projection = true);
dim(Hfrac)
```
The full space of these modes has dimension ``2^{32}``. `subregion` gives the space of the left half compatible with the constraint, which is what `partial_trace(ρ, Hfrac => Hsub_frac)` needs:
```@example hubbard
Hsub_full = hilbert_space(f, [(i, s) for i in 1:div(N,2) for s in (:↑,:↓)])
Hsub_frac = subregion(Hsub_full, Hfrac)
```

[^1]: T. Rakovszky, P. Sala, R. Verresen, M. Knap and F. Pollmann, "Statistical localization: from strong fragmentation to strong edge modes", [Phys. Rev. B 101, 125126 (2020)](https://doi.org/10.1103/PhysRevB.101.125126), [arXiv:1910.06341](https://arxiv.org/abs/1910.06341).
