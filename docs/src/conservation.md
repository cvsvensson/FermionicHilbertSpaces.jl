# Constrained Hilbert spaces and conserved quantum numbers
This page shows how to restrict a Hilbert space to the states allowed by a conservation law and split it into sectors of fixed quantum numbers, so that large systems are only built in the sector you need. A constraint is applied when constructing a space, `hilbert_space(f, labels, constraint)`, in a tensor product, `tensor_product((H1, H2, ...); constraint)`, or afterwards with `constrain_space(H, constraint)`. Constraints can be combined by multiplying them.

## Constraints
`NumberConservation(total=missing; spaces=missing, weights=missing)` conserves a possibly weighted particle number. `total` is an integer or a collection of allowed values, `spaces` restricts the count to selected spaces or modes, and `weights` gives each selected space an integer weight. `ParityConservation(parities=[-1, 1], spaces=missing)` conserves fermionic parity, keeping odd (`-1`) and/or even (`1`) states; `ParityConservation(1)` keeps only even parity.
```@example constraints
using FermionicHilbertSpaces
@fermions f
@boson b
Hf = hilbert_space(f, 1:3, NumberConservation(2))
Hb = hilbert_space(b, 10)
# fermions in Hf minus bosons in Hb is 0 or 1
H = tensor_product(Hf, Hb; constraint = NumberConservation(0:1, [Hf, Hb], [1, -1]))
Heven = constrain_space(H, ParityConservation(1, [Hb])) # even boson parity only

# fixed fermion number with unconstrained bosons, built in two equivalent ways
@fermions c
@bosons d
Hc = hilbert_space(c, 1:3, NumberConservation(1))
Hd = hilbert_space(d, 1:3, 4)
H1 = tensor_product(Hc, Hd)
Hc_full = hilbert_space(c, 1:3)
H2 = tensor_product(Hc_full, Hd; constraint = NumberConservation(1, [Hc_full]))
(H1, H2)
```
The two tensor products have the same basis states in a different order.

Constraints for custom rules iterate over all states of the factor spaces. `FilterConstraint(reducer; spaces, maps)` applies `maps` to the states of each space in `spaces`, and keeps the states for which `reducer` returns `true`. `SectorConstraint` does the same but the reducer returns a sector label, or `missing` to discard the state, and states with equal labels form a sector. `AdditiveConstraint` generalizes number conservation to additive quantum numbers that can also be found by branch pruning, and `BranchConstraint(f)` prunes during state construction: `f(partial_state, depth, spaces)` returning `false` removes the branch, which avoids the exponentially large full space. Below, a `FilterConstraint` keeps the states whose particle numbers are sorted, and a `SectorConstraint` keeps those with all different particle numbers, grouped by total particle number. Products of constraints keep track of the quantum numbers of each factor:
```@example constraints
import FermionicHilbertSpaces: particle_number
Hb3 = hilbert_space(d, 1:2, 3)
sorted = FilterConstraint(issorted; maps = particle_number, spaces = factors(Hb3))
basisstates(constrain_space(Hb3, sorted))

distinct = SectorConstraint(numbers -> allunique(numbers) ? sum(numbers) : missing;
    maps = particle_number, spaces = factors(Hb3))
Hdistinct = constrain_space(Hb3, distinct)
map((qn, Hsec) -> qn => basisstates(Hsec), quantumnumbers(Hdistinct), sectors(Hdistinct))

# each pair of modes holds at most one fermion, and two fermions in total
Hs = [hilbert_space(f, 2k-1:2k) for k in 1:3]
pairs_constraint = prod(NumberConservation(0:1, Hk) for Hk in Hs) * NumberConservation(2)
tensor_product(Hs; constraint = pairs_constraint)
```

## Spinful fermions and the Hubbard model
This package does not special case spinful fermions: spin is just an extra label. If the numbers of spin up and spin down fermions are separately conserved, `NumberConservation(Hup)*NumberConservation(Hdn)` divides the space into sectors, and `sector([2,0], Hsectors)` picks out the one with two spin up and no spin down fermions. It is more efficient to construct only that sector, with `constrain_space(H, NumberConservation(2, Hup)*NumberConservation(0, Hdn))` or by passing the constraint to `tensor_product`. When a constraint is a product of number conservations, the sectors are found without enumerating the full Hilbert space, which is what makes the Hubbard model
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + U \sum_i n_{i,↑} n_{i,↓}
```
tractable for many sites. The full space on 20 sites has dimension ``4^{20} \approx 10^{12}``, but the sector with two spin up and one spin down fermion is small and is generated directly:
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
For strong interaction one can forbid double occupation. Allowing zero or one fermion on each site `k`, `NumberConservation(0:1, [f[k, :↑], f[k, :↓]])`, is a product over sites, so this sector is also constructed without enumerating the full space. The hamiltonian maps states of this subspace outside it, and `projection = true` ignores those terms:
```@example hubbard
local_modes(k) = [f[k, :↑], f[k, :↓]]
no_double_occ = prod(NumberConservation(0:1, local_modes(k)) for k in 1:N)
H_ndo = constrain_space(H, no_double_occ)
ham_ndo = representation(hubbard_hamiltonian(f, N, 1, 0), H_ndo; projection = true);
size(ham_ndo)
```

## Constraints without a particle number
Spins have no particle number, so the Tavis–Cummings model of ``N`` spin-1/2 emitters coupled to a cavity mode,
```math
H = \omega a^\dagger a + \omega_0 \sum_i S^z_i + g \sum_i \left(a^\dagger S^-_i + a S^+_i\right),
```
which conserves the boson number plus ``\sum_i (S^z_i + 1/2)``, needs the spin part of the excitation count written explicitly in an `AdditiveConstraint`, with one map per factor space. The Hamiltonian is block diagonal in the resulting sectors, one per excitation number.
```@example tavis_cummings
using FermionicHilbertSpaces
import FermionicHilbertSpaces: particle_number
N = 3
nmax = 4
@spins S 1//2
@boson a
spaces = [hilbert_space(a, nmax + 1); [hilbert_space(S[i]) for i in 1:N]]
excitations = AdditiveConstraint(0:nmax+N; spaces, maps = [particle_number; fill(s -> Int(s.m + 1//2), N)])
H = tensor_product(spaces; constraint = excitations)
```

The ``t-J_z`` model
```math
H = -t \sum_{i,\sigma} (c_{i,\sigma}^\dagger c_{i+1,\sigma} + \mathrm{h.c}) + J_z \sum_{i} S^z_i S^z_{i+1},
```
with ``S^z_i = n_{i,↑} - n_{i,↓}`` and no double occupation, has a fractionalized Hilbert space that splits into exponentially many dynamically disconnected sectors, see [[1910.06341]](https://arxiv.org/abs/1910.06341). Each sector is defined by a spin ordering such as `[:↑, :↑, :↓, :↑, :↓]`: its states have the spins of the occupied sites in exactly that order, with holes allowed in between. A `BranchConstraint` constructs such a sector without the exponentially large full space, and `subregion` gives the Hilbert space of a subsystem compatible with the constraint, for use in `partial_trace(mat, Hfrac => Hsub_frac)`.
```@example hubbard
function tjz(c, N, t, Jz)
    spins = (:↑,:↓)
    Sz(i) = c[i,:↑]'c[i,:↑] - c[i,:↓]'c[i,:↓]
    -t*sum(c[i,σ]'c[i+1,σ] + hc for σ in spins for i in 1:N-1) + Jz*sum(Sz(i)Sz(i+1) for i in 1:N-1)
end

function valid_state(partials, depth::Int, spaces, expected_order)
    # partials holds the fock numbers of the 2N modes, of which the first `depth` are assigned
    exc_count = 0
    prev_site = 0
    for n in 1:depth
        if count_ones(partials[n]) == 1
            exc_count += 1
            site, spin = only(spaces[n].modes).label
            exc_count > length(expected_order) && return false # too many excitations
            site == prev_site && return false # two excitations on one site; the spaces are ordered by site
            prev_site = site
            expected_order[exc_count] == spin || return false
        end
    end
    depth == length(spaces) && return exc_count == length(expected_order)
    return true
end
spin_order_constraint(order) = BranchConstraint((p,d,s) -> valid_state(p,d,s,order))

N = 16
labels = [(i, s) for i in 1:N for s in (:↑,:↓)]
Hfrac = hilbert_space(f, labels, spin_order_constraint([:↑, :↑, :↓, :↑, :↓]))
ham = representation(tjz(f, N, 1, 1/4), Hfrac; projection = true);

Hsub_full = hilbert_space(f, [(i, s) for i in 1:div(N,2) for s in (:↑,:↓)])
Hsub_frac = subregion(Hsub_full, Hfrac)
(Hfrac, Hsub_frac)
```
