# # Spatial symmetry sectors
# Block-diagonalizing a Hamiltonian with a spatial symmetry such as translations shows which symmetry sector holds the ground state and shrinks the matrices to diagonalize. This example builds the sectors of an irreducible representation (irrep) with `symmetric_sector` and uses them to resolve the spectrum of a Heisenberg ring by momentum and total spin.
#
# ## Permutations and irreducible representations
#
# Suppose a group $G$ acts on the sites of a system by permutations. Each
# permutation induces an operator $U(g)$ on the Hilbert space. For an
# irreducible character $\chi$, the projector onto its symmetry sector is
#
# ```math
# P_\chi = \frac{d_\chi}{|G|}
#     \sum_{g \in G} \chi(g)^* U(g),
# \qquad d_\chi = \chi(e).
# ```
#
# To calculate the projector, you need the permutation operator and the value
# $\chi(g)$ for each $g$. FermionicHilbertSpaces.jl handles the permutations,
# and you supply the characters. With a list of permutations and their weights
# $\chi(g)^*$ for a particular irrep, `symmetric_sector(H, Hs, (perms, weights))`
# constructs a matrix whose columns form a basis for the sector.
#
# For translations the characters are simple phases, as shown below. For other
# groups, such as the dihedral group that also includes reflections, character
# tables can be computed with e.g. [Oscar.jl](https://github.com/oscar-system/Oscar.jl).

# ## Example: a Heisenberg ring
#
# Consider a spin chain with nearest-neighbour coupling:
#
# ```math
# H = J\sum_{k=1}^{N}\mathbf{S}_k\cdot\mathbf{S}_{k+1},
# \qquad \mathbf{S}_{N+1}=\mathbf{S}_1.
# ```
#
# We first separate total-magnetization sectors $M=\sum_k S_k^z$,
# then use the ring's translation symmetry and its spin-rotation symmetry.
# Let's consider 5 sites, each with spin 3/2.

using FermionicHilbertSpaces, LinearAlgebra

N = 5
J = 1.0

@spins S 3 // 2
magnetization = sum(S[k][:z] for k in 1:N)
space = hilbert_space(S, 1:N, magnetization)

H = J * sum(S[k][op] * S[mod1(k + 1, N)][op] for k in 1:N for op in (:x, :y, :z))

# Translations form the cyclic group $\mathbb{Z}_N$ with $N$ elements. We write
# each as a permutation vector, which is the format FermionicHilbertSpaces.jl
# takes: translation `s` sends site `i` to `mod1(i + s, N)`, for `s` in `0:N-1`.

perms = [[mod1(i + s, N) for i in 1:N] for s in 0:N-1]

# Every irrep of $\mathbb{Z}_N$ is one-dimensional and labelled by a momentum
# $k$, with character $\chi_k(s) = e^{-2\pi i k s/N}$. The weights are the
# conjugated characters $\chi_k(s)^* = e^{2\pi i k s/N}$.

momenta = (-(N - 1) ÷ 2):(N ÷ 2)
labels = ["k = $k" for k in momenta]
weights = [[cis(2π * k * s / N) for s in 0:N-1] for k in momenta];

# Translations commute with the total raising operator $S^+ = \sum_k S^+_k$. A multiplet with total spin $S$ has
# one state with $M = S$, and it is the one that $S^+$ annihilates. In each sector $M$, `symmetric_sector`
# constructs a matrix whose columns form a basis for an irrep's subspace, and `factors(sec)` identifies the tensor
# factors permuted by the group. Given these bases, `highest_weight_states` returns the states annihilated by $S^+$
# in each irrep, keyed by total spin. We diagonalize $H$ on that subspace; an absent block gets energy `Inf`.

using FermionicHilbertSpaces: highest_weight_states
Splus = sum(S[k][:+] for k in 1:N)
spins = filter(>=(0), quantumnumbers(space))
blocks = Dict(s => representation(H, sector(s, space)) for s in spins)
function tower_energies(w)
    irrep_basis(M) = (sec = sector(M, space); symmetric_sector(sec, factors(sec), (perms, w)))
    hws = highest_weight_states(space, Splus; basis = irrep_basis)
    return map(spins) do s
        haskey(hws, s) || return Inf
        V = hws[s]
        eigmin(Hermitian(Matrix(V' * blocks[s] * V)))
    end
end
energies = stack(map(tower_energies, weights); dims = 1)
E0, idx = findmin(energies)
@assert E0 ≈ eigmin(Hermitian(representation(H, space, Matrix)))
(; E0, irrep = labels[idx[1]], S = spins[idx[2]])

# Let's plot the results. The heatmap shows the lowest energy in each (irrep, $S$) block, the tower of states of the
# ring. Green marks the lowest energy for each irrep; red marks the lowest energy for each total spin.

using Plots
best_irrep_per_S = map(argmin, eachcol(energies))
best_S_per_irrep = map(argmin, eachrow(energies))
p = heatmap(
    labels, spins, permutedims(energies) ./ J;
    xlabel="momentum (units of 2π/N)",
    ylabel = "total spin S",
    title = "Tower of states", colorbar_title = "E/J",
    c = :viridis, yticks = spins, frame = :box
)
scatter!(
    p, labels[best_irrep_per_S], spins;
    marker=:vline, markersize=6, markerstrokewidth=5,
    color = :red, label = "lowest state at fixed S"
)
scatter!(
    p, labels, spins[best_S_per_irrep];
    marker=:hline, markersize=8, markerstrokewidth=5,
    color=:green, label="lowest state in fixed irrep")
