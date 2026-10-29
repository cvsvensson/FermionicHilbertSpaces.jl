# # Spatial symmetry sectors 
# This file demonstrates how to handle spatial symmetries like translations.
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
# Consider a spin chain with nearest-neighbour coupling and a
# uniform field:
#
# ```math
# H = J\sum_{k=1}^{N}\mathbf{S}_k\cdot\mathbf{S}_{k+1}
#     - h\sum_{k=1}^{N}S^z_k,
# \qquad \mathbf{S}_{N+1}=\mathbf{S}_1.
# ```
#
# We first separate total-magnetization sectors $M=\sum_k S_k^z$,
# then use the ring's translation symmetry.
# Let's consider 5 sites, each with spin 3/2.

using FermionicHilbertSpaces, LinearAlgebra

N = 5
J = 1.0
h = 0.04

@spins S 3 // 2
magnetization(s::SpinState) = s.m
space = hilbert_space(S, 1:N, SectorConstraint(sum; maps=magnetization))
mags = quantumnumbers(space)

H = J * sum(S[k][op] * S[mod1(k + 1, N)][op] for k in 1:N for op in (:x, :y, :z)) -
    h * sum(S[k][:z] for k in 1:N)

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

# In each magnetization sector, `symmetric_sector` constructs a matrix `P`
# whose columns form a basis for an irrep's subspace. `factors(sec)` identifies
# the tensor factors permuted by the group. We diagonalize $P^\dagger H P$;
# an absent irrep gets energy `Inf`. 

function sector_energies(M)
    sec = sector(M, space)
    HM = representation(H, sec)
    return map(weights) do w
        P = symmetric_sector(sec, factors(sec), (perms, w))
        size(P, 2) == 0 && return Inf
        eigmin(Hermitian(Matrix(P' * HM * P)))
    end
end
energies = stack(map(sector_energies, mags))
E0, idx = findmin(energies)
(; E0, irrep=labels[idx[1]], M=mags[idx[2]])

# Let's plot the results. The heatmap shows the lowest energy in each (irrep, $M$) block. Green marks the lowest energy for each irrep; red marks the lowest energy for each magnetization.

using Plots
best_irrep_per_M = map(argmin, eachcol(energies))
best_M_per_irrep = map(argmin, eachrow(energies))
p = heatmap(labels, mags, permutedims(energies);
    xlabel="momentum (units of 2π/N)",
    ylabel="magnetization M",
    title="Lowest energy per symmetry sector",
    c=:viridis, yticks=mags, frame=:box)
scatter!(p, labels[best_irrep_per_M], mags;
    marker=:vline, markersize=6, markerstrokewidth=5,
    color=:red, label="lowest state at fixed M")
scatter!(p, labels, mags[best_M_per_irrep];
    marker=:hline, markersize=8, markerstrokewidth=5,
    color=:green, label="lowest state in fixed irrep")
