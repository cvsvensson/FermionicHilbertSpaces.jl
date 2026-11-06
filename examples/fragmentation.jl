# # Hilbert-space fragmentation and quantum many-body scars
#
# Some Hamiltonians split the space into blocks that no conventional (local) symmetry labels. [`fragment`](@ref)
# finds the block containing given basis states, by following the nonzero matrix elements of the Hamiltonian from them.

using FermionicHilbertSpaces, LinearAlgebra, Plots

# ## The PXP model
# Rydberg atoms in the blockade regime are described by the PXP chain
# ```math
# H = \sum_k P_{k-1} X_k P_{k+1},
# ```
# where ``X_k`` flips atom ``k`` and ``P_k`` projects it onto the ground state ``|{\downarrow}\rangle``. An atom can
# only flip when its neighbours are in the ground state. We take a ring, so atoms ``N`` and ``1`` are neighbours.

@spins S 1 // 2
N = 14
Hfull = hilbert_space(S, 1:N)
P(k) = 1 // 2 - S[mod1(k, N)][:z]
X(k) = 2S[k][:x]
pxp = sum(P(k - 1) * X(k) * P(k + 1) for k in 1:N)
H = fragment(pxp, Hfull, ["↓"^N])
dim(H), dim(Hfull)

# The fragment of the state with all atoms in the ground state is the space without neighbouring excited atoms, which
# a constraint on the basis states also gives. The blockade space, with open or periodic boundaries, is a single
# fragment, so for the PXP model `fragment` gives no new information.
#
# A [`BranchConstraint`](@ref) (see [Constrained Hilbert spaces](../conservation.md)) builds the states site by site and
# calls its function after each site is set, with the partial state `p`, the current site `d` and the spaces.
# Returning `false` discards every state that starts that way, so the blockade space is built without enumerating the
# full space. The function rejects two neighbouring excited atoms and, once the last site is set, an excited pair across
# the bond that closes the ring.

up(s) = s.m > 0
blockade = BranchConstraint(
    (p, d, spaces) -> d == 1 ||
        !(up(p[d]) && up(p[d - 1])) && (d < length(spaces) || !(up(p[d]) && up(p[1])))
)
Hblockade = hilbert_space(S, 1:N, blockade)
@assert Set(basisstates(H)) == Set(basisstates(Hblockade))

# Each term ``P_{k-1} X_k P_{k+1}`` maps blockade states to blockade states, but the Hamiltonian is stored expanded
# into products such as ``S^z_{k-1} S^x_k``, which map states of the fragment outside it. `projection = true` drops
# those matrix elements.

ham = representation(pxp, H, Matrix; projection = true);

# ## Scars
# The states ``|\mathbb{Z}_2\rangle = |{\uparrow\downarrow\uparrow\downarrow\cdots}\rangle`` and its translate
# ``|\mathbb{Z}_2'\rangle`` are swapped by the dynamics. We follow their cat state
# ``(|\mathbb{Z}_2\rangle + |\mathbb{Z}_2'\rangle)/\sqrt{2}``, which has zero momentum and is even under inversion, in
# that sector of the ring's translations and reflections, built with [`symmetric_sector`](@ref).

perms = [[mod1(r * i + s, N) for i in 1:N] for s in 0:(N - 1) for r in (1, -1)]
B = symmetric_sector(H, factors(H), (perms, ones(length(perms))))
E, V = eigen(Hermitian(Matrix(B' * ham * B)))
basis_vector(label) = (ψ = zeros(dim(H)); ψ[state_index(label, H)] = 1; ψ)
catstate = B' * (basis_vector("↑↓"^(N ÷ 2)) + basis_vector("↓↑"^(N ÷ 2))) / sqrt(2)
down = B' * basis_vector("↓"^N);

# A quench from the cat state shows revivals of the fidelity, while the state with all atoms in the ground state
# relaxes without revivals.

ts = range(0, 20; length = 200)
function fidelity(ψ)
    c = V' * ψ
    return [abs2(sum(abs2.(c) .* exp.(-im * E * t))) for t in ts]
end
plot(ts, fidelity(catstate); label = "cat state", xlabel = "t", ylabel = "fidelity", size = (400, 250))
plot!(ts, fidelity(down); label = "↓↓↓↓…")

# The revivals come from a tower of eigenstates at nearly equally spaced energies, the quantum many-body scars, which
# overlap much more with the cat state than the other eigenstates do. The scar states alternate between zero momentum
# and momentum ``\pi``, so this sector holds every other state of the tower.

scatter(
    E, abs2.(V' * catstate); yscale = :log10, ylims = (1.0e-9, 1), label = false, ms = 2, xlabel = "E",
    ylabel = "overlap with cat state", size = (400, 250)
)

# ## Fragmentation from dipole conservation
# The pair-hopping chain, here with open boundaries,
# ```math
# H = \sum_k S^+_k S^-_{k+1} S^-_{k+2} S^+_{k+3} + \mathrm{h.c.}
# ```
# conserves the charge ``\sum_k S^z_k`` and the dipole moment ``\sum_k k S^z_k``. These do not label all its blocks:
# the two states below both lie in the sector with zero charge and zero dipole moment, but in fragments of very
# different sizes.

N = 12
Hfull = hilbert_space(S, 1:N)
pairhop = sum(S[k][:+] * S[k + 1][:-] * S[k + 2][:-] * S[k + 3][:+] for k in 1:(N - 3)) + hc
neutral = AdditiveConstraint(0; maps = s -> s.m) * AdditiveConstraint(0; maps = [s -> k * s.m for k in 1:N])
Hneutral = hilbert_space(S, 1:N, neutral)
seeds = ["↑↓↑↓↑↓↓↑↓↑↓↑", "↑↑↑↓↓↓↓↓↓↑↑↑"]
@assert all(!iszero(state_index(s, Hneutral)) for s in seeds)
[dim(fragment(pairhop, Hfull, [s])) for s in seeds]

# Splitting the whole sector into fragments shows that one fragment holds a large share of it, while most of the
# others are single frozen states.

remaining = Set(basisstates(Hneutral))
sizes = Int[]
while !isempty(remaining)
    F = fragment(pairhop, Hneutral, [first(remaining)])
    setdiff!(remaining, basisstates(F))
    push!(sizes, dim(F))
end
(; sector = dim(Hneutral), fragments = length(sizes), frozen = count(==(1), sizes), largest = maximum(sizes))
