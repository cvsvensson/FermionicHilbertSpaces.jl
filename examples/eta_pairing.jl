# # η-pairing in a large Hubbard chain

# Symbolic kets live in a Hilbert space without enumerating its basis, so they can be used where
# the full space is far too large for vectors. Here we build exact eigenstates of a Hubbard chain with
# `L = 100` spinful sites, a Hilbert space of dimension 4^100.
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
L = 100
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
dim(H)

# The Hubbard Hamiltonian with hopping `t`, on-site repulsion `U` and chemical potential `μ`,
# written as a list of local terms that sum to `H`.
t, U, μ = 1.0, 4.0, 0.5
n(j, s) = c[j, s]' * c[j, s]
hterms = [[-t * c[j, s]' * c[j+1, s] + hc for j in 1:L-1 for s in (:↑, :↓)];
          [U * n(j, :↑) * n(j, :↓) - μ * (n(j, :↑) + n(j, :↓)) for j in 1:L]]

# The η-pairing operator creates a doubly occupied site with a staggered phase.
# The phase makes η† commute with the hopping on a bipartite lattice, so the states
# ψₙ = (η†)ⁿ|0⟩ are eigenstates of `H` with energy Eₙ = n(U - 2μ).
η = sum((-1)^j * c[j, :↑]' * c[j, :↓]' for j in 1:L)

# Starting from the vacuum, we apply η† repeatedly. ψₙ is a sparse sum over the
# binomial(L, n) ways of placing n doublons.
ψs = [Kets(H)("0"^(2L)) + 0]
for _ in 1:2
    push!(ψs, η * ψs[end])
end

# `H` acts term by term on each symbolic state, and we compare with the expected energy.
for (k, ψ) in enumerate(ψs)
    N = k - 1
    E = N * (U - 2μ)
    residual = norm(sum(h * ψ for h in hterms) - E * ψ)
    println("n = $N: E = $E, basis states = $(length(ψ.dict)) = $(binomial(L, N)), residual = $residual")
end

# The residuals vanish exactly: the states form an equally spaced tower inside a space of 2^200 states.
# See C. N. Yang, Phys. Rev. Lett. 63, 2144 (1989) and Moudgalya, Regnault, Bernevig, arXiv:2004.13727.
