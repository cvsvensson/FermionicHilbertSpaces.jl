# # η-pairing in a large Hubbard chain

# Symbolic kets store only their nonzero amplitudes and operators act on them without a basis, so exact
# calculations are possible in spaces far too large for vectors. Here we build Yang's η-pairing eigenstates[^1] of a
# 200-site Hubbard chain and show that they have off-diagonal long-range order: their pair correlations do not decay
# with distance.

# ## η-pairing states

# The η-pairing operator ``η^\dagger = \sum_j (-1)^j c_{j\uparrow}^\dagger c_{j\downarrow}^\dagger`` creates a doubly
# occupied site with a staggered phase, which makes it commute with the hopping on a bipartite lattice. The states
# ``ψ_n = (η^\dagger)^n|0⟩`` are then eigenstates of the Hubbard model with energy ``n(U - 2μ)``. Each is an
# equal-weight sum over the ``\binom{L}{n}`` ways to place ``n`` doublons, so the pair correlation
# ```math
# ⟨c_{i\uparrow}^\dagger c_{i\downarrow}^\dagger c_{j\downarrow} c_{j\uparrow}⟩ = (-1)^{i+j} \frac{n(L-n)}{L(L-1)}
# ```
# is the same for every pair of distinct sites.

# ## A 200-site chain

# The Hilbert space has dimension ``4^L = 2^{2L}``, while ``ψ_n`` is a sum of only ``\binom{L}{n}`` basis states.
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.NonCommutativeProducts: nterms

@fermions c
L = 200
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
t, U, μ = 1.0, 4.0, 0.5
n(j, s) = c[j, s]' * c[j, s]
ham = sum(-t * c[j, s]' * c[j + 1, s] + hc for j in 1:(L - 1) for s in (:↑, :↓)) +
    sum(U * n(j, :↑) * n(j, :↓) - μ * (n(j, :↑) + n(j, :↓)) for j in 1:L)
η = sum((-1)^j * c[j, :↑]' * c[j, :↓]' for j in 1:L)
ψs = accumulate((ψ, _) -> η * ψ, 1:2; init = Ket("0"^(2L), H))
for (N, ψ) in enumerate(ψs)
    @assert nterms(ψ) == binomial(L, N)
    @assert norm(ham * ψ - N * (U - 2μ) * ψ) < 1.0e-10
end
(log2_dimension = log2(dim(H)), terms_in_ψ₂ = nterms(ψs[2]))

# The pair correlation of ``ψ_2``, here from site 1 to every other site, is the same at all distances.
pair(i, j) = c[i, :↑]' * c[i, :↓]' * c[j, :↓] * c[j, :↑]
ψ = ψs[2]
correlation(r) = dot(ψ, pair(1, 1 + r) * ψ) / dot(ψ, ψ)
@assert all(r -> correlation(r) ≈ (-1)^r * 2(L - 2) / (L * (L - 1)), 1:(L - 1))
correlation(L - 1)

# [^1]: C. N. Yang, "η pairing and off-diagonal long-range order in a Hubbard model", [Phys. Rev. Lett. **63**, 2144 (1989)](https://doi.org/10.1103/PhysRevLett.63.2144).
