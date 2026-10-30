# # η-pairing in a large Hubbard chain

# This tutorial applies a symbolic Hamiltonian to symbolic kets in a space too large for vectors, to build exact
# eigenstates of a Hubbard chain.
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
L = 200
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
log2(dim(H))

# The Hubbard Hamiltonian is a sum of local terms. Its matrix cannot be built at this size.
t, U, μ = 1.0, 4.0, 0.5
n(j, s) = c[j, s]' * c[j, s]
ham = sum(-t * c[j, s]' * c[j + 1, s] + hc for j in 1:(L - 1) for s in (:↑, :↓)) +
    sum(U * n(j, :↑) * n(j, :↓) - μ * (n(j, :↑) + n(j, :↓)) for j in 1:L)

# The η-pairing operator creates a doubly occupied site with a staggered phase. The phase makes η† commute with
# the hopping, so the states ψₙ = (η†)ⁿ|0⟩ are eigenstates with energy Eₙ = n(U - 2μ).
η = sum((-1)^j * c[j, :↑]' * c[j, :↓]' for j in 1:L)

# Applying η† repeatedly to the vacuum gives ψₙ as a sparse sum over the `binomial(L, n)` ways of placing n doublons.
ψs = [Ket("0"^(2L), H) + 0]
for _ in 1:2
    push!(ψs, η * ψs[end])
end

# `ham` acts directly on each symbolic state. We check the residual of the eigenvalue equation and the number of
# basis states in ψₙ.
for (k, ψ) in enumerate(ψs)
    N = k - 1
    E = N * (U - 2μ)
    residual = norm(ham * ψ - E * ψ)
    println("n = $N: E = $E, basis states = $(length(ψ.dict)) = $(binomial(L, N)), residual = $residual")
end

# The residuals vanish exactly, so the states form an equally spaced tower of eigenstates.
# See C. N. Yang, Phys. Rev. Lett. 63, 2144 (1989) and Moudgalya, Regnault, Bernevig, arXiv:2004.13727.
