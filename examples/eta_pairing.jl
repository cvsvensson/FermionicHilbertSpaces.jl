# # η-pairing in a large Hubbard chain

# Symbolic kets store only their nonzero amplitudes and operators act on them without a basis, so exact
# calculations are possible in spaces far too large for vectors. Here we build a tower of exact eigenstates of a
# 200-site Hubbard chain (a space of dimension 2⁴⁰⁰), the η-pairing states of
# C. N. Yang, Phys. Rev. Lett. 63, 2144 (1989), and verify the eigenvalue equation directly.
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.NonCommutativeProducts: terms

@fermions c
L = 200
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
t, U, μ = 1.0, 4.0, 0.5
n(j, s) = c[j, s]' * c[j, s]
ham = sum(-t * c[j, s]' * c[j + 1, s] + hc for j in 1:(L - 1) for s in (:↑, :↓)) +
    sum(U * n(j, :↑) * n(j, :↓) - μ * (n(j, :↑) + n(j, :↓)) for j in 1:L)

# The η-pairing operator creates a doubly occupied site with a staggered phase, which makes η† commute with the
# hopping. The states ψₙ = (η†)ⁿ|0⟩ are then eigenstates with energy n(U - 2μ), each a sum of `binomial(L, n)`
# basis states.
η = sum((-1)^j * c[j, :↑]' * c[j, :↓]' for j in 1:L)
ψ0 = Ket("0"^(2L), H) + 0
ψs = accumulate((ψ, _) -> η * ψ, 1:2; init = ψ0)
for (N, ψ) in enumerate([ψ0; ψs])
    @assert length(collect(terms(ψ))) == binomial(L, N - 1)
    @assert norm(ham * ψ - (N - 1) * (U - 2μ) * ψ) < 1.0e-10
end
