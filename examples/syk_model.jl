# # Sachdev-Ye-Kitaev model
#
# It assumes [Getting started](../index.md).
#
# We build the SYK model of ``M`` Majoranas with random all-to-all four-body couplings,
# diagonalize it in one parity sector, and use the level-spacing ratio to detect chaos.
# `@majoranas` creates operators with ``\gamma_i^2 = 1``, ``\{\gamma_i, \gamma_j\} = 2\delta_{ij}``.
using FermionicHilbertSpaces
import FermionicHilbertSpaces: add!!
using LinearAlgebra, Random, Statistics

# The Hamiltonian is ``H = \sqrt{6}\,J/(4M^{3/2}) \sum_{i<j<k<l} J_{ijkl}\gamma_i\gamma_j\gamma_k\gamma_l``
# with Gaussian ``J_{ijkl}``. `add!!` accumulates the terms in place.
function syk_hamiltonian(g, M; J = 1.0)
    prefactor = sqrt(6) * J / (4 * M^(3 / 2))
    ham = 0
    for i in 1:M-3, j in i+1:M-2, k in j+1:M-1, l in k+1:M
        ham = add!!(ham, prefactor * randn() * g[i] * g[j] * g[k] * g[l])
    end
    return ham
end

# Fix the seed and restrict to the even-parity sector.
M = 16
@majoranas g
H = hilbert_space(g, 1:M, ParityConservation(1))
Random.seed!(11)
ham = Hermitian(representation(syk_hamiltonian(g, M), H, :dense))
energies = eigvals(ham);

# The mean ratio ``\min(s_n, s_{n+1}) / \max(s_n, s_{n+1})`` of adjacent level spacings
# is independent of the level density.
function adjacent_gap_ratio(energies)
    s = diff(energies)
    return mean(min(a, b) / max(a, b) for (a, b) in zip(s[1:(end - 1)], s[2:end]))
end
adjacent_gap_ratio(energies)

# For ``M`` a multiple of 8 the even sector is in the GOE class, with reference value
# ``\approx 0.5307``. ``M \equiv 2, 6 \bmod 8`` gives GUE (``\approx 0.5996``) and
# ``M \equiv 4 \bmod 8`` gives GSE (``\approx 0.6744``), where every level is doubly degenerate:
# keep one of each pair, e.g. `energies[1:2:end]`, before computing the ratio
# (You, Ludwig, Xu, PRB 95, 115150 (2017); reference values from Atas et al., PRL 110, 084101 (2013)).
# A single sample fluctuates around it; averaging over seeds converges.
mean(adjacent_gap_ratio(eigvals(Hermitian(representation(syk_hamiltonian(g, M), H, :dense)))) for _ in 1:10)
