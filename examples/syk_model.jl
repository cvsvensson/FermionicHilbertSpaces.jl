# # Sachdev-Ye-Kitaev model
#
# The SYK model of ``M`` Majoranas with random all-to-all four-body couplings is chaotic, and its
# level statistics follow random-matrix theory. Which ensemble depends on ``M \bmod 8``: within one
# fermion-parity sector it is GOE for ``M \equiv 0``, GUE for ``M \equiv 2, 6`` and GSE for
# ``M \equiv 4``, where every level is doubly degenerate.[^1]
# This page reproduces the three classes from the mean ratio of adjacent level spacings. On the
# package side it shows how to build a dense random Majorana Hamiltonian symbolically and restrict
# it to a parity sector, which is where the statistics must be taken.
#
# `@majoranas` creates operators with ``\{\gamma_i, \gamma_j\} = 2\delta_{ij}``, and `add!!`
# accumulates the ``\binom{M}{4}`` terms in place. The coupling scale is left out, since the
# spacing ratio does not depend on it.
using FermionicHilbertSpaces
import FermionicHilbertSpaces: add!!
using LinearAlgebra, Random, Statistics

@majoranas γ
function syk_hamiltonian(M)
    ham = 0
    for i in 1:M-3, j in i+1:M-2, k in j+1:M-1, l in k+1:M
        ham = add!!(ham, randn() * γ[i] * γ[j] * γ[k] * γ[l])
    end
    return ham
end

function gap_ratio(energies)
    s = diff(energies)
    return mean(min(a, b) / max(a, b) for (a, b) in zip(s[1:(end - 1)], s[2:end]))
end;

# The statistics must be taken within one fermion-parity sector: over the full space the two
# sectors are interleaved, and for ``M \equiv 2 mod 8`` the odd sector is an exact copy of the
# even one, so every level is degenerate and the ratio is zero whatever the dynamics.
# We therefore diagonalize the even sector for a few disorder samples, keeping one level of each
# degenerate pair for GSE. The table lists the measured mean ratio next to the random-matrix
# value of the expected class (Atas *et al.*[^2]); they should agree.
Random.seed!(1)
cases = [(M = 16, class = "GOE", ref = 0.5307), (M = 18, class = "GUE", ref = 0.5996), (M = 20, class = "GSE", ref = 0.6744)]
results = map(cases) do (; M, class, ref)
    Heven = hilbert_space(γ, 1:M, ParityConservation(1))
    r = mean(1:20) do _
        E = eigvals(Hermitian(representation(syk_hamiltonian(M), Heven, :dense)))
        gap_ratio(class == "GSE" ? E[1:2:end] : E)
    end
    (majoranas = M, expected_class = class, measured_ratio = round(r; digits = 3), random_matrix_ratio = ref)
end
@assert all(abs(c.measured_ratio - c.random_matrix_ratio) < 0.02 for c in results)
results

# [^1]: Y.-Z. You, A. W. W. Ludwig and C. Xu, "Sachdev-Ye-Kitaev model and thermalization on the boundary of many-body localized fermionic symmetry protected topological states", [Phys. Rev. B 95, 115150 (2017)](https://doi.org/10.1103/PhysRevB.95.115150), [arXiv:1602.06964](https://arxiv.org/abs/1602.06964).
# [^2]: Y. Y. Atas, E. Bogomolny, O. Giraud and G. Roux, "Distribution of the ratio of consecutive level spacings in random matrix ensembles", [Phys. Rev. Lett. 110, 084101 (2013)](https://doi.org/10.1103/PhysRevLett.110.084101), [arXiv:1212.5611](https://arxiv.org/abs/1212.5611).
