# # Bound doublons in the Hubbard ring

# KrylovKit's `eigsolve` accepts a symbolic Hamiltonian as the linear map and symbolic kets as vectors, so
# eigenstates can be found without ever building a matrix. We use this to find the bound pair (doublon) of two
# fermions on an attractive Hubbard ring and compare its dispersion with the exact result.
using FermionicHilbertSpaces, KrylovKit, Plots

@fermions c
L = 16
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
t, U = 1.0, -4.0
ham = sum(-t * c[j, s]' * c[mod1(j + 1, L), s] + hc for j in 1:L for s in (:↑, :↓)) +
    U * sum(c[j, :↑]' * c[j, :↑] * c[j, :↓]' * c[j, :↓] for j in 1:L);

# A doublon with momentum `K` is a superposition of a spin-up and a spin-down fermion on the same site. The Krylov
# space generated from it stays in the two-particle sector with momentum `K`, so the lowest eigenvalue is the bound
# state. `eager = true` stops as soon as it has converged, before rounding errors pull in other momenta, and a loose
# tolerance is enough since the energies are only plotted.
vac = Ket("0"^(2L), H)
doublon(K) = sum(cis(K * j) * c[j, :↑]' * c[j, :↓]' * vac for j in 1:L) / sqrt(L)
bound_energy(K) = eigsolve(ham, doublon(K), 1, :SR; ishermitian = true, eager = true, tol = 1.0e-6)[1][1];

# On an infinite chain the bound state energy is `E(K) = -√(U² + 16t² cos²(K/2))`[^1]; on the ring the deviation is a
# finite-size effect.
Eexact(K) = -sqrt(U^2 + 16t^2 * cos(K / 2)^2)
Ks = [2π * m / L for m in 0:(L ÷ 2)]
Es = bound_energy.(Ks)
plot(Eexact, 0, π; label = "exact", xlabel = "K", ylabel = "E", lw = 2, frame = :box, size = (500, 300))
scatter!(Ks, Es; label = "eigsolve, L = $L")

# [^1]: F. H. L. Essler et al., *The One-Dimensional Hubbard Model* (Cambridge, 2005).
