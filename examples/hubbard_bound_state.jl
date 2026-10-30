# # Bound doublons in the Hubbard ring

# This tutorial uses KrylovKit's `eigsolve` with a symbolic Hamiltonian as the linear map and symbolic kets as vectors,
# to find the bound pair (doublon) of two fermions on an attractive Hubbard ring.
using FermionicHilbertSpaces, KrylovKit, Plots
using FermionicHilbertSpaces.NonCommutativeProducts: terms

@fermions c
L = 16
H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
dim(H)

# The Hubbard Hamiltonian with periodic boundary conditions.
t, U = 1.0, -4.0
ham = sum(-t * c[j, s]' * c[mod1(j + 1, L), s] + hc for j in 1:L for s in (:↑, :↓)) +
    U * sum(c[j, :↑]' * c[j, :↑] * c[j, :↓]' * c[j, :↓] for j in 1:L)

# A doublon with momentum `K` is a superposition of a spin-up and a spin-down fermion sitting on the same site.
vac = Ket("0"^(2L), H)
doublon(K) = sum(cis(K * j) * c[j, :↑]' * c[j, :↓]' * vac for j in 1:L) / sqrt(L)

# The Krylov space generated from `doublon(K)` stays in the two-particle sector with momentum `K`, and its lowest
# eigenvalue is the bound state. `eager = true` stops as soon as the eigenvalue has converged, before rounding errors
# pull in other momenta.
function bound_state(K)
    vals, vecs, info = eigsolve(ham, doublon(K), 1, :SR; ishermitian = true, eager = true)
    return vals[1], vecs[1]
end
E0, ψ0 = bound_state(0.0)
# The eigenvector is supported on a small fraction of the basis states of `H`.
(; hilbert_dim = dim(H), support = length(collect(terms(ψ0))))

# On an infinite chain the bound state energy is `E(K) = -√(U² + 16t² cos²(K/2))`. On the ring `K = 2πm/L`, and the
# deviation from this formula is a finite-size effect that decays exponentially with `L`.
Eexact(K) = -sqrt(U^2 + 16t^2 * cos(K / 2)^2)
Ks = [2π * m / L for m in 0:(L ÷ 2)]
Es = [first(bound_state(K)) for K in Ks]
maximum(abs, Es .- Eexact.(Ks))
# The eigenvalues follow the exact dispersion.
plot(Eexact, 0, π; label = "exact", xlabel = "K", ylabel = "E", lw = 2, frame = :box, size = (500, 300))
scatter!(Ks, Es; label = "eigsolve, L = $L")

# See e.g. F. H. L. Essler et al., *The One-Dimensional Hubbard Model* (Cambridge, 2005) for the two-particle problem.
