# # Schrieffer–Wolff transformation of the Anderson model

# `linsolve` from KrylovKit works on symbolic kets, so second-order perturbation theory can be done exactly in a
# space far too large for matrices. We compute the effective Hamiltonian of an Anderson impurity coupled to 81 bath
# levels in its low-energy subspace and recover the Kondo exchange ``J\,\mathbf{S}\cdot\mathbf{s}`` of
# Schrieffer and Wolff, Phys. Rev. 149, 491 (1966).
using FermionicHilbertSpaces, KrylovKit, LinearAlgebra
import KrylovKit.VectorInterface: inner

@fermions d c
N = 81
εd, U, V = -1.0, 3.0, 0.05
εk = range(-1, 1, length = N)
kF = (N + 1) ÷ 2
spins = [:↑, :↓]
H = tensor_product(hilbert_space(d, spins), hilbert_space(c, [(k, σ) for k in 1:N for σ in spins]))
H0 = sum(εd * d[σ]' * d[σ] for σ in spins) + U * d[:↑]' * d[:↑] * d[:↓]' * d[:↓] +
    sum(εk[k] * c[k, σ]' * c[k, σ] for k in 1:N for σ in spins)
Vhyb = sum(V * d[σ]' * c[k, σ] + hc for k in 1:N for σ in spins)

# The low-energy states have one electron on the impurity, one at the Fermi level and all levels below filled.
# The effective Hamiltonian is ``H_\mathrm{eff} = P V (E_0 - H_0)^{-1} V P``; for each low-energy state `p` we solve
# ``(E_0 - H_0) x = V|p⟩`` and take overlaps with ``V|p'⟩``. Its eigenvalues split into a singlet and a triplet, and the
# splitting is the Kondo coupling ``J = 2V^2 \left(1/|\varepsilon_d| + 1/(\varepsilon_d + U)\right)``. The spin-flip
# element between ``d_\uparrow^\dagger c_{k_F\downarrow}^\dagger`` and ``d_\downarrow^\dagger c_{k_F\uparrow}^\dagger`` is ``J/2``.
vacuum = Ket("0"^(2N + 2), H)
fermi_sea = prod(c[k, σ]' for k in 1:(kF - 1) for σ in spins) * vacuum
low_energy = [d[σ]' * c[kF, τ]' * fermi_sea for σ in spins for τ in spins]
E0 = εd + 2sum(εk[1:(kF - 1)])

xs = [linsolve(x -> E0 * x - H0 * x, Vhyb * p)[1] for p in low_energy]
Heff = [inner(Vhyb * p, x) for p in low_energy, x in xs]
λ = eigvals(Hermitian(Heff))
J = 2V^2 * (1 / abs(εd) + 1 / (εd + U))
@assert (λ[2] - λ[1]) ≈ J rtol = 1.0e-3
@assert Heff[2, 3] ≈ J / 2 rtol = 1.0e-3
(splitting = λ[2] - λ[1], J)

# See also S. Bravyi, D. P. DiVincenzo, and D. Loss, Ann. Phys. 326, 2793 (2011), arXiv:1105.0675.
