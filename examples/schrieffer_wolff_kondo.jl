# # Schrieffer–Wolff Transformation of the Anderson Model

# This tutorial uses KrylovKit's `linsolve` on symbolic kets to compute the second-order effective Hamiltonian of
# an Anderson impurity coupled to a bath, and checks it against the Schrieffer–Wolff Kondo coupling
# ``J\,\mathbf{S}\cdot\mathbf{s}``. It assumes [Symbolic states](../symbolic_states.md).
using FermionicHilbertSpaces, KrylovKit, LinearAlgebra
import KrylovKit.VectorInterface: inner

# The impurity `d` has repulsion `U` and hybridizes with strength `V` to bath levels `c`, whose energies span [-1, 1] with the middle one at the Fermi level.
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

# The low-energy states have one electron on the impurity, one at the Fermi level, and all levels below it filled.
vacuum = Ket("0"^(2N + 2), H)
fermi_sea = prod(c[k, σ]' for k in 1:(kF - 1) for σ in spins) * vacuum
low_energy = [d[σ]' * c[kF, τ]' * fermi_sea for σ in spins for τ in spins]
E0 = εd + 2sum(εk[1:(kF - 1)])

# The second-order effective Hamiltonian is ``H_\mathrm{eff} = P V (E_0 - H_0)^{-1} V P``.
# `Vhyb * p` has no overlap with the low-energy states, so the inverse is well defined.
# For each low-energy state `p` we solve ``(E_0 - H_0) x = V|p⟩`` and take overlaps with ``V|p'⟩``.
function effective_hamiltonian(basis, H0, V, E0)
    xs = map(basis) do p
        x, info = linsolve(x -> E0 * x - H0 * x, V * p)
        x
    end
    return [inner(V * p, x) for p in basis, x in xs]
end
Heff = effective_hamiltonian(low_energy, H0, Vhyb, E0)

# The four states split into a singlet and a triplet. Their splitting is the Kondo coupling
# ``J = 2V^2 \left(1/|\varepsilon_d| + 1/(\varepsilon_d + U)\right)``, which we compare with the numerical splitting.
λ = eigvals(Hermitian(Heff))
J_SW = 2V^2 * (1 / abs(εd) + 1 / (εd + U))
(splitting = λ[2] - λ[1], J_SW = J_SW)

# The spin-flip element between ``d_\uparrow^\dagger c_{k_F\downarrow}^\dagger`` and ``d_\downarrow^\dagger c_{k_F\uparrow}^\dagger`` is ``J/2``, as for ``J\,\mathbf{S}\cdot\mathbf{s}``.
(Heff[2, 3], J_SW / 2)

# References: J. R. Schrieffer and P. A. Wolff, Phys. Rev. 149, 491 (1966); S. Bravyi, D. P. DiVincenzo, and D. Loss, Ann. Phys. 326, 2793 (2011), arXiv:1105.0675.
