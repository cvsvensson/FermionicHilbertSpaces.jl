# # Braiding Majorana zero modes

# Exchanging two Majorana zero modes ``γ_i`` and ``γ_j`` acts on the degenerate ground states with the unitary
# ```math
# B_{ij} = \exp\left(\frac{π}{4} γ_i γ_j\right) = \frac{1 + γ_i γ_j}{\sqrt{2}},
# ```
# which maps ``γ_j → γ_i`` and ``γ_i → -γ_j``. The second form uses ``(γ_i γ_j)^2 = -1``, which holds because
# the package normalises Majoranas as ``γ^2 = 1``. Here we braid two Majoranas adiabatically in a tri-junction,
# where a central Majorana ``γ_0`` couples to three outer ones, and check that the time evolution reproduces ``B_{32}``.
using FermionicHilbertSpaces
using LinearAlgebra, Plots
@majoranas γ
γ[1] * γ[1]

# The Majoranas `γ[4]` and `γ[5]` are spectators that never couple to anything. They pair up with the two
# zero modes of the junction so that the ground space within one parity sector is two-dimensional. The
# Hamiltonian conserves parity, so it suffices to work in the even sector.
H = hilbert_space(γ, 0:5, ParityConservation(1))
hs = [representation(1im * γ[0] * γ[i], H, :dense) for i in 1:3]

# The couplings ``Δ_i(t)`` between ``γ_0`` and ``γ_i`` are smooth steps, switched on and off in turn so that at least one
# is always on. While ``γ_0`` couples to ``γ_1`` only, ``γ_2`` and ``γ_3`` are zero modes. Over one period a zero mode
# moves from ``γ_2`` to ``γ_1``, then from ``γ_3`` to ``γ_2`` and finally from ``γ_1`` to ``γ_3``, which exchanges ``γ_2`` and ``γ_3``.
coupling(t, T, k) = (1 + tanh(k * cos(2π * t / T))) / 2
couplings(t, T, k) = [coupling(t - (i - 1) * T / 3, T, k) for i in 1:3]
hamiltonian(t, T, k) = Hermitian(sum(couplings(t, T, k) .* hs))
T, k = 1000, 20
ts = range(0, T, 300)
plot(ts, stack(couplings.(ts, T, k))'; label = ["Δ₁" "Δ₂" "Δ₃"], xlabel = "t", legend = :outerright, frame = :box, size = (450, 250), lw = 2)

# For a slow protocol, the Schrödinger evolution can be approximated by a product of exponentials of the Hamiltonian
# at the midpoints of short time steps.
dt = 1.0
U = foldl((U, t) -> exp(-1im * dt * hamiltonian(t, T, k)) * U, (dt / 2):dt:T; init = Matrix{ComplexF64}(I, dim(H), dim(H)))

# The ground states at ``t = 0`` and ``t = T`` coincide. Restricted to them, the evolution should equal the braid
# operator up to the dynamical phase, which is the same for all ground states.
V = eigvecs(hamiltonian(0, T, k))[:, 1:2]
B = representation((1 + γ[3] * γ[2]) / sqrt(2), H, :dense)
Uground = V' * U * V
Bground = V' * B * V
phase = tr(Uground * Bground') / 2
@assert Uground ≈ phase * Bground
abs(phase)

# The opposite exchange ``B_{23} = B_{32}^†`` is not reproduced, since it differs from ``B_{32}`` by more than a phase.
Binv = V' * B' * V
abs(tr(Uground * Binv') / 2)
