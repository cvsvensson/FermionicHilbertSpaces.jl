# # Self-consistent mean-field theory
# Mean-field theory replaces an interacting Hamiltonian by a quadratic one whose couplings depend on the state, and then
# looks for a state that reproduces itself. Here we solve the Hartree-Fock-Bogoliubov self-consistency of the Hubbard
# model in the BdG space and compare the result with exact diagonalization.
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces: correlation_matrix

# ## The Hubbard ring
# We take a ring of `L` sites with spin. At `μ = U/2` the model is particle-hole symmetric, so the ring is half filled.
@fermions f
L = 6
spins = (:↑, :↓)
labels = [(i, σ) for i in 1:L for σ in spins]
n(i, σ) = f[i, σ]' * f[i, σ]
hopping = sum(-f[i, σ]' * f[mod1(i + 1, L), σ] + hc for i in 1:L, σ in spins)
hubbard(U) = hopping + U * sum(n(i, :↑) * n(i, :↓) for i in 1:L) - U / 2 * sum(n(i, σ) for i in 1:L, σ in spins)

# ## Self-consistency
# The state is a Nambu correlation matrix `G` on the BdG space. `mean_field(ham, G, Hbdg)` decouples the interaction
# in this state into its Hartree, Fock and pairing terms. The result is quadratic, so it has a BdG matrix, whose ground
# state gives the next `G`.
Hbdg = hilbert_space(f, labels, BdG())
ground_state_correlations(op, Hbdg) = correlation_matrix(eigen(representation(op, Hbdg, :dense), BdGEigen()))

# Taking the new state directly can make the iteration oscillate, so we mix it with the previous one.
function self_consistent(ham, G, Hbdg; mixing = 0.5, tol = 1.0e-8, maxiter = 1000)
    for _ in 1:maxiter
        Gnew = ground_state_correlations(mean_field(ham, G, Hbdg), Hbdg)
        norm(Gnew - G) < tol && return Gnew
        G = mixing * Gnew + (1 - mixing) * G
    end
    error("The mean-field iteration did not converge in $maxiter steps.")
end

# ## Antiferromagnet at repulsive U
# The iteration keeps every symmetry of the initial state, so a symmetry-broken solution needs a seed that breaks it.
# We start from the ground state of the hopping in a staggered magnetic field.
staggered = sum((-1)^i * (n(i, :↑) - n(i, :↓)) for i in 1:L)
ham = hubbard(6)
G = self_consistent(ham, ground_state_correlations(hopping + staggered, Hbdg), Hbdg)
# The seed field is gone from the mean-field Hamiltonian, but the moments alternate from site to site and stay large.
[real(expectation((n(i, :↑) - n(i, :↓)) / 2, G, Hbdg)) for i in 1:L]

# ## Superconductor at attractive U
# With attractive interactions we seed with uniform on-site pairing instead. The solution has a uniform gap
# `Δ = U⟨c↓c↑⟩`.
Uattr = -4
pairing = sum(f[i, :↑]' * f[i, :↓]' + hc for i in 1:L)
Gsc = self_consistent(hubbard(Uattr), ground_state_correlations(hopping + pairing, Hbdg), Hbdg)
[real(Uattr * expectation(f[i, :↓] * f[i, :↑], Gsc, Hbdg)) for i in 1:L]

# ## Comparison with exact diagonalization
# The mean-field state is Gaussian, so the energy of the interacting Hamiltonian in it follows from Wick's theorem,
# and the decoupled Hamiltonian has the same energy.
E_mf = real(expectation(ham, G, Hbdg))
@assert expectation(mean_field(ham, G, Hbdg), G, Hbdg) ≈ E_mf
# Since the mean-field state is a variational state, its energy lies above the exact ground-state energy.
H = hilbert_space(f, labels)
E0 = eigmin(Hermitian(representation(ham, H, :dense)))
@assert E_mf > E0
E_mf, E0
