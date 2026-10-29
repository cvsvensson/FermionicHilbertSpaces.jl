# # Self-consistent mean-field theory
# Mean-field theory replaces an interacting Hamiltonian by a quadratic one whose couplings depend on the state, and then
# looks for a state that reproduces itself. Here we solve the Hartree-Fock-Bogoliubov self-consistency of the Hubbard
# model in the BdG space and compare the result with exact diagonalization.
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces: correlation_matrix, BdGEigen

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

function self_consistent(ham, G, Hbdg; tol = 1.0e-8, maxiter = 1000)
    for _ in 1:maxiter
        Gnew = ground_state_correlations(mean_field(ham, G, Hbdg), Hbdg)
        norm(Gnew - G) < tol && return Gnew
        G = Gnew
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
# Since the mean-field state is a variational state, its energy lies above the exact ground-state energy. The exact
# ground state is half filled with as many up as down electrons, so we diagonalize in that sector only.
Hup = hilbert_space(f, [(i, :↑) for i in 1:L])
Hdn = hilbert_space(f, [(i, :↓) for i in 1:L])
Hsector = tensor_product(Hup, Hdn; constraint = NumberConservation(L ÷ 2, Hup) * NumberConservation(L ÷ 2, Hdn))
E0 = eigmin(Hermitian(representation(ham, Hsector, :dense)))
@assert E_mf > E0
E_mf, E0

# ## Phase diagram
# Repeating this for a range of `U` maps out where each ordered phase exists. We sweep from strong coupling towards
# `U = 0` and start each point from the solution at the previous one, so the iteration follows the ordered branch for as
# long as it exists. Energies are measured from the Fermi sea of the hopping, which is the mean-field solution without
# order.
G_fermi_sea = ground_state_correlations(hopping, Hbdg)
function sweep(Us, seed, order_parameter)
    G = ground_state_correlations(hopping + seed, Hbdg)
    return map(Us) do U
        ham = hubbard(U)
        G = self_consistent(ham, G, Hbdg)
        E_sea = real(expectation(ham, G_fermi_sea, Hbdg))
        E0 = eigmin(Hermitian(representation(ham, Hsector, :dense)))
        (order = order_parameter(G), E_mf = real(expectation(ham, G, Hbdg)) - E_sea, E0 = E0 - E_sea)
    end
end
Us = 8:-0.25:0
afm = sweep(Us, staggered, G -> abs(real(expectation(staggered, G, Hbdg))) / 2L)
sc = sweep(-Us, pairing, G -> abs(expectation(f[1, :↓] * f[1, :↑], G, Hbdg)))

using Plots
Uall, both = [-Us; reverse(Us)], [sc; reverse(afm)]
p1 = plot(-Us, getfield.(sc, :order); label = "pairing |⟨c↓c↑⟩|", ylabel = "order parameter", legend = :top)
plot!(p1, Us, getfield.(afm, :order); label = "staggered moment")
p2 = plot(Uall, getfield.(both, :E_mf); label = "mean field", ylabel = "E − E(Fermi sea)", legend = :bottom)
plot!(p2, Uall, getfield.(both, :E0); label = "exact")
plot(p1, p2; layout = (1, 2), xlabel = "U", lw = 2, size = 0.7 .* (1100, 420), frame = :box, margins = 4Plots.mm)
# Each order appears only beyond a threshold coupling and then grows towards saturation. The two sides are mirror
# images: at half filling, a particle-hole transformation of the down spins maps the repulsive ring onto the attractive
# one and turns the staggered moment into the pairing amplitude. The threshold is a finite-size effect. The ring has a
# gap at the Fermi level, which the interaction must overcome before ordering lowers the energy, so below it the
# mean-field state is just the Fermi sea. The exact ground state, which on a finite ring breaks no symmetry, gains energy
# from correlations at any `U`. Mean field stays above it everywhere, as a variational state must, misses the whole
# energy gain at weak coupling, and captures most of it at strong coupling.
