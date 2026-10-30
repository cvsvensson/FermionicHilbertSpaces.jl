# # Self-consistent mean-field theory
# `mean_field(ham, G, Hbdg)` decouples an interacting operator in a Gaussian state with Nambu correlation matrix `G`.
# Iterating it to self-consistency gives Hartree-Fock-Bogoliubov theory; we find an antiferromagnet at repulsive and a
# superconductor at attractive Hubbard interaction, and compare with exact diagonalization.

using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions

# The Hubbard ring has `L` sites with spin and is half filled at `μ = U/2`.
@fermions f
L = 6
spins = (:↑, :↓)
labels = [(i, σ) for i in 1:L for σ in spins]
n(i, σ) = f[i, σ]' * f[i, σ]
hopping = sum(-f[i, σ]' * f[mod1(i + 1, L), σ] + hc for i in 1:L, σ in spins)
hubbard(U) = hopping + U * sum(n(i, :↑) * n(i, :↓) for i in 1:L) - U / 2 * sum(n(i, σ) for i in 1:L, σ in spins);

# The decoupled Hamiltonian is quadratic, so its BdG ground state gives the next `G`.
Hbdg = hilbert_space(f, labels, BdG())
ground_state_correlations(op, Hbdg) = correlation_matrix(eigen(representation(op, Hbdg, :dense), BdGEigen()));

function self_consistent(ham, G, Hbdg; tol = 1.0e-8, maxiter = 1000)
    for _ in 1:maxiter
        Gnew = ground_state_correlations(mean_field(ham, G, Hbdg), Hbdg)
        norm(Gnew - G) < tol && return Gnew
        G = Gnew
    end
    error("The mean-field iteration did not converge in $maxiter steps.")
end;

# The iteration preserves the symmetries of its seed, so we seed with a staggered field (repulsive `U`) or on-site
# pairing (attractive `U`). The decoupled Hamiltonian reproduces the interacting energy in the Gaussian state.
staggered = sum((-1)^i * (n(i, :↑) - n(i, :↓)) for i in 1:L)
pairing = sum(f[i, :↑]' * f[i, :↓]' + hc for i in 1:L)
G = self_consistent(hubbard(6), ground_state_correlations(hopping + staggered, Hbdg), Hbdg)
@assert expectation(mean_field(hubbard(6), G, Hbdg), G, Hbdg) ≈ expectation(hubbard(6), G, Hbdg)

# We sweep `U` toward zero, starting each point from the previous solution, and compare with the exact ground state in
# the half-filled sector. Energies are measured from the Fermi sea of the hopping.
Hup = hilbert_space(f, [(i, :↑) for i in 1:L])
Hdn = hilbert_space(f, [(i, :↓) for i in 1:L])
Hsector = tensor_product(Hup, Hdn; constraint = NumberConservation(L ÷ 2, Hup) * NumberConservation(L ÷ 2, Hdn))
G_fermi_sea = ground_state_correlations(hopping, Hbdg)
function sweep(Us, seed, order_parameter)
    G = ground_state_correlations(hopping + seed, Hbdg)
    return map(Us) do U
        ham = hubbard(U)
        G = self_consistent(ham, G, Hbdg)
        E_sea = real(expectation(ham, G_fermi_sea, Hbdg))
        E_exact = eigmin(Hermitian(representation(ham, Hsector, :dense)))
        E_mf = real(expectation(ham, G, Hbdg))
        @assert E_mf > E_exact
        (order = order_parameter(G), E_mf = E_mf - E_sea, E_exact = E_exact - E_sea)
    end
end;
Us = 8:-0.5:0
afm = sweep(Us, staggered, G -> abs(real(expectation(staggered, G, Hbdg))) / 2L)
sc = sweep(-Us, pairing, G -> abs(expectation(f[1, :↓] * f[1, :↑], G, Hbdg)));

using Plots
Uall, both = [-Us; reverse(Us)], [sc; reverse(afm)]
p1 = plot(-Us, getfield.(sc, :order); label = "pairing |⟨c↓c↑⟩|", ylabel = "order parameter", legend = :top)
plot!(p1, Us, getfield.(afm, :order); label = "staggered moment")
p2 = plot(Uall, getfield.(both, :E_mf); label = "mean field", ylabel = "E − E(Fermi sea)", legend = :bottom)
plot!(p2, Uall, getfield.(both, :E_exact); label = "exact")
plot(p1, p2; layout = (1, 2), xlabel = "U", lw = 2, size = 0.7 .* (1100, 420), frame = :box, margins = 4Plots.mm)

# Each order sets in beyond a threshold coupling, which is a finite-size effect from the gap at the Fermi level. The two
# sides mirror each other by a particle-hole transformation of the down spins. Mean field stays above the exact energy, as
# a variational state must.
