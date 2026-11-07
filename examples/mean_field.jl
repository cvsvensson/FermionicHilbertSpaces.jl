# # Self-consistent mean-field theory
# Hamiltonians are symbolic operators, so `mean_field(ham, G, Hbdg)` derives the decoupled Hartree-Fock-Bogoliubov
# Hamiltonian directly from the interacting `ham`, given the Nambu correlation matrix `G` of a Gaussian state. No
# mean-field equations are derived by hand. `expectation(ham, G, Hbdg)` gives the interacting energy in that state. We
# use both to find the magnetic and superconducting states of a Hubbard ring and compare with exact diagonalization.

using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions

# The Hubbard ring is half filled at `μ = U/2`.
@fermions f
L = 6
t = 1
spins = (:↑, :↓)
n(i, σ) = f[i, σ]' * f[i, σ]
hopping = sum(-t * f[i, σ]' * f[mod1(i + 1, L), σ] + hc for i in 1:L, σ in spins)
hubbard(U) = hopping + U * sum(n(i, :↑) * n(i, :↓) for i in 1:L) - U / 2 * sum(n(i, σ) for i in 1:L, σ in spins);

# Each step takes the BdG ground state of the decoupled Hamiltonian as the next `G`.
Hbdg = hilbert_space(f, [(i, σ) for i in 1:L for σ in spins], BdG())
ground_state_correlations(op) = correlation_matrix(eigen(representation(op, Hbdg, Matrix), BdGEigen()))
function self_consistent(ham, G; tol = 1.0e-8, maxiter = 1000)
    for _ in 1:maxiter
        Gnew = ground_state_correlations(mean_field(ham, G, Hbdg))
        norm(Gnew - G) < tol && return Gnew
        G = Gnew
    end
    error("The mean-field iteration did not converge in $maxiter steps.")
end;

# The iteration keeps the symmetries of its starting state, so we seed it with the order we look for: alternating
# spins for repulsive `U` and on-site pairing for attractive `U`. We sweep `U` toward zero, starting each point from the
# previous solution, and measure energies from the Fermi sea. The exact energy comes from the half-filled sector.
staggered = sum((-1)^i * (n(i, :↑) - n(i, :↓)) for i in 1:L)
pairing = sum(f[i, :↑]' * f[i, :↓]' + hc for i in 1:L)
Hup = hilbert_space(f, [(i, :↑) for i in 1:L])
Hdn = hilbert_space(f, [(i, :↓) for i in 1:L])
Hsector = tensor_product(Hup, Hdn; constraint = NumberConservation(L ÷ 2, Hup) * NumberConservation(L ÷ 2, Hdn))
G_sea = ground_state_correlations(hopping)
function sweep(Us, seed)
    G = ground_state_correlations(hopping + seed)
    return map(Us) do U
        ham = hubbard(U)
        G = self_consistent(ham, G)
        E_sea = real(expectation(ham, G_sea, Hbdg))
        E_mf = real(expectation(ham, G, Hbdg)) - E_sea
        E_exact = eigmin(Hermitian(representation(ham, Hsector, Matrix))) - E_sea
        (; E_mf, E_exact)
    end
end;
Us = t .* (8:-0.5:0)
results = [sweep(-Us, pairing); reverse(sweep(Us, staggered))];

using Plots
Uall = [-Us; reverse(Us)]
plot(Uall ./ t, getproperty.(results, :E_mf) ./ t; label = "mean field", xlabel = "U/t", ylabel = "(E − E(Fermi sea))/t", lw = 2, frame = :box)
plot!(Uall ./ t, getproperty.(results, :E_exact) ./ t; label = "exact", lw = 2)

# Mean field finds a superconductor for attractive and a magnet for repulsive `U`, but only beyond a threshold. For weak
# coupling the best Gaussian state is the Fermi sea, and the mean-field energy stays at zero. The exact energy is lower
# for any nonzero `U`, as it must be for a variational method, and mean field captures most of it at strong coupling.
