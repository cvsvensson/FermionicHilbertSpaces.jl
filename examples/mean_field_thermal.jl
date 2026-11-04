# # Mean-field theory at finite temperature
# Hamiltonians are symbolic operators, so `mean_field(ham, G, Hbdg)` derives the decoupled Hartree-Fock-Bogoliubov
# Hamiltonian directly from the interacting `ham`, given the Nambu correlation matrix `G` of a Gaussian state. At finite
# temperature the next `G` is the thermal state of that Hamiltonian, which `correlation_matrix(F; β)` gives from its BdG
# eigendecomposition. We follow the superconducting order of an attractive Hubbard ring as the temperature rises.

using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions

# The attractive Hubbard ring is half filled at `μ = U/2`.
@fermions f
L = 6
U = -4
spins = (:↑, :↓)
n(i, σ) = f[i, σ]' * f[i, σ]
hopping = sum(-f[i, σ]' * f[mod1(i + 1, L), σ] + hc for i in 1:L, σ in spins)
ham = hopping + U * sum(n(i, :↑) * n(i, :↓) for i in 1:L) - U / 2 * sum(n(i, σ) for i in 1:L, σ in spins);

# Close to the transition the iteration converges slowly, so it stops after `maxiter` steps.
Hbdg = hilbert_space(f, [(i, σ) for i in 1:L for σ in spins], BdG())
thermal_correlations(op, β) = correlation_matrix(eigen(representation(op, Hbdg, Matrix), BdGEigen()); β)
function self_consistent(G, β; tol = 1.0e-6, maxiter = 200)
    for _ in 1:maxiter
        Gnew = thermal_correlations(mean_field(ham, G, Hbdg), β)
        norm(Gnew - G) < tol && return Gnew
        G = Gnew
    end
    return G
end;

# We seed the iteration with on-site pairing and raise the temperature, starting each point from the previous solution.
# The order parameter is the pairing amplitude, the expectation value of `f[i, :↓] * f[i, :↑]`.
pairing = sum(f[i, :↑]' * f[i, :↓]' + hc for i in 1:L)
Ts = 0.1:0.05:1.2
G = thermal_correlations(hopping + pairing, 1 / first(Ts))
amplitude = map(Ts) do T
    global G = self_consistent(G, 1 / T)
    abs(expectation(f[1, :↓] * f[1, :↑], G, Hbdg))
end;

using Plots
plot(Ts, amplitude; xlabel = "temperature", ylabel = "pairing amplitude", legend = false, lw = 2, frame = :box)

# The pairing amplitude shrinks as the temperature rises and vanishes at a critical temperature, above which the
# thermal state of the free ring is self-consistent. Mean field gives this sharp transition even on a small ring because
# it ignores fluctuations, which in a finite system smooth the transition into a crossover.
