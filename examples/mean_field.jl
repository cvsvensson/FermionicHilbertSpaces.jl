# # Self-consistent mean-field theory
# Mean-field theory replaces an interacting Hamiltonian by a quadratic one whose couplings depend on the state, and then
# looks for a state that reproduces itself. Here we build the Hartree-Fock-Bogoliubov decoupling of the Hubbard model
# directly from the symbolic Hamiltonian, solve the self-consistency in the BdG space, and compare the result with
# exact diagonalization.
using FermionicHilbertSpaces, LinearAlgebra
import NonCommutativeProducts as NCP

# ## The Hubbard ring
# We take a ring of `L` sites with spin. At `μ = U/2` the model is particle-hole symmetric, so the ring is half filled.
@fermions f
L = 4
spins = (:↑, :↓)
labels = [(i, σ) for i in 1:L for σ in spins]
n(i, σ) = f[i, σ]' * f[i, σ]
hopping = sum(-f[i, σ]' * f[mod1(i + 1, L), σ] + hc for i in 1:L, σ in spins)
hubbard(U) = hopping + U * sum(n(i, :↑) * n(i, :↓) for i in 1:L) - U / 2 * sum(n(i, σ) for i in 1:L, σ in spins)

# ## Decoupling
# Products of fermions are kept normal ordered, so every interaction term is a product `abcd` of four operators
# with the creation operators first. In a Gaussian state, its expectation value is given by Wick's theorem as a sum over
# the three ways to pair the factors. Linearizing this around the state replaces each pair in turn by its contraction
# `ctr(a, b) = ⟨ab⟩`. The constant term makes sure that the quadratic operator has the same
# expectation value as `abcd`. Quadratic terms are already of mean-field form and pass through unchanged.
function decouple(term, ctr)
    length(NCP.factors(term)) == 4 || return term
    a, b, c, d = NCP.factors(term)
    return NCP.coefficient(term) * (
        ctr(a, b) * c * d + a * b * ctr(c, d) -
            ctr(a, c) * b * d - a * c * ctr(b, d) +
            ctr(a, d) * b * c + a * d * ctr(b, c) -
            (ctr(a, b) * ctr(c, d) - ctr(a, c) * ctr(b, d) + ctr(a, d) * ctr(b, c))
    )
end
mean_field(op, ctr) = mapreduce(term -> decouple(term, ctr), NCP.add!!, NCP.terms(op); init = 0)

# The contractions can be anything that multiplies operators, for example symbolic variables. On a single site,
# keeping the densities and the pairing amplitude `κ = ⟨c↑c↓⟩`, this gives the familiar Hartree and BCS terms.
using Symbolics
@variables U n_up n_dn κ κ̄
up, dn = f[1, :↑], f[1, :↓]
contractions = Dict((up', up) => n_up, (dn', dn) => n_dn, (up, dn) => κ, (up', dn') => κ̄)
mean_field(U * n(1, :↑) * n(1, :↓), (a, b) -> get(contractions, (a, b), 0))

# ## Self-consistency
# The state is a Nambu correlation matrix `G` on the BdG space, so the contractions are expectation values of
# quadratic operators. The mean-field Hamiltonian then has a BdG matrix, whose ground state gives the next `G`.
Hbdg = hilbert_space(f, labels, BdG())
mean_field(op, G, Hbdg) = mean_field(op, (a, b) -> expectation(a * b, G, Hbdg))
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
# The mean-field state is a Gaussian state on the full Fock space, so `gaussian_state` gives its density matrix
# and the energy of the interacting Hamiltonian in it can be evaluated exactly.
H = hilbert_space(f, labels)
ρ = gaussian_state(G, H, Hbdg)
E_mf = real(expectation(ham, ρ, H))
# By construction of the constant term, the decoupled Hamiltonian has the same energy in this state.
@assert expectation(mean_field(ham, G, Hbdg), G, Hbdg) ≈ E_mf
# Since the mean-field state is a variational state, its energy lies above the exact ground-state energy.
E0 = eigmin(Hermitian(representation(ham, H, :dense)))
@assert E_mf > E0
E_mf, E0
