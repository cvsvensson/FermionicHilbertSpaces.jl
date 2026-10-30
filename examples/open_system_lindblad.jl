# # Open Quantum Systems

# This page shows how to turn a Lindblad master equation into an ordinary matrix problem.
# The density matrix is vectorized on a doubled Hilbert space, so that the Liouvillian
# is a matrix we can diagonalize, exponentiate or restrict to symmetry sectors.
# We apply it to a single fermionic level with incoherent gain and loss, where the
# steady state is known analytically, and then to a double dot coupled to a lossy cavity.
#
# `open_system` builds the doubled space as a tensor product of a "left" and a "right"
# copy of the system space. The right copy is time-reversed: it gives transposed matrix
# representations, `representation(A*B, Hr) == representation(B, Hr) * representation(A, Hr)`,
# which is what a term like ``A\rho B C`` needs, since ``B`` acts on ``\rho`` before ``C``.
# The functions `left` and `right` map expressions in the original basis to the two copies,
# and `reshape` converts between a vectorized state on `Hlr` and a matrix on `(Hl, Hr)`.
#
# The level has energy ``\varepsilon`` and is filled and emptied at rates ``\gamma_\text{in}``
# and ``\gamma_\text{out}``,
# ```math
# H = \varepsilon\, c^\dagger c, \qquad
# L_\text{in} = \sqrt{\gamma_\text{in}}\, c^\dagger, \qquad
# L_\text{out} = \sqrt{\gamma_\text{out}}\, c,
# ```
# so the steady-state occupation is ``\gamma_\text{in}/(\gamma_\text{in} + \gamma_\text{out})``.
# The Liouvillian is ``-i(H_l - H_r)`` plus `dissipator(L)` for each jump operator, where
# `dissipator(L)` is the superoperator ``\rho \mapsto L\rho L^\dagger - \tfrac12\{L^\dagger L, \rho\}``.
# The steady state is the eigenvector with eigenvalue closest to zero. We check that it
# reproduces the analytic occupation and that, starting from the empty level, ``e^{\mathcal{L}t}\rho_0``
# relaxes to it.
using FermionicHilbertSpaces
using LinearAlgebra
import FermionicHilbertSpaces: dissipator
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)

ε, γ_in, γ_out = 1.0, 0.3, 0.7
ham = ε * c[1]'c[1]
lindbladian = -1im * (left(ham) - right(ham)) + dissipator(√γ_in * c[1]') + dissipator(√γ_out * c[1])
L = representation(lindbladian, Hlr, :dense)   # eigen and exp need a dense matrix

Ivec = reshape(I(dim(Hl)), (Hl, Hr) => Hlr)
@assert iszero(L' * Ivec)                      # the Liouvillian preserves the trace

vals, vecs = eigen(L; sortby = abs)
ρ = reshape(vecs[:, 1], Hlr => (Hl, Hr))
ρ ./= tr(ρ)
n = representation(left(c[1]'c[1]), Hl)
@assert tr(n * ρ) ≈ γ_in / (γ_in + γ_out)

ρ0 = reshape(I - n, (Hl, Hr) => Hlr)          # empty level
ρt(t) = reshape(exp(L * t) * ρ0, Hlr => (Hl, Hr))
@assert tr(n * ρt(0.0)) ≈ 0 atol = 1.0e-12
@assert ρt(50.0) ≈ ρ

# The difference in particle number between the left and right copies, ``Q = n_l - n_r``,
# is conserved by this Liouvillian. A constraint on it block-diagonalizes the matrix, and the
# ``Q=0`` block, which holds the populations, contains the steady state.
constraint = NumberConservation(-1:1, [Hl, Hr], [1, -1])
Hcons = tensor_product((Hl, Hr); constraint)
L0 = representation(lindbladian, sector(0, Hcons), :dense)
@assert any(abs(λ) < 1.0e-12 for λ in eigvals(L0))

# `open_system` also accepts a Hilbert space directly. Here one electron is shared between
# two dots that couple to a lossy cavity mode truncated to four photon levels,
# ```math
# H = t\,(f_1^\dagger f_2 + f_2^\dagger f_1) + \omega\, a^\dagger a + g\,(n_1 - n_2)(a + a^\dagger), \qquad
# L = \sqrt{\kappa}\, a.
# ```
# The steady-state photon number follows from the same recipe.
@fermions f
@bosons a
Hsys = tensor_product(hilbert_space(f, 1:2, NumberConservation(1)), hilbert_space(a[1], 4))
Hlr, Hl, Hr, left, right = open_system(Hsys)
t, ω, g, κ = 0.5, 1.0, 0.3, 0.2
ham = t * (f[1]'f[2] + f[2]'f[1]) + ω * a[1]'a[1] + g * (f[1]'f[1] - f[2]'f[2]) * (a[1] + a[1]')
lindbladian = -1im * (left(ham) - right(ham)) + dissipator(√κ * a[1])
vals, vecs = eigen(representation(lindbladian, Hlr, :dense); sortby = abs)
ρ = reshape(vecs[:, 1], Hlr => (Hl, Hr))
ρ ./= tr(ρ)
real(tr(representation(left(a[1]'a[1]), Hl) * ρ))
