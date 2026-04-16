# # Bosonic HEOM: Symbolic Representation and Matrix Assembly
#
# This tutorial shows how to build the bosonic Hierarchical Equations of Motion (HEOM)
# Liouvillian superoperator symbolically using FermionicHilbertSpaces.jl.
#
# We reuse two key patterns from the package:
# 1. The **doubled Hilbert space** trick from `open_system_lindblad.jl`:
#    left/right copies of the system operators encode the superoperator.
# 2. A **custom auxiliary algebra** (modelled on `floquet_tutorial.jl`):
#    new symbolic operator types that act on the ADO index space.
#
# ## Background
#
# The bosonic HEOM (arXiv 2306.07522, Eq. 16) propagates an infinite hierarchy of
# auxiliary density operators (ADOs) ``\rho_{\mathbf{j}}^{(m)}(t)``, where ``\mathbf{j}``
# is a multiset of exponential-expansion indices.  Truncating at tier ``m_{\max}``
# and arranging all ADOs into one large vector ``|\boldsymbol{\rho}\rangle\rangle``,
# the equation of motion becomes
# ```math
# \frac{d}{dt}|\boldsymbol{\rho}\rangle\rangle = \mathbf{M}\,|\boldsymbol{\rho}\rangle\rangle,
# ```
# where the **HEOM Liouvillian** ``\mathbf{M}`` is
# ```math
# \mathbf{M} = i(H_l - H_r) \otimes \mathbf{1}_{\text{aux}}
#   + \mathbf{1}_{\text{sys}} \otimes W_{\text{aux}}
#   - i \sum_l (V_l - V_r) \otimes A_l^+
#   - i \sum_l (\xi_l V_l - \xi_l^* V_r) \otimes A_l^-.
# ```
#
# Here ``H_{l/r}`` and ``V_{l/r}`` are the Hamiltonian and bath-coupling operator
# applied to the *left* (ket) or *right* (bra) copy of the system, respectively,
# and the auxiliary operators act on the ADO-index space:
# - ``A_l^+ |n\rangle = |n + e_l\rangle`` (zero if total tier ``= m_{\max}``)
# - ``A_l^- |n\rangle = n_l |n - e_l\rangle``
# - ``W |n\rangle = (-\sum_l \chi_l n_l) |n\rangle``
#
# The bath correlation function is decomposed as
# ```math
# C_\beta(\tau) = \sum_{l=1}^{N_{\text{exp}}} \xi_l \, e^{-\chi_l \tau},
# ```
# with (complex) amplitudes ``\xi_l`` and decay rates ``\chi_l``.
# For the **Drude-Lorentz spectral density**
# ``J_\beta(\omega) = 4\Delta W\omega/(\omega^2 + W^2)``
# and one Padé term (``N_\beta = 1``), the exponent parameters are
# (Eqs. 31–32 of arXiv 2306.07522):
# ```math
# \xi_1 = \Delta W\!\left[-i + \cot\!\left(\tfrac{W}{2k_BT}\right)\right],
# \qquad \chi_1 = W.
# ```

using FermionicHilbertSpaces
using FermionicHilbertSpaces: HEOMBosonicBath, heom_bosonic_aux_space, heom_generator, heom_aux_dim
using LinearAlgebra

# ## Physical setup
#
# **System**: a two-level system (spin-1/2) with splitting ``\omega``.
# **Bath**: a Drude-Lorentz bosonic reservoir coupled via ``V_s = \sigma_z``.
#
# Hamiltonian and coupling operator:
# ```math
# H_s = \tfrac{\omega}{2}\,\sigma_z,\qquad V_s = \sigma_z.
# ```
#
# Parameters

ω = 1.0          # TLS level splitting
Δ = 0.5          # coupling strength (system–bath)
W = 2.0          # bath bandwidth (Drude-Lorentz cutoff)
kBT = 0.5         # thermal energy k_B T

# Drude-Lorentz correlation function coefficients (1 Padé term)
ξ1 = Δ * W * (-im + cot(W / (2 * kBT)))
χ1 = W + 0.0im
m_max = 2          # maximum hierarchy depth

bath = HEOMBosonicBath(1, 1, m_max, [χ1], [ξ1])
println("Bath: ", bath)
println("ξ₁ = ", round(ξ1, digits=4), "  χ₁ = ", χ1)

# ## Doubled Hilbert space
#
# Two independent spin groups represent the left (ket) and right (bra) copies
# of the density matrix.  Operators from different groups *commute*,
# so we can freely mix them — exactly as in the Lindblad superoperator tutorial.

@spin σ_l 1 // 2 # left and right spins 
@spin σ_r 1 // 2

H_l = ω / 2 * σ_l[:z]
H_r = ω / 2 * σ_r[:z]
V_l = σ_l[:z]
V_r = σ_r[:z]

# ## Symbolic HEOM generator
#
# `heom_generator` assembles the full symbolic expression from the doubled system
# operators and the bath.  The result is a sum of products mixing `SpinSym`
# (via `@commutative`) with the new `HEOMBosonicOps` auxiliary algebra.

M_sym = heom_generator(H_l, H_r, V_l, V_r, bath)
println("\nSymbolic HEOM generator:\n  ", M_sym)

# ## Constructing the full Hilbert space
#
# The full augmented space is the tensor product of the doubled physical space
# and the auxiliary ADO space.

Hs_l = hilbert_space(σ_l) 
Hs_r = hilbert_space(σ_r)  
Haux = heom_bosonic_aux_space(bath) # dim = heom_aux_dim(N_exp=1, m_max=2) = 3

Hfull = tensor_product((Hs_l, Hs_r, Haux))
println("\nAuxiliary space dimension: ", dim(Haux),
    "  (expected: ", heom_aux_dim(1, m_max), ")")
println("Full augmented space dimension: ", dim(Hfull),
    "  (= 2 × 2 × ", dim(Haux), ")")

# ## Matrix representation
#
# `matrix_representation` dispatches on `symbolic_group` for each factor in the
# symbolic expression, partitions the operator across the product-space
# subsystems, and assembles the sparse matrix via tensor (Kronecker) products.

M_mat = matrix_representation(M_sym, Hfull)
println("\nHEOM matrix size: ", size(M_mat))

# ## Structure checks
#
# 1. **Correct dimension**.
@assert size(M_mat, 1) == dim(Hfull)

# 2. **Coherent part is traceless**: the coherent contribution ``i(H_l - H_r)``
#    corresponds to the commutator ``-i[H_s, \rho]`` and must be trace-preserving.
coherent_mat = matrix_representation(1im * (H_l - H_r), tensor_product((Hs_l, Hs_r)))
@assert abs(tr(coherent_mat)) < 1e-12 "coherent part not traceless"

# 3. **Damping is real and diagonal** (for real χ).
Maux = matrix_representation(HEOMBosonicDamping(bath), Haux)
@assert isdiag(Maux)
@assert all(isreal, diag(Maux))

# 4. **Round-trip reshape consistency** between the m=0 level of the augmented
#    vector and the physical density matrix.
#    The density-matrix elements sit in the m=0 sector of the auxiliary index.
m0_inds = [i for (i, s) in enumerate(basisstates(Hfull)) if s.states[3].n == [0]]
@assert length(m0_inds) == dim(Hs_l) * dim(Hs_r)

# ## Density-matrix reshape
#
# The mapping between the ``d^2``-vector (restricted to the ``m=0`` sector) and
# the ``d \times d`` density matrix uses the same `reshape` utilities as the
# ordinary Lindblad tutorial.

Hdouble = tensor_product((Hs_l, Hs_r))
to_vec(ρ) = reshape(ρ, (Hs_l, Hs_r) => Hdouble)
to_mat(v) = reshape(v, Hdouble => (Hs_l, Hs_r))

ρ_test = randn(ComplexF64, dim(Hs_l), dim(Hs_r))
@assert to_mat(to_vec(ρ_test)) ≈ ρ_test "round-trip reshape failed"

# ## Inspecting the auxiliary-space operators
#
# Let's look at the individual auxiliary operators for the single-mode case.

Aup1 = HEOMBosonicUp(1, bath)
Adown1 = HEOMBosonicDown(1, bath)
W_op = HEOMBosonicDamping(bath)

println("\nAup(1) matrix on auxiliary space:")
display((matrix_representation(Aup1, Haux; projection=true)))

println("\nAdown(1) matrix on auxiliary space:")
display((matrix_representation(Adown1, Haux)))

println("\nDamping W matrix on auxiliary space:")
display((matrix_representation(W_op, Haux)))

# ## Summary
#
# The workflow for bosonic HEOM in FermionicHilbertSpaces.jl is:
#
# | Step                          | Tool                                             |
# |:------------------------------|:-------------------------------------------------|
# | Doubled system operators      | `@spins σ_l σ_r` (or `@fermions`, `@boson`)     |
# | Bath correlation function     | `HEOMBosonicBath(id, N_exp, m_max, χ, ξ)`       |
# | Symbolic HEOM generator       | `heom_generator(H_l, H_r, V_l, V_r, bath)`      |
# | Auxiliary Hilbert space       | `heom_bosonic_aux_space(bath)`                   |
# | Full augmented space          | `tensor_product((Hs_l, Hs_r, Haux))`            |
# | Sparse matrix representation  | `matrix_representation(M_sym, Hfull)`            |
# | Density matrix ↔ vector       | `reshape` with doubled-space syntax              |
#
# From here, standard Julia tools (DifferentialEquations.jl, LinearSolve.jl,
# `eigen`) can be applied to `M_mat` to integrate the dynamics or find
# steady states — exactly as done in the ordinary Lindblad tutorial.
println("\nAll checks passed.")
