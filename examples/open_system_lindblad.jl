# # Open Quantum Systems and the Lindblad Master Equation

# This tutorial shows how to simulate open fermionic quantum systems using the
# Lindblad master equation.  We exploit a key feature of FermionicHilbertSpaces.jl:
# operators that belong to **different fermionic groups commute**, which lets us
# construct the Lindblad superoperator as a symbolic expression over a
# *doubled Hilbert space*.

# ## Background

# An open quantum system evolves according to the Lindblad master equation
# ```math
# \frac{d\rho}{dt} = -i[H,\rho] + \sum_j \left(L_j \rho L_j^\dagger
#   - \tfrac{1}{2}\bigl\{L_j^\dagger L_j, \rho\bigr\}\right) \equiv \mathcal{L}[\rho].
# ```
# A standard trick turns this into a *linear map on vectors*: if we stack the
# columns of the ``d \times d`` density matrix into a ``d^2``-dimensional vector
# ``|\rho\rangle\rangle``, the master equation becomes
# ```math
# \frac{d}{dt}|\rho\rangle\rangle = \mathbf{L}\,|\rho\rangle\rangle,
# ```
# where ``\mathbf{L}`` is the **Liouvillian superoperator**.  The steady state
# corresponds to the eigenvector of ``\mathbf{L}`` with eigenvalue zero.

# ### Doubled Hilbert space

# In FermionicHilbertSpaces.jl the column-stacking of ``\rho`` is encoded by
# **two independent sets of fermionic modes**:
# - **left** modes `c_l` act on ``\rho`` from the left (ket side),
# - **right** modes `c_r` act on ``\rho`` from the right (bra side, with a sign
#   flip from transposition).
#
# Because they belong to different fermionic groups they *commute* with each
# other, so the superoperator is simply
# ```math
# \mathbf{L} = i(H_l - H_r) + \sum_j \left(L_j^{(l)} L_j^{(r)}
#   - \tfrac{1}{2}\bigl(L_j^{\dagger(l)} L_j^{(l)}
#   + L_j^{\dagger(r)} L_j^{(r)}\bigr)\right),
# ```
# where superscripts ``(l/r)`` mean "built from left/right modes".
# This symbolic expression is then evaluated on the tensor-product space
# ``\mathcal{H}_l \otimes \mathcal{H}_r``.

# ## Physical setup

# We study a **single fermionic level** at energy ``\varepsilon`` coupled to a
# Markovian reservoir, with incoherent gain rate ``\gamma_\text{in}`` (particles
# flowing in) and loss rate ``\gamma_\text{out}`` (particles flowing out):
# ```math
# H = \varepsilon\, c^\dagger c, \qquad
# L_\text{in} = \sqrt{\gamma_\text{in}}\, c^\dagger, \qquad
# L_\text{out} = \sqrt{\gamma_\text{out}}\, c.
# ```
# The analytical steady-state occupation is
# ```math
# \langle n \rangle_\text{ss} = \frac{\gamma_\text{in}}{\gamma_\text{in} + \gamma_\text{out}}.
# ```

using FermionicHilbertSpaces
using LinearAlgebra

# Parameters
ε = 1.0   # on-site energy
γ_in = 0.3   # gain rate
γ_out = 0.7   # loss rate

# ## Constructing the Liouvillian symbolically

# Define two **independent** groups of fermionic modes.  The `@fermions` macro
# creates a symbolic fermion basis — indexing it gives annihilation operators.
@fermions c_l   # left (ket) copy
@fermions c_r   # right (bra) copy

# Symbolic Hamiltonian and jump operators
H_sym(c) = ε * c[1]' * c[1]
L_in(c) = √γ_in * c[1]'
L_out(c) = √γ_out * c[1]

# The Liouvillian superoperator in the doubled basis.
# Note:  c_l operators commute with c_r operators, so we can freely mix them.
lindbladian = let Hl = H_sym(c_l),
    Hr = H_sym(c_r),
    Ll_in = L_in(c_l), Lr_in = L_in(c_r),
    Ll_out = L_out(c_l), Lr_out = L_out(c_r)

    iL_coherent = 1im * (Hl - Hr)

    D_in = Ll_in * Lr_in - 0.5 * (Ll_in' * Ll_in + Lr_in' * Lr_in)
    D_out = Ll_out * Lr_out - 0.5 * (Ll_out' * Ll_out + Lr_out' * Lr_out)

    iL_coherent + D_in + D_out
end

# ## Building the Hilbert space and matrix representation

# The physical Hilbert space for a single mode has dimension 2 (empty / occupied).
Hl = hilbert_space(c_l[1])   # dim = 2
Hr = hilbert_space(c_r[1])   # dim = 2

# The doubled space has dimension 2×2 = 4, corresponding to the 4 independent
# matrix elements of the 2×2 density matrix.
Hlr = tensor_product((Hl, Hr))

mat = matrix_representation(lindbladian, Hlr)

# ## Density-matrix vectorization

# The connection between the ``d^2``-vector picture and the ``d \times d`` density
# matrix is handled by the `reshape` function with the special `(Hl, Hr) => Hlr`
# notation, which takes care of the correct fermionic ordering.

to_vec(ρ) = reshape(ρ, (Hl, Hr) => Hlr)  # d×d matrix  → d²-vector
to_mat(v) = reshape(v, Hlr => (Hl, Hr)) # d²-vector   → d×d matrix

# As a quick sanity check, round-tripping should be the identity:
ρ_test = randn(ComplexF64, dim(Hl), dim(Hr))
@assert to_mat(to_vec(ρ_test)) ≈ ρ_test "round-trip reshape failed"

# ## Finding the steady state

# The steady state is the null vector of ``\mathbf{L}``.  For our small 4×4
# matrix we can use a full eigendecomposition.
vals, vecs = eigen(Matrix(mat); sortby=real)

ss_idx = argmin(abs.(real.(vals)))
λ_ss = vals[ss_idx]
v_ss = vecs[:, ss_idx]

# The real part of all eigenvalues should be ≤ 0; the steady-state eigenvalue
# is the one closest to zero:
@assert abs(real(λ_ss)) < 1e-10 "no zero eigenvalue found — check the Liouvillian"

# Reshape back to a density matrix and normalize
ρ_ss = to_mat(v_ss)
ρ_ss ./= tr(ρ_ss)   # ensure unit trace

# Verify physicality: trace-one Hermitian matrix with positive diagonal
@assert abs(tr(ρ_ss) - 1) < 1e-10 "steady state not trace-one"
@assert norm(ρ_ss - ρ_ss') < 1e-10 "steady state not Hermitian"

# Extract the steady-state occupation ⟨n⟩ = Tr(c†c · ρ_ss)
c_mat = matrix_representation(c_l[1], Hl)   # annihilation on H_l
n_mat = Matrix(c_mat' * c_mat)         # number operator on H_l
n_ss = real(tr(n_mat * ρ_ss))

n_analytical = γ_in / (γ_in + γ_out)

@assert abs(n_ss - n_analytical) < 1e-10 "occupation doesn't match analytic formula"

println("Steady-state occupation ⟨n⟩_ss = $n_ss  (analytic: $n_analytical)")

# ## Exploiting a conserved quantity

# Since particles jump in and out, particle number is not conserved. But the *difference* in
# particle number between the left and right copies,
# ```math
# Q = n_l - n_r,
# ```
# is conserved: We can therefore block-diagonalise the superoperator by constraining the
# tensor-product space to sectors of fixed ``Q``.

constraint = NumberConservation(-1:1, [Hl, Hr], [1, -1])
Hcons = tensor_product((Hl, Hr); constraint)

mat_cons = matrix_representation(lindbladian, Hcons)

# `sectors` returns the individual symmetry blocks:
blocks = map(sectors(Hcons)) do Hsector
    matrix_representation(lindbladian, Hsector)
end

# The block-diagonal assembly must reproduce the full constrained matrix:
@assert cat(blocks...; dims=(1, 2)) == mat_cons "block assembly failed"

println("Block dimensions: ", [size(b, 1) for b in blocks])

# ## Summary

# FermionicHilbertSpaces.jl makes it straightforward to set up and study
# Lindblad dynamics for fermionic systems:
#
# | Step                        | Tool                                              |
# |:---------------------------|:--------------------------------------------------|
# | Symbolic Liouvillian        | Two `@fermions` groups, arithmetic on operators   |
# | Matrix representation       | `matrix_representation(lindbladian, Hlr)`         |
# | Density-matrix ↔ vector     | `reshape` with `(Hl, Hr) => Hlr` syntax           |
# | Steady state                | `eigen` on the Liouvillian matrix                 |
# | Symmetry block structure    | `NumberConservation`, `sectors`, `sector`         |
#
# The same workflow extends to multi-mode systems, spin or bosonic degrees of
# freedom, and more exotic jump operators — simply add more modes and terms to
# the symbolic Liouvillian.
