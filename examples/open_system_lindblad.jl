# # Open Quantum Systems 
# In FermionicHilbertSpaces, open systems can be represented as a tensor product
# of a "left" and "right" copy of the system Hilbert space. The right copy is 
# time-reversed, so that right-actions correspond to transposed matrices.
# The `open_system` helper function sets up this structure:
using FermionicHilbertSpaces
using FermionicHilbertSpaces: open_system
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)
# This generates two versions of hilbert_space(c, 1:1), tagged as `:left` and `:right`. 
# `Hr` is a transposed space which automatically gives transposed matrix representations
# satisfying `matrix_representation(A*B, Hr) == matrix_representation(B, Hr) * matrix_representation(A, Hr)`,
# i.e. `A` acts on `Hr` before `B` does. This choice is natural when working with density matrices as
# in an expression like
# ```math
# \rho * A * B,
# ```
# A acts before B.

# The left and right spaces each have a symbolic basis associated to them, which are tagged 
# copies of the original `c` basis. The `left` and `right` functions map symbolic expressions 
# in the original basis to the respective copy, so that we can write an operator in the 
# original basis and then map it to the left/right space as needed. For example, if the
# Hamiltonian is
ham_sym = 1.0 * c[1]'c[1]
# then the Liouvillian superoperator is
L_sym = 1im * (left(ham_sym) - right(ham_sym))
# We can get a matrix representation of the Liouvillian on the full space `Hlr` with
L_mat = matrix_representation(L_sym, Hlr)
# A density matrix has two indices, correspinding to (Hl, Hr) and the vectorized form 
# has one index corresponding to Hlr. To translate between these, use `reshape`.
# E.g. to vectorize the identity operator we can do
Ivec = reshape(I(dim(Hl)), (Hl, Hr) => Hlr)
# The Liouvillian preserves trace which is equivalent to the condition that the 
# adjoint annihilates the identity matrix:
iszero(L_mat' * Ivec)
# Reshaping a vector in `Hlr` to a matrix in `(Hl, Hr)`:
v = rand(dim(Hlr))
reshape(v, Hlr => (Hl, Hr))

# ## Lindblad example

# We study a **single fermionic level** at energy ``\varepsilon`` coupled to a
# Markovian reservoir, with incoherent gain/loss rates:
# ```math
# H = \varepsilon\, c^\dagger c, \qquad
# L_\text{in} = \sqrt{\gamma_\text{in}}\, c^\dagger, \qquad
# L_\text{out} = \sqrt{\gamma_\text{out}}\, c.
# ```
# The analytical steady-state occupation is
# ```math
# \langle n \rangle_\text{ss} = \frac{\gamma_\text{in}}{\gamma_\text{in} + \gamma_\text{out}}.
# ```
using FermionicHilbertSpaces, LinearAlgebra

# Parameters
ε = 1.0
γ_in = 0.3
γ_out = 0.7

# ## Step 1: Build the open-system Hilbert space
@fermions c
Hlr, Hl, Hr, left, right = open_system(c, 1:1)
ham_sym = ε * c[1]' * c[1]
L_in = √γ_in * c[1]'
L_out = exp(rand() * 2pi * im) * √γ_out * c[1]

# Let's define a helper for the dissipator superoperator and write down the Lindbladian
dissipator(L) = left(L) * right(L') - 0.5 * (left(L'L) + right(L'L))
lindbladian = 1im * (left(ham_sym) - right(ham_sym)) + dissipator(L_in) + dissipator(L_out)
# Let's get the matrix representation and check that it preserves trace 
mat = matrix_representation(lindbladian, Hlr)
Ivec = reshape(I(dim(Hl)), (Hl, Hr) => Hlr)
iszero(mat' * Ivec)
# ## Solve for the steady state
# The steady state is the eigenvector of `mat` with eigenvalue closest to zero.
vals, vecs = eigen(Matrix(mat); sortby=abs)
v = vecs[:, 1]
ρ = reshape(v, Hlr => (Hl, Hr))
ρ ./= tr(ρ)

# Extract occupation ⟨n⟩ = Tr(c†c · ρ_ss) on the physical (left) space.
n_mat = matrix_representation(left(c[1]'c[1]), Hl)
n_ss = tr(n_mat * ρ)
# or equivalently in the vectorized space
n_matlr = matrix_representation(left(c[1]'c[1]), Hlr)
n_ss = Ivec' * n_matlr * v / (Ivec' * v)
n_analytical = γ_in / (γ_in + γ_out)
println("Steady-state occupation ⟨n⟩_ss = $n_ss  (analytic: $n_analytical)")

# ## Use a conserved quantity for block structure

# For this model, while there are particles jumping in and out of the system,
# the difference in particle number between the left and right space is conserved.
# However, because the right space is transposed, particles and holes are swapped,
# so the conserved quantity is in fact the total particle number:
# ```math
# Q = n_l + n_r.
# ```
# is conserved. We can constrain to fixed `Q` sectors and get block matrices.
constraint = NumberConservation(-1:1, [Hl, Hr], [1, -1])
Hcons = tensor_product((Hl, Hr); constraint)
mat_cons = matrix_representation(lindbladian, Hcons)
blocks = map(sectors(Hcons)) do Hsector
    matrix_representation(lindbladian, Hsector)
end
