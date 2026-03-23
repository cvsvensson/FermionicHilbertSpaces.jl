using FermionicHilbertSpaces

@fermions c_l
@fermions c_r

hamiltonian(c) = c[1]'c[1]
jump_op(c) = c[1]'
lindbladian = 1im * (hamiltonian(c_l) - hamiltonian(c_r)) + jump_op(c_l) * jump_op(c_r) + 0.5 * (jump_op(c_l)' * jump_op(c_l) + jump_op(c_r)' * jump_op(c_r))

Hl = hilbert_space(c_l[1])
Hr = hilbert_space(c_r[1])
Hlr = tensor_product((Hl, Hr))
mat = matrix_representation(lindbladian, Hlr)
vals, vecs = eigen(Matrix(mat); sortby=abs)
stationary_state = vecs[:, 1]
reshape(stationary_state, Hlr, (Hl, Hr))

##
using FermionicHilbertSpaces

@fermions c_l
@fermions c_r

# Hamiltonian: pair creation/annihilation (maps to σ_x on even sector)
Δ = 1.0
hamiltonian(c) = Δ * (c[1]' * c[2]' + hc)

# Jump operator: pair creation (non-Hermitian)
γ = 0.5
jump_op(c) = c[1] * c[2]  # = σ_+ on even sector

# Lindbladian: -i[H,·] + γ(L·L† - ½{L†L,·})
# Note: your convention has 1im*(H_left - H_right) for coherent part
lindbladian = let Ll = jump_op(c_l), Lr = jump_op(c_r), Hl = hamiltonian(c_l), Hr = hamiltonian(c_r)
    -1im * (Hl - Hr) + γ * (Ll * Lr' - 0.5 * (Ll' * Ll + Lr * Lr'))
end
# Build Hilbert space for TWO fermionic modes (need c[1] and c[2])
Hl = hilbert_space(c_l, 1:2, ParityConservation(1))  # two modes on left
Hr = hilbert_space(c_r, 1:2, ParityConservation(1))  # two modes on right
Hlr = tensor_product((Hl, Hr))
mat = matrix_representation(lindbladian, Hlr)
mat * reshape(I(dim(Hlr)), (Hl, Hr) => Hlr) # This should give zero, but does not


vals, vecs = eigen(Matrix(mat); sortby=abs)

# Stationary state: eigenvalue with smallest |real part| (should be ~0)
stationary_idx = argmin(abs.(real.(vals)))
stationary_state = vecs[:, stationary_idx]

# Reshape to operator on Hl ⊗ Hr
ρ_ss = reshape(stationary_state, Hlr, (Hl, Hr))

#
normalize_density_matrix(ρ) = ρ / tr(ρ)
rhos = map(state -> reshape(state, Hlr, (Hl, Hr)), eachcol(vecs))
rhos = map(state -> normalize_density_matrix(reshape(state, Hlr, (Hl, Hr))), eachcol(vecs))

