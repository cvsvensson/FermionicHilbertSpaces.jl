using FermionicHilbertSpaces
N = 3
@spins S 1 // 2
H = hilbert_space(S, 1:N)
## Heisenberg chain
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)

## Try spin 1
@spins S 1
constraint = FermionicHilbertSpaces.AdditiveConstraint(1, s -> s.m)
H = hilbert_space(S, 1:N, constraint)
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
total_spin_op = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
total_spin_mat = matrix_representation(total_spin_op, H)
M = matrix_representation(ham, H)
P = nullspace(Matrix(total_spin_mat - 2 * I))
P' * M * P

