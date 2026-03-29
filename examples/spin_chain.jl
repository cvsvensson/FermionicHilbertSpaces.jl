using FermionicHilbertSpaces
N = 3
@spins S
H = hilbert_space(S, 1:N, 1 // 2)
## Heisenberg chain
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)

## Try spin 1
H = hilbert_space(S, 1:N, 1)
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)

