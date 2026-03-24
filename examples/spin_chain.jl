import FermionicHilbertSpaces: SpinSpace
N = 3
@spins S 1:N
Hs = SpinSpace{1 // 2}.(S)
H = tensor_product(Hs)
## Heisenberg chain
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)

## Try spin 1
Hs = SpinSpace{1}.(S)
H = tensor_product(Hs)
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)

