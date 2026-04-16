using FermionicHilbertSpaces
using Combinatorics
N = 4
@spins S 1 // 2
H = hilbert_space(S, 1:N)
## Heisenberg chain
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
M = matrix_representation(ham, H)
Peven = symmetric_sector(H, factors(H), :symmetric)
Podd = symmetric_sector(H, factors(H), :antisymmetric)
using AbstractAlgebra
G = AbstractAlgebra.SymmetricGroup(N)
projectors = map(AbstractAlgebra.Generic.partitions(N)) do λ
    χ = AbstractAlgebra.character(λ)
    perms = Iterators.map(p -> p.d, G)
    weights = Iterators.map(χ, G)
    P = symmetric_sector(H, factors(H), (perms, weights), method=:states)
end
blocks = map(projectors) do p # block-diagonalize M in the symmetry sectors
    p' * M * p
end

## Try spin 1
@spins S 1
constraint = FermionicHilbertSpaces.AdditiveConstraint(1, s -> s.m)
H = hilbert_space(S, 1:N, constraint)
ham = sum(S[k][op] * S[k+1][op] for k in 1:N-1 for op in (:x, :y, :z))
total_spin_op = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
total_spin_mat = matrix_representation(total_spin_op, H)
M = matrix_representation(ham, H)
P = nullspace(Matrix(total_spin_mat - 4 * I))
P' * M * P
Peven = symmetric_sector(H, factors(H), :symmetric)
Podd = symmetric_sector(H, factors(H), :antisymmetric)

G = AbstractAlgebra.SymmetricGroup(N)
projectors = map(AbstractAlgebra.Generic.partitions(N)) do λ
    χ = AbstractAlgebra.character(λ)
    perms = Iterators.map(p -> p.d, G)
    weights = Iterators.map(χ, G)
    P = symmetric_sector(H, factors(H), (perms, weights))
end

blocks = map(projectors) do p # block-diagonalize M in the symmetry sectors
    p' * M * p
end


sum(first ∘ size, blocks)