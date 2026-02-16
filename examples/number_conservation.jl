using FermionicHilbertSpaces, LinearAlgebra, SparseArrays

N = 62 # If you go above 62, BigInt is used which is slower
n = 2

H = hilbert_space(1:N, number_conservation(n)) # Spans all states with n fermions on N sites
@fermions c

# Define a Hamiltonian that conserves the number of fermions
hamiltonian = sum((c[i]' * c[i+1] + hc) for i in 1:N-1) + sum(rand() * c[i]' * c[i] * c[i+1]' * c[i+1] for i in 1:N-1) #+ sum(rand() * c[i]' * c[i] for i in 1:N);

ham = Hermitian(matrix_representation(hamiltonian, H))

## Solve for ground state
using KrylovKit
vals, vecs = eigsolve(v -> ham * v, rand(size(ham, 1)), 1)

## Partial trace to half the system
Hsub = subregion(1:div(N, 2), H)
#vecs[1]*vecs[1]' runs out of memory for large N, so let's use LowRankMatrices.jl
using LowRankMatrices
rho = LowRankMatrix(vecs[1], conj(vecs[1]))
@time subrho = partial_trace(rho, H => Hsub);
sum(v -> -v * log(abs(v) + 1e-16), eigvals(subrho))