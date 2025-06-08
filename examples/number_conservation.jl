using FermionicHilbertSpaces

N = 62 # Can't do more than 63, because of Int64
n = 3

H = hilbert_space(1:N, FermionConservation(n)) # Spans all states with n fermions on N sites
@fermions c

# Define a Hamiltonian that conserves the number of fermions
hamiltonian = sum(rand() * (c[i]' * c[i+1] + hc) for i in 1:N-1) +
              sum(rand() * c[i]' * c[i] for i in 1:N) +
              sum(rand() * c[i]' * c[i] * c[i+1]' * c[i+1] for i in 1:N-1);

ham = Hermitian(matrix_representation(hamiltonian, H))

## Solve for ground state
using KrylovKit
vals, vecs = eigsolve(v -> ham * v, rand(size(ham, 1)), 1)

## Partial trace to half the system
Hsub = subspace(1:div(N, 2), H)
#vecs[1]*vecs[1]' runs out of memory for large N, so we use a rank-1 matrix
struct Rank1Matrix{T} <: AbstractMatrix{T}
    vec::Vector{T}
end
Base.getindex(m::Rank1Matrix, i::Int, j::Int) = m.vec[i] * conj(m.vec[j])
Base.size(m::Rank1Matrix) = (length(m.vec), length(m.vec))
rho = Rank1Matrix(vecs[1]);
subrho = partial_trace(rho, H => Hsub)
sum(v -> -v * log(abs(v)), eigvals(subrho))
