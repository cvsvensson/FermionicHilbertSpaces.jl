using FermionicHilbertSpaces

N = 20
n = 5
fs = FermionicHilbertSpaces.fockstates(N, n) # Generate all fock states on N modes with n particles
H = hilbert_space(1:N, FermionConservation(), fs)
@fermions c

# Define a Hamiltonian that conserves the number of fermions
hamiltonian = sum(rand() * (c[i]' * c[i+1] + hc) for i in 1:N-1) +
              sum(rand() * c[i]' * c[i] for i in 1:N) +
              sum(rand() * c[i]' * c[i] * c[i+1]' * c[i+1] for i in 1:N-1)

ham = Hermitian(matrix_representation(hamiltonian, H))

## Solve for ground state
using KrylovKit
vals, vecs = eigsolve(v -> ham * v, rand(size(ham, 1)), 1)

## Partial trace to half the system
Hsub = FermionicHilbertSpaces.subspace(1:div(10, 2), H)
subrho = partial_trace(vecs[1] * vecs[1]', H => Hsub)
sum(v -> -v * log(abs(v)), eigvals(subrho))
