using FermionicHilbertSpaces, LinearAlgebra, SparseArrays

@fermions f
kitaev_chain(N, f, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:N-1) +
                                 sum(Δ * f[i] * f[i+1] + hc for i in 1:N-1) +
                                 sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:N-1) +
                                 sum(μ * f[i]' * f[i] for i in 1:N)
N = 4
H = hilbert_space(1:N, ParityConservation())
params = (μ, t, Δ, U) = (1.0, 1.0, 0.5, 2.0)

# Use symbolic fermions and evaluate on the hilbert space
ham = kitaev_chain(N, f, μ, t, Δ, U)
h = matrix_representation(ham, H)

# Or, define fermions first and use them in the kitaev_chain function
fH = fermions(H)
h ≈ kitaev_chain(N, fH, μ, t, Δ, U)

## Even subspace
Heven = hilbert_space(1:N, [FockNumber(n) for n in 0:2^N-1 if iseven(count_ones(n))])
heven = matrix_representation(ham, Heven)
h[H.symmetry.qntoinds[1], H.symmetry.qntoinds[1]] ≈ heven
# fermions(H2) # errors, because fermions map outside the subspace

