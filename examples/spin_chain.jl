# # Spin chains

# This tutorial builds spin operators and a Heisenberg chain, and restricts it to a magnetization
# sector and a total-spin sector. It assumes [Getting started](index.md).

using FermionicHilbertSpaces
using LinearAlgebra

# ## Spin operators
# `@spin T` declares one spin whose components are indexed by `:x`, `:y`, `:z`
# and the ladder operators `:+`, `:-`. Without a spin value, the operators satisfy only the spin algebra.
@spin T
T[:x] * T[:y] - T[:y] * T[:x] == 1im * T[:z]

# Giving a spin value `1//2` or `1` adds the corresponding normalization.
@spin Shalf 1//2
@spin Sone 1
Shalf[:x]^2 == 1//4
sum(op^2 for op in Sone[:]) == 2

# For spin one half, expressions simplify to a unique form. For higher spins they do not,
# so equal operators can compare unequal as symbolic expressions.
# Compare their matrices instead:
representation(Sone[:x]^3, hilbert_space(Sone), :dense) ≈ representation(Sone[:x], hilbert_space(Sone), :dense)

# ## Heisenberg chain
# `@spins S` declares a field of spins, with `S[k][:x]` the operator on site `k`.
# `hilbert_space(S, labels)` builds the space of the sites in `labels`.
@spins S 1 // 2
N = 4
H = hilbert_space(S, 1:N)
ham = sum(S[k][:]'S[k+1][:] for k in 1:(N-1))
representation(ham, H, :dense)

# The chain conserves the total magnetization. An `AdditiveConstraint` restricts the space to
# states where the sum over sites of `s.m` is fixed:
Hm = hilbert_space(S, 1:N, AdditiveConstraint(0; maps = s -> s.m))
representation(ham, Hm, :dense)

# ## Advanced: total spin sectors
# The total spin $J$ is fixed by the Casimir operator $ec S_\mathrm{tot}^2 = J(J+1)$. Projecting onto its
# eigenspaces block-diagonalizes the Hamiltonian, and the blocks together reproduce its spectrum.
Mham = representation(ham, Hm, :dense)
Stot2 = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
Mtot = representation(Stot2, Hm, :dense)
blocks = map(J -> nullspace(Mtot - J * (J + 1) * I), 0:2)
sort(reduce(vcat, [eigvals(Hermitian(P' * Mham * P)) for P in blocks])) ≈ eigvals(Hermitian(Mham))

# Permutation symmetry of the sites gives further sectors, see [`symmetric_sector`](@ref).
