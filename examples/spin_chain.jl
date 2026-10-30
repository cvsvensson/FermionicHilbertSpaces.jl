# # Spin chains

# Spin operators in this package are symbolic, with or without a fixed spin value, and spin chains can be
# restricted to symmetry sectors like any other Hilbert space. Here we build a Heisenberg chain, fix its
# total magnetization with an `AdditiveConstraint`, and block-diagonalize it by total spin $J$ using the
# Casimir operator, which shows that the ground state is a singlet.

using FermionicHilbertSpaces
using LinearAlgebra

# `@spin T` declares one spin with components `:x`, `:y`, `:z` and ladder operators `:+`, `:-`. Without a spin value
# only the spin algebra holds; a value such as `1//2` adds the normalization. `@spins S 1//2` declares a field of spins,
# with `S[k][:x]` the operator on site `k`, and `hilbert_space(S, labels)` builds the space of the sites in `labels`.
# The chain conserves the total magnetization, so we restrict to the sector where the sum over sites of `s.m` vanishes.

@spin T
T[:x] * T[:y] - T[:y] * T[:x] == 1im * T[:z]
@spin Shalf 1//2
Shalf[:x]^2 == 1 // 4

@spins S 1 // 2
N = 4
ham = sum(S[k][:]'S[k + 1][:] for k in 1:(N - 1))
Hm = hilbert_space(S, 1:N, AdditiveConstraint(0; maps = s -> s.m))
Mham = representation(ham, Hm, :dense);

# The total spin is fixed by the Casimir operator $\vec S_\mathrm{tot}^2 = J(J+1)$. Projecting onto its eigenspaces
# block-diagonalizes the Hamiltonian, and the blocks together reproduce its spectrum. The lowest energy of each block
# is printed below and is smallest for $J=0$. Dense matrices are used here because `nullspace` and `eigvals` need them.

Stot2 = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
Mtot = representation(Stot2, Hm, :dense)
blocks = map(J -> nullspace(Mtot - J * (J + 1) * I), 0:2)
spectra = [eigvals(Hermitian(P' * Mham * P)) for P in blocks]
@assert sort(reduce(vcat, spectra)) ≈ eigvals(Hermitian(Mham))
minimum.(spectra)

# For Hamiltonians symmetric under permuting sites, [`symmetric_sector`](@ref) projects onto permutation sectors; see [Spatial symmetries](translation_symmetry.md).
