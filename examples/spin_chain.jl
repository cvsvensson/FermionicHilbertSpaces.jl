# # Spins

# This page shows how to declare spin operators, what conventions they follow, and how to build spin Hilbert spaces.
# A Heisenberg chain then shows how magnetization and total-spin sectors block-diagonalize a spin Hamiltonian,
# and that its ground state is a singlet.

# ## Spin operators and spaces

# `@spin T` declares one spin. Indexing it with `:x`, `:y`, `:z` gives the spin components, `:+` and `:-` the
# ladder operators $S^\pm = S^x \pm i S^y$, and `T[:]` the vector of components. These are spin operators, not
# Pauli matrices: they satisfy $[S^x, S^y] = i S^z$, and $S^z$ has eigenvalues $-J, \dots, J$.

using FermionicHilbertSpaces
using LinearAlgebra

@spin T
T[:x] * T[:y] - T[:y] * T[:x] == 1im * T[:z]

# The spin value $J$ fixes the local dimension $2J+1$. It can be given when declaring the spin, which lets
# expressions simplify with relations that hold only for that $J$, such as $\vec S^2 = J(J+1)$. Without it, only the
# commutation relations hold, so a single symbolic Hamiltonian can be reused for any $J$, which is then passed to
# `hilbert_space`. The basis states are ordered by increasing $S^z$ eigenvalue.

@spin Shalf 1 // 2
sum(s^2 for s in Shalf[:]) == 3 // 4
representation(T[:z], hilbert_space(T, 1))

# `@spins S` declares a field of spins, where `S[k]` is the spin on site `k`, and `hilbert_space(S, labels)` builds
# the space of the sites in `labels` (with `J` as a third argument if the field has no spin value).
# Constraints act on the basis states of each site through their $S^z$ eigenvalue `s.m`; for example,
# `AdditiveConstraint(M; maps = s -> s.m)` keeps the states with total $S^z$ equal to `M`.

# ## Heisenberg chain

# The spin-1/2 Heisenberg chain $H = \sum_k \vec S_k \cdot \vec S_{k+1}$ commutes with the total magnetization
# $S^z_\mathrm{tot} = \sum_k S^z_k$, so we can restrict to the sector where it vanishes.

@spins S 1 // 2
N = 4
ham = sum(S[k][:]'S[k + 1][:] for k in 1:(N - 1))
Sz = sum(S[k][:z] for k in 1:N)
@assert ham * Sz - Sz * ham == 0
Hm = hilbert_space(S, 1:N, AdditiveConstraint(0; maps = s -> s.m))
Mham = representation(ham, Hm, :dense);

# The chain also conserves the total spin, fixed by the Casimir operator $\vec S_\mathrm{tot}^2 = J(J+1)$.
# Projecting onto its eigenspaces block-diagonalizes the Hamiltonian, and the blocks together reproduce its spectrum.
# The lowest energy of each block, printed below, is smallest for $J=0$. Dense matrices are used because
# `nullspace` and `eigvals` need them.

Stot2 = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
Mtot = representation(Stot2, Hm, :dense)
blocks = map(J -> nullspace(Mtot - J * (J + 1) * I), 0:2)
spectra = [eigvals(Hermitian(P' * Mham * P)) for P in blocks]
@assert sort(reduce(vcat, spectra)) ≈ eigvals(Hermitian(Mham))
minimum.(spectra)

# For Hamiltonians symmetric under permuting sites, [`symmetric_sector`](@ref) projects onto permutation sectors; see [Spatial symmetries](translation_symmetry.md).
