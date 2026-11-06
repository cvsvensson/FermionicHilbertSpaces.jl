# # Spins

# This page shows how to declare spin operators, what conventions they follow, and how to build spin Hilbert spaces.
# A Heisenberg chain then shows how magnetization and total-spin sectors block-diagonalize a spin Hamiltonian,
# and that its ground state is a singlet.

# ## Spin operators and spaces

# `@spin T` declares one spin. Indexing it with `:x`, `:y`, `:z` gives the spin components, `:+` and `:-` the
# ladder operators $S^\pm = S^x \pm i S^y$, and `T[:]` the vector of components. These are spin operators, not
# Pauli matrices: they satisfy $[S^x, S^y] = i S^z$, and $S^z$ has eigenvalues $-J, \dots, J$.

using FermionicHilbertSpaces
using FermionicHilbertSpaces: Axes
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
# A diagonal operator constrains a space; for example, `sum(S[k][:z] for k in 1:N) => M` keeps the states with
# total $S^z$ equal to `M`.

# ## Heisenberg chain

# The spin-1/2 Heisenberg chain $H = \sum_k \vec S_k \cdot \vec S_{k+1}$ commutes with the total magnetization
# $S^z_\mathrm{tot} = \sum_k S^z_k$, so we can restrict to the sector where it vanishes.

@spins S 1 // 2
N = 4
ham = sum(S[k][:]'S[k + 1][:] for k in 1:(N - 1))
Sz = sum(S[k][:z] for k in 1:N)
@assert ham * Sz - Sz * ham == 0
Hm = hilbert_space(S, 1:N, Sz => 0)
Mham = representation(ham, Hm, Matrix);

# The chain also conserves the total spin, fixed by the Casimir operator $\vec S_\mathrm{tot}^2 = J(J+1)$.
# Projecting onto its eigenspaces block-diagonalizes the Hamiltonian, and the blocks together reproduce its spectrum.
# The lowest energy of each block, printed below, is smallest for $J=0$. Dense matrices are used because
# `nullspace` and `eigvals` need them.

Stot2 = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
Mtot = representation(Stot2, Hm, Matrix)
blocks = map(J -> nullspace(Mtot - J * (J + 1) * I), 0:2)
spectra = [eigvals(Hermitian(P' * Mham * P)) for P in blocks]
@assert sort(reduce(vcat, spectra)) ≈ eigvals(Hermitian(Mham))
minimum.(spectra)

# The same blocks follow without forming $\vec S_\mathrm{tot}^2$. A multiplet with total spin $J$ has exactly one
# highest-weight state, with $S^z_\mathrm{tot} = J$ and annihilated by the raising operator $S^+_\mathrm{tot}$.
# Building one space with all magnetization sectors, the raising operator maps sector $J$ to sector $J+1$, and its
# nullspace gives these states, one per multiplet. In the top sector $J = N/2$ every state is a highest-weight state.

Splus = sum(S[k][:+] for k in 1:N)
Hsz = hilbert_space(S, 1:N, Sz)
function highest_weight_states(J)
    HJ = sector(J, Hsz)
    J + 1 in quantumnumbers(Hsz) || return HJ, Matrix(I, dim(HJ), dim(HJ))
    raising = representation(Splus, Axes(sector(J + 1, Hsz), HJ), Matrix)
    return HJ, nullspace(raising)
end
hw_spectra = map(0:2) do J
    HJ, V = highest_weight_states(J)
    eigvals(Hermitian(V' * representation(ham, HJ, Matrix) * V))
end
@assert all(hw_spectra .≈ spectra)

# States symmetric under all permutations of the sites have the maximal total spin, so the symmetric
# sector, built with [`symmetric_sector`](@ref), reproduces the highest-weight spectrum for $J = N/2$.

Psym = symmetric_sector(Hm, factors(Hm))
@assert eigvals(Hermitian(Psym' * Mham * Psym)) ≈ hw_spectra[end]

# For Hamiltonians symmetric under permuting sites, [`symmetric_sector`](@ref) projects onto permutation sectors; see [Spatial symmetries](translation_symmetry.md).
