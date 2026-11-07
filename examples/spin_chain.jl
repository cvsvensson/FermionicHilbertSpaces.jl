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
# On a space with all magnetization sectors, `highest_weight_states` returns these states for each sector $J$,
# one per multiplet, in the basis of `sector(J, Hsz)`.

using FermionicHilbertSpaces: highest_weight_states, multiplets
Splus = sum(S[k][:+] for k in 1:N)
Hsz = hilbert_space(S, 1:N, Sz)
hw_spectra = [eigvals(Hermitian(V' * representation(ham, sector(J, Hsz), Matrix) * V)) for (J, V) in highest_weight_states(Hsz, Splus)]
@assert all(hw_spectra .≈ spectra)

# Lowering them with $S^-_\mathrm{tot}$ gives `multiplets`, for each $J$ an orthonormal basis of the states with
# total spin $J$ in each sector $M$. The triplets have the same energies in each of the sectors $M = -1, 0, 1$.

W = multiplets(Hsz, Splus)
@assert all(eigvals(Hermitian(B' * representation(ham, sector(M, Hsz), Matrix) * B)) ≈ spectra[2] for (M, B) in W[1])

# States symmetric under all permutations of the sites have the maximal total spin, so the symmetric
# sector, built with [`symmetric_sector`](@ref), reproduces the highest-weight spectrum for $J = N/2$.

Psym = symmetric_sector(Hm, factors(Hm))
@assert eigvals(Hermitian(Psym' * Mham * Psym)) ≈ hw_spectra[end]

# Total spin $J$ needs only the sectors from $M = J$ up, so longer chains never need the full space.
# The $J = 3$ states of a Heisenberg ring with $N = 10$ sites come from the space with $S^z_\mathrm{tot} \geq 3$.

N = 10
ring = sum(S[k][:]'S[mod1(k + 1, N)][:] for k in 1:N)
Sz = sum(S[k][:z] for k in 1:N)
Splus = sum(S[k][:+] for k in 1:N)
H3 = hilbert_space(S, 1:N, Sz => 3:(N // 2))
V = highest_weight_states(H3, Splus)[3]
E3 = eigvals(Hermitian(V' * representation(ring, sector(3, H3), Matrix) * V))

# The states with $J = 3$ and $M = 2$ follow by lowering once. They form a subspace whose dimension is the number of
# $J = 3$ multiplets, with the same energies. As a check, we compare with the eigenstates in the $M = 2$ sector that
# have $\vec S_\mathrm{tot}^2 = 12$.

H2 = hilbert_space(S, 1:N, Sz => 2:(N // 2))
W32 = multiplets(H2, Splus)[3][2]
E32 = eigvals(Hermitian(W32' * representation(ring, sector(2, H2), Matrix) * W32))
@assert E32 ≈ E3
Stot2 = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
P = nullspace(representation(Stot2, sector(2, H2), Matrix) - 12I)
@assert E32 ≈ eigvals(Hermitian(P' * representation(ring, sector(2, H2), Matrix) * P))

# For Hamiltonians symmetric under permuting sites, [`symmetric_sector`](@ref) projects onto permutation sectors; see [Spatial symmetries](translation_symmetry.md).
