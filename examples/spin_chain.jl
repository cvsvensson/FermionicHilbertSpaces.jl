# # Spin chains

# In this example, we build Heisenberg spin chains and show how to exploit conserved
# quantities and permutation symmetry (invariance under exchanging sites) to
# block-diagonalize the Hamiltonian.

using FermionicHilbertSpaces
using LinearAlgebra

# ## Spin algebra
# Declare a spin mode and index into to get symbolic spin operators
@spin S
S[:x] * S[:y] - S[:y] * S[:x] == 1im * S[:z]; #true

# Allowed operator arguments are the spin components `:x`, `:y`, `:z` and the ladder operators `:+`, `:-`, and identity :I. There are operator aliases, so e.g. `S[1] == S[:x] == S[:X]`. The spin basis defined here has no fixed spin value. We can define an operator with a specific total spin which has more algebraic structure.
@spin Shalf 1//2
@spin Sone 1
Shalf[:x]^2 == 1//4
sum(s^2 for s in Sone[:]) == 2
# where `Sone[:]` gives a vector of the spin operators `[Sone[:x], Sone[:y], Sone[:z]]`.

# For spin half, expressions are simplified to a unique expression, but for higher spins this is not necessarily the case. One should be careful when comparing different algebraic expressions.

# Spin hilbert spaces can be constructed by 
Hhalf = hilbert_space(Shalf)
Hone = hilbert_space(Sone)
Hten = hilbert_space(S, 10)

# When dealing with multiple spins it is convenient to define a spin field which gives spin basis when indexing
@spins S
S[1][:+] # spin + operator on the site labeled 1

# A hilbert space for this field is obtained by hilbert_space(S, labels, spin)

# ## Basic example: Heisenberg chain
# A spin-1/2 Heisenberg chain hamiltonian can then be constructed as
N = 4
@spins S 1 // 2
H = hilbert_space(S, 1:N)
ham = sum(S[k][:]'S[k+1][:] for k in 1:(N-1))
M = representation(ham, H)

# We can exploit the fact that this hamiltonian conserves the total magnetization by
using FermionicHilbertSpaces: AdditiveConstraint
constraint = AdditiveConstraint(1; maps=s -> s.m)
H = hilbert_space(S, 1:N, constraint)
M = representation(ham, H)

# ## Advanced example: Conserving total spin, permutation symmetry
# We are not restricted to spin-1/2, but can use any spin `J`. On the symbolic level, we can also omit the spin value, and supply it later when we build the Hilbert space. 
@spins S
J = 1
N = 4
H = hilbert_space(S, 1:N, J)

# Let's make a permutationally invariant hamiltonian  as
# ham = sum(S[i][op] * S[mod1(i+1, N)][op] for i in 1:N for op in (:x, :y, :z))
ham = sum(S[i][op] * S[j][op] for i in 1:N, j in 1:N, op in (:x, :y, :z))

# This hamiltonian conserves the total magnetization, the total spin, and is invariant under any permutation of the sites. Let's exploit all these symmetries.
# ### Fixing the total magnetization

# `AdditiveConstraint` restricts the Hilbert space to states where the sum of a
# per-site quantity, here the magnetization `s.m`, equals a fixed value.
constraint = AdditiveConstraint(1; maps=s -> s.m)
H = hilbert_space(S, 1:N, J, constraint)


# ### Full decomposition into irreducible representations
# More generally, every irreducible representation of $S_N$ gives a symmetry sector.
# `AbstractAlgebra.jl` provides the character tables needed to build the corresponding
# projectors: each integer partition `λ` of `N` labels an irrep with character `χ`, and
# `symmetric_sector` accepts the permutations and their weights `(perms, weights) = (G, χ.(G))`
# directly instead of the `:symmetric`/`:antisymmetric` shorthand.
import AbstractAlgebra
G = collect(AbstractAlgebra.SymmetricGroup(N))
perms = Iterators.map(p -> p.d, G)
weights = Vector{Int}(undef, length(perms))
partitions = AbstractAlgebra.Generic.partitions(N)
projectors = map(partitions) do λ
    χ = AbstractAlgebra.character(λ)
    map!(Int ∘ χ, weights, G)
    symmetric_sector(H, factors(H), (perms, weights))
end
M = representation(ham, H)
map(partitions, projectors) do λ, p
    λ => size(p' * M * p)
end
# Each projector block-diagonalizes `M` into the corresponding symmetry sector.

# ### Total spin Casimir

# We can also project onto a fixed total spin using the Casimir operator $\vec{S}^2$.
# For example, the nullspace of $\vec{S}^2 - S(S+1)$ with $S=1$ gives the total-spin-1
# sector.
total_spin_op = sum(sum(S[k][op] for k in 1:N)^2 for op in (:x, :y, :z))
total_spin_mat = representation(total_spin_op, H)
P = nullspace(Matrix(total_spin_mat - 2 * I))
P' * M * P

# ### Fully decomposed space
# Let's find the ground state energies in each block of total spin, magnetization and permutation symmetry sector
magnetizations = (-N*J):(N*J)
energies = mapreduce(vcat, magnetizations) do m
    magcons = AdditiveConstraint(m; maps=s -> s.m)
    H = hilbert_space(S, 1:N, J, magcons)
    total_spin_mat = representation(total_spin_op, H)
    ham_mat = representation(ham, H)
    mapreduce(vcat, partitions) do λ
        χ = AbstractAlgebra.character(λ)
        map!(Int ∘ χ, weights, G)
        p = symmetric_sector(H, factors(H), (perms, weights))
        block = Matrix(p'*total_spin_mat*p)
        Mblock = p'*ham_mat*p
        total_spins = unique!(map(s -> round(Int, 4s)//4, eigvals(block)))
        map(total_spins) do s
            P = nullspace(block - s*I; atol=1e-1)
            (λ, s, m) => first(eigvals(Hermitian(P' * Mblock * P)))
        end
    end
end
sort(energies; by=last)