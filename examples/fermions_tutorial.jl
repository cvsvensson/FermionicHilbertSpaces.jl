# # Fermions

# In this tutorial, we build a chain of spinless fermions step by step,
# use particle-number conservation to work in a smaller space,
# and find the ground state of the chain.
# Finally, we compute the reduced density matrix of a subsystem
# and see how fermionic partial traces sometimes differ from ordinary partial traces.

# ## Fermionic operators

# `@fermions c` declares a set of symbolic fermionic operators.
# For example, `c[1]` annihilates a fermion on site 1,
# and the adjoint `c[1]'` creates one.
# Let's begin by creating the set of fermions and checking that the operators anticommute.

using FermionicHilbertSpaces
@fermions c
c[1] * c[2] == -c[2] * c[1]

# A fermion cannot be created twice on the same site:

c[1]' * c[1]' == 0

# Products are put in a standard order. Swapping the two operators gives a minus sign:

c[2] * c[1]'

# ## A first matrix

# To turn operators into matrices we need a Hilbert space.
# `hilbert_space(c, 1:2)` is the space containing the fermionic modes `c[1]` and `c[2]`.
# Its basis states are written as occupations, so `10` has a fermion on site 1 and none on site 2:

H2 = hilbert_space(c, 1:2)
basisstates(H2)

# Let's now write down a simple hopping operator.
# Note the use of `hc` for adding the Hermitian conjugate.

hop = c[1]' * c[2] + hc

# `representation` turns the operator into a matrix in the basis of the
# Hilbert space. We ask for a dense `Matrix` so it prints nicely and works
# with LinearAlgebra. By default you get a sparse matrix, which is better for
# large systems.

representation(hop, H2, Matrix)

# The hopping only connects `10` and `01`: it moves a fermion, but the total number of fermions is conserved.

# ## Number conservation

# Now take a chain of $N$ sites, with hopping $t$ between neighbours and an interaction $V$ between fermions on
# neighbouring sites,
# ```math
# H = \sum_{j=1}^{N-1} \left[ -t\,(c_j^\dagger c_{j+1} + \mathrm{h.c.}) + V c_j^\dagger c_j c_{j+1}^\dagger c_{j+1} \right].
# ```
# We take six sites:

N = 6
t = 1.0
V = 4.0
ham = sum(-(t * c[j]' * c[j + 1] + hc) + V * c[j]' * c[j] * c[j + 1]' * c[j + 1] for j in 1:(N - 1))

# Like the two-site hopping, this Hamiltonian conserves the number of fermions.
# We can check this symbolically by evaluating the commutator:

Ntot = sum(c[j]' * c[j] for j in 1:N)
iszero(ham * Ntot - Ntot * ham)

# So we only need the states with a given number of fermions.
# With three fermions on six sites, providing `NumberConservation(3)` keeps 20 of the 64 states:

Hfull = hilbert_space(c, 1:N)
H = hilbert_space(c, 1:N, NumberConservation(3))
dim(Hfull), dim(H)

# We diagonalize the Hamiltonian in this space and keep the ground state:

using LinearAlgebra
vecs = eigvecs(Hermitian(representation(ham, H, Matrix)))
ψ = vecs[:, 1];

# ## The reduced density matrix

# What does the ground state look like on the left half of the chain alone?
# The partial trace answers this: it reduces `ψ` to a density matrix on a
# subsystem.
# To compute the reduced density matrix, we need another Hilbert space, representing the left half of the chain.
# We give the left half `NumberConservation()` without an argument,
# so it contains all particle numbers, with basis states sorted by how many fermions they hold:

HL = hilbert_space(c, 1:3, NumberConservation())
basisstates(HL)

# We can now compute the reduced density matrix of `ψ` on the left half. The second argument, `H => HL`, reads
# "from `H` to `HL`". We pass the state as the vector `ψ`, which `partial_trace` treats as the pure state `ψψ†`,
# and round the result to make it easier to read:

ρL = partial_trace(ψ, H => HL)
round.(ρL; digits = 2)

# The density matrix is block diagonal: it has no coherences between different
# numbers of fermions on the left, because the whole chain has a fixed number of fermions.
# The diagonal of `ρL` holds the probability of each basis state. Summing it over the states with `n` fermions
# gives the probability of finding `n` fermions on the left. Since we provided `NumberConservation()` when
# creating `HL`, `quantumnumbers(HL)` lists the particle numbers it contains, and `indices(n, HL)` gives the
# positions of the states with `n` fermions:

p = diag(ρL)
[n => sum(p[indices(n, HL)]) for n in quantumnumbers(HL)]

# More than one number has nonzero probability, so the particle number on the
# left half fluctuates. For a state with a fixed total number, that can only
# happen if the two halves are entangled.

# ## Why the partial trace is fermionic

# A reduced density matrix should give the same expectation values as the full state for operators that act only on the subsystem.
# Let us check this for hopping between sites 1 and 3, using `expectation` to calculate expectation values.

H13 = hilbert_space(c, [1, 3])
ρ13 = partial_trace(ψ, H => H13)
hop13 = c[1]' * c[3] + hc
expectation(hop13, ψ, H), expectation(hop13, ρ13, H13)

# The two agree. This is not automatic: a fermion on site 2, between the two sites, changes the sign of some
# terms. The ordinary partial trace (which `phase_factors = false` gives) ignores these signs and gets an
# answer about four times too large:

ρ13_naive = partial_trace(ψ, H => H13; phase_factors = false)
expectation(hop13, ρ13_naive, H13)

# Every function in the package that combines or splits fermionic spaces includes these signs. Why they appear
# is explained in [Fermionic tensor products, partial traces, and partial transpose](../fermions.md).

# ## Next steps

# - [Constrained spaces and quantum numbers](../conservation.md): parity, several conserved quantities and
#   custom constraints.
# - [Hilbert space and array operations](../hilbert_space_operations.md): tensor products, embeddings and sectors.
# - [Entanglement and negativity](partial_transpose.md): the fermionic partial transpose.
# - [Non-interacting systems](../non_interacting.md): methods for Hamiltonians without interactions.
# - The [Bosons](bosons.md) and [Spins](spin_chain.md) tutorials, which do the same for other particle types.
