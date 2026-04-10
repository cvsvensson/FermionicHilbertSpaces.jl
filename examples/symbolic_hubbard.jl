using FermionicHilbertSpaces
using SymPy, LinearAlgebra

# We construct a two-site Hubbard model (e.g., a double quantum dot) 
# with symbolic parameters U (interaction) and t (hopping amplitude),
# and restrict to the sector with one spin-up and one spin-down fermion (S_z = 0).

@syms U, t
@fermions f

# Define spinful Hilbert space as tensor product of spin-up and spin-down sectors
Hup = hilbert_space(f, [(j, :↑) for j in 1:2])
Hdn = hilbert_space(f, [(j, :↓) for j in 1:2])

# Restrict to N↑ = 1 and N↓ = 1
H = tensor_product(Hup, Hdn;
    constraint = NumberConservation(1, Hup) * NumberConservation(1, Hdn)
)

# Define the Hubbard Hamiltonian
ham = sum(-t * (f[1, s]' * f[2, s] + hc) for s in (:↑, :↓)) +
      U * sum(f[j, :↑]' * f[j, :↑] * f[j, :↓]' * f[j, :↓] for j in 1:2)

# Construct matrix representation in the constrained Hilbert space
M = Matrix(matrix_representation(ham, H))

# Inspect basis ordering (Fock states)
# Basis: |↑↓,0⟩, |↑,↓⟩, |↓,↑⟩, |0,↑↓⟩
basisstates(H)
FermionicHilbertSpaces.mode_ordering(H)

# Diagonalize symbolically using SymPy
vals, vecs = eigen(M)
simplify.(vecs)

# --- Revealing structure via a basis transformation ---
#
# The Hamiltonian is invariant under exchange of the two sites. By transforming
# to symmetric and antisymmetric combinations of basis states, we align the basis
# with this symmetry. This block-diagonalizes the Hamiltonian.
#
# Such symmetry sectors are not diagonal in the Fock basis and 
# cannot be constructed automatically by the package. They can nevertheless be
# accessed via explicit basis transformations.

sqrt2 = sympy.sqrt(2)
T = [
    1/sqrt2   0         0         1/sqrt2   # symmetric doublon
    0         1/sqrt2   1/sqrt2   0         # symmetric spin
    0         1/sqrt2  -1/sqrt2   0         # antisymmetric spin
    1/sqrt2   0         0        -1/sqrt2   # antisymmetric doublon
]

# Transform Hamiltonian
T' * M * T
