using FermionicHilbertSpaces
using SymPy, LinearAlgebra, Combinatorics

# We construct a two-site Hubbard model (e.g., a double quantum dot) 
# with symbolic parameters U (interaction) and t (hopping amplitude),
# and restrict to the sector with one spin-up and one spin-down fermion (S_z = 0).

@syms U::real, t::real
@fermions f

# Define spinful Hilbert space as tensor product of spin-up and spin-down sectors
Hup = hilbert_space(f, [(j, :↑) for j in 1:2])
Hdn = hilbert_space(f, [(j, :↓) for j in 1:2])

# Restrict to N↑ = 1 and N↓ = 1
H = tensor_product(Hup, Hdn;
    constraint=NumberConservation(1, Hup) * NumberConservation(1, Hdn)
)

# Define the Hubbard Hamiltonian
ham = sum(-t * (f[1, s]' * f[2, s] + hc) for s in (:↑, :↓)) +
      U * sum(f[j, :↑]' * f[j, :↑] * f[j, :↓]' * f[j, :↓] for j in 1:2)

# Construct matrix representation in the constrained Hilbert space
M = Matrix(matrix_representation(ham, H))

# --- Revealing structure via a basis transformation ---
#
# The Hamiltonian is invariant under exchange of the two sites. By transforming
# to symmetric and antisymmetric combinations of basis states, we align the basis
# with this symmetry. This block-diagonalizes the Hamiltonian.
Hleft = subregion([f[1, :↑], f[1, :↓]], H)
Hright = subregion([f[2, :↑], f[2, :↓]], H)
Peven = symmetric_sector(H, [Hleft, Hright], :symmetric, Sym)
Podd = symmetric_sector(H, [Hleft, Hright], :antisymmetric, Sym)
Meven = Peven' * M * Peven

# The off-diagonal block should vanish
Peven' * M * Podd

# Anti-symmetric sector
Modd = Podd' * M * Podd

# Diagonalize symbolically using SymPy
vals_odd, vecs_odd = eigen(Modd)
vals_even, vecs_even = eigen(Meven)

# --- Reduced density matrix ---
#
# As an example of post-processing, we compute the reduced density matrix
# of the first site by tracing out the second site. This provides access
# to local observables and entanglement properties. 
ψ = Peven * vecs_even[:, 1] # map back to original basis
ρ = ψ * ψ'
ρ /= tr(ρ)  # normalize, since sympy eigenvectors are not guaranteed to be normalized
ρ_sub = partial_trace(ρ, H => Hleft)
simplify(tr(ρ_sub))
purity = simplify(tr(ρ_sub^2))
