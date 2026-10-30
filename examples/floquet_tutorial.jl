# # Floquet Systems

# This example defines an operator algebra and matrix representations for the Floquet space of a
# periodically driven system, and couples it to a spin-1 to compute the quasienergies of a driven three-level system.

# ## Background: Floquet theory 

# A quantum system with a time-periodic Hamiltonian ``H(t) = H(t+T)`` can be reduced to
# a static eigenvalue problem by embedding it into a Floquet-Hilbert space
# ``\mathcal{H}_\text{phys} \otimes \mathcal{H}_\text{floquet}``.
# The second factor carries the Fourier harmonics of the drive.
# On the Floquet space we have the operators:
# - the **Floquet ladder** ``F`` with ``F|n\rangle = |n{+}1\rangle`` (absorption of one drive photon),
# - the **Floquet number** ``N`` with ``N|n\rangle = n|n\rangle``.
#
# For a periodically driven two-level system
# ```math
# H(t) = \tfrac{\varepsilon}{2}\,\sigma_z + A\cos(\omega t)\,\sigma_x
# ```
# the Floquet Hamiltonian is
# ```math
# H_F = \tfrac{\varepsilon}{2}\,\sigma_z + \omega N + \tfrac{A}{2}\,\sigma_x(F + F^\dagger).
# ```
# Its eigenvalues are the *quasienergies* of the driven system;
# because ``\mathcal{H}_\text{floquet}`` is infinite-dimensional we truncate to
# ``|n| \le n_\text{max}``.

# ## Defining the custom Floquet algebra

# The required pieces are listed in [Extending and performance hooks](../extending.md). Compared to the
# [Clock operators](clock_operators.md), the Floquet operators also commute with operators of other
# algebras through `@commutative`, and their `mul_effect` simplifies products.

# Let's load the package and import some things we need to overload or use
using FermionicHilbertSpaces
using FermionicHilbertSpaces: AbstractSym
import FermionicHilbertSpaces.NonCommutativeProducts:
    @nc, NCMul, @commutative, mul_effect

# ### Symbolic types

# `FloquetBasis` represents a Floquet degree of freedom (having an `id` allows multiple Floquet modes).
# `FloquetLadder` represents ``F^k`` and `FloquetNumber` represents ``N^p``.
struct FloquetBasis
    id::Int
end
struct FloquetLadder <: AbstractSym
    shift::Int        # +1 → F (raising), −1 → F† (lowering), k → F^k
    basis::FloquetBasis
end
struct FloquetNumber <: AbstractSym
    power::Int        # exponent p in N^p
    basis::FloquetBasis
end
FloquetLadder(b::FloquetBasis) = FloquetLadder(1, b)
Base.adjoint(F::FloquetLadder) = FloquetLadder(-F.shift, F.basis)
Base.adjoint(N::FloquetNumber) = N          # N is Hermitian
Base.show(io::IO, F::FloquetLadder) = print(io, "F^", F.shift)
Base.show(io::IO, N::FloquetNumber) = print(io, "N^", N.power)
const Floquets = Union{FloquetLadder,FloquetNumber}

# ### `@nc` declaration

# Registering types with `@nc` enables symbolic algebra operations (`+`, `*`)
# on these types and marks their multiplication as noncommutative.

@nc FloquetLadder FloquetNumber

# ### `@commutative` declarations

# To make Floquet operators commute with all other AbstractSyms operators we write

@commutative Floquets AbstractSym

# ### Multiplication rules via `mul_effect`

# `mul_effect(a, b)` is called when two adjacent factors `a·b` appear in a symbolic
# product and determines how the pair should be rewritten:
# - `nothing`               → already in canonical order; keep as-is (you need this to avoid an infinite loop)
# - `Swap(c)`               → reorder as `c·b·a` (exchange with coefficient `c`)
# - `AddTerms((Swap(c), r, ...))`→ reorder and add: `a·b = c·b·a + r + ...` 
#
# We will sort Floquet operators from different modes by their 'id', and for operators of the same mode
# we will put number operators before ladder operators.

function mul_effect(a::Floquets, b::Floquets)
    a.basis.id > b.basis.id && return b * a   # different subsystem: sort by id
    a.basis.id < b.basis.id && return nothing   # already in order
    return floquet_mul(a, b)                    # same subsystem: apply algebra
end

# ``F^j \cdot F^k = F^{j+k}``.  Special case ``j+k=0``: ``F \cdot F^\dagger = \mathbf{1}``.
function floquet_mul(a::FloquetLadder, b::FloquetLadder)
    total_shift = a.shift + b.shift
    total_shift == 0 && return 1
    return FloquetLadder(total_shift, a.basis)
end

# ``N^p \cdot N^q = N^{p+q}``.  Special case ``p=q=0``: ``N^0 = \mathbf{1}``.
function floquet_mul(a::FloquetNumber, b::FloquetNumber)
    a.power == b.power == 0 && return 1
    return FloquetNumber(a.power + b.power, a.basis)
end

# ``F^k \cdot N = N \cdot F^k - k\,F^k``, from ``[N, F^k] = k\,F^k`` rearranged as
# ``F^k N = N F^k - k F^k``.
function floquet_mul(a::FloquetLadder, b::FloquetNumber)
    if b.power == 1
        return b * a - a.shift * a         # Base case: F^k * N = N * F^k - k * F^k
    else
        N_remaining = FloquetNumber(b.power - 1, b.basis)
        term1 = FloquetNumber(1, b.basis) * a * N_remaining   # N * F^k * N^(p-1)
        term2 = -a.shift * a * N_remaining                    # -k * F^k * N^(p-1)
        return term1 + term2
    end
    throw(ArgumentError("Should not reach here"))
end

# ``N \cdot F^k`` is already in normal order (``N`` sorts before ``F``).
floquet_mul(::FloquetNumber, ::FloquetLadder) = nothing

# ### `apply_local_operator`

# This is the bridge between symbolic operators and matrix elements.
# Given an `NCMul` of Floquet factors and a single basis state, it returns
# the resulting state(s) and amplitude(s).
#
# Let's make a type for the Floquet basis states 
using FermionicHilbertSpaces: AbstractBasisState
struct FloquetState <: AbstractBasisState
    mode::Int
end
# To hook it up to the matrix-representation machinery, we need to define `apply_local_operator(op, state::FloquetState, space, precomp)
import FermionicHilbertSpaces: apply_local_operator
# ``F^k`` shifts the Floquet index by ``k``.
apply_local_operator(op::FloquetLadder, state::FloquetState, space, precomp) = FloquetState(state.mode + op.shift), 1

# ``N^p`` multiplies the amplitude by ``n^p``, where ``n`` is the current mode index.
apply_local_operator(op::FloquetNumber, state::FloquetState, space, precomp) = state, state.mode^op.power

# ### `symbolic_group`

# The matrix-representation machinery uses `symbolic_group` to match operators to Hilbert spaces.
# We'll associate each Floquet operator with a FloquetBasis and use that object as the label for the Hilbert space
import FermionicHilbertSpaces: symbolic_group
symbolic_group(f::Floquets) = f.basis
symbolic_group(b::FloquetBasis) = b

# ## Building the Hilbert space

# Now we can make a Hilbert space and get matrix representations of the Floquet operators.  
floquet_basis = FloquetBasis(0)
F = FloquetLadder(floquet_basis)       # raising operator F
Nf = FloquetNumber(1, floquet_basis)    # photon number N
n_max = 5       # keep Floquet modes n = −n_max … n_max
Hfloq = GenericHilbertSpace(floquet_basis, FloquetState.((-n_max):n_max))
representation(F, Hfloq; projection=true) # we need projection=true because we've truncated the Floquet space

# Now Floquet operators will work together algebraically with the spin operators, and we can get matrix representations of such mixed operators. Let's do some physics.
# ## Driven spin 1
# We take a spin 1 as the physical system and couple it to the Floquet space to model a driven three-level system.  The full Hilbert space is a product of the spin and Floquet spaces, and the Hamiltonian is a symbolic expression mixing spin and Floquet operators.
@spin σ
Hspin = hilbert_space(σ, 1)
H = tensor_product((Hspin, Hfloq))

# Floquet operators can be used together with spin operators to build the Floquet Hamiltonian, with an added anisotropy ``D\sigma_z^2`` (since we used @commutative to declare that they commute):

ω = 1.0         # Drive frequency (energy unit)
ε = 0.2 * ω     # Zeeman splitting: sets |+1⟩ and |-1⟩ baseline energies
D = 0.3 * ω     # Single-ion anisotropy (D * Sz^2): activates the |0⟩ state
floquet_ham(A) = ε * σ[:z] + D * σ[:z]^2 + ω * Nf +
    (A / 2 * σ[:x] * F + hc)
floquet_ham(2)

# ## Quasienergy spectrum vs drive amplitude

# Without a drive the quasienergies are the spin energies shifted by integer multiples of ``\omega``.
# We check this at ``A=0`` and then sweep the drive amplitude.
using LinearAlgebra
spin_energies = [ε * m + D * m^2 for m in -1:1]
bare = sort(vec([e + ω * n for e in spin_energies, n in (-n_max):n_max]))
mat0 = representation(floquet_ham(0), H, :dense; projection = true)
@assert eigvals(Hermitian(mat0)) ≈ bare

A_values = range(0, 4 * ω, length=100)
spectrum = reduce(hcat, [eigvals(Hermitian(representation(floquet_ham(A), H, :dense; projection = true))) for A in A_values])'

# Each dot is a quasienergy, folded into the interval ``[-\omega/2, \omega/2)``.
using Plots
plot(
    A_values ./ ω, mod.(spectrum .+ ω / 2, ω) .- ω / 2; seriestype = :scatter, ms = 1, legend = false,
    xlabel = "Drive amplitude A/ω", ylabel = "Quasienergy", frame = :box
)
