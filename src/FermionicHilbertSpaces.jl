"""
    FermionicHilbertSpaces

Symbolic fermions, bosons and spins, Hilbert spaces with constraints, and the tensor
products, partial traces and matrix representations that connect them.

"""
module FermionicHilbertSpaces

using LinearAlgebra: LinearAlgebra, ColumnNorm, Diagonal, Hermitian,
    UniformScaling, diag, diagind, dot,
    eigen!, eigvals, eigvecs, isdiag, ishermitian, issymmetric,
    mul!, norm, normalize, nullspace, qr, rank, rmul!, svd, svdvals, tr, I
using LowRankMatrices: LowRankMatrices, LowRankMatrix
import SciMLOperators
using FillArrays: Zeros, Fill, Eye
using OrderedCollections: OrderedDict
using BitPermutations: BitPermutations, AbstractBitPermutation, BitPermutation, bitpermute
using BitIntegers: UInt256, UInt512, UInt1024, @define_integers
# Widest first, so that `widen` of each type finds the next wider one.
@define_integers 8192
@define_integers 4096
@define_integers 2048
using HalfIntegers: HalfIntegers, Half, HalfInt, HalfInteger, half, twice, ishalfinteger
using SparseArrays: SparseArrays, AbstractSparseArray, AbstractSparseMatrix, SparseVector, findnz,
    nnz, nonzeros, nzrange, rowvals, sparse, sprandn, spzeros, SparseMatrixCSC
using TestItems: TestItems, @testitem, @testmodule
import TupleTools
using NonCommutativeProducts: NonCommutativeProducts, @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, add!!


export FockNumber, hc, basisstates, dim, state_index, basisstate
export hilbert_space, subregion
export parityoperator, numberoperator, representation, expectation
export Kets, Ket, ProductState, SpinState, BosonicState

export partial_trace, generalized_kron, tensor_product, embed, partial_transpose
export @fermion, @fermions, @majoranas, @boson, @bosons, @spin, @spins
export NoConstraint, ParityConservation, NumberConservation, constrain_space
export BranchConstraint, GenericHilbertSpace
export SectorHilbertSpace, quantumnumbers, sector, sectors, indices, factors
export symmetric_sector
export open_system
public particle_number, Axes, dissipator, FreeFermions, NonCommutativeProducts, add!!, matrix_representation, vector_representation

"""
    add!!(a, b)

Return `a + b`, reusing the storage of `a` when possible, so use the returned value. `foldl(add!!, terms; init=0)` sums
many operators faster than `sum`.
"""
add!!

"""
    HC
Represents the Hermitian conjugate.
"""
struct HC end
Base.:+(m, ::HC) = (m + m')
Base.:-(m, ::HC) = (m - m')
"""
    hc
`op + hc` is `op + op'` and `op - hc` is `op - op'`, for example `f[1]'f[2] + hc`.
"""
const hc = HC()

"""
    AbstractSym

Supertype of the factors of symbolic expressions: operators on fields, which are [`Op`](@ref)s, and symbolic
states.
"""
abstract type AbstractSym end
NonCommutativeProducts.atom_group(op::AbstractSym) = symbolic_group(op)

# The order of labels, tags and groups. Values are ranked by kind, and those of one kind are compared with `isless`.
# Kinds depend only on the code, so the order is the same in every session.
# Numbers and strings of different types are one kind, so that they are ordered consistently with `==`, as `1 == 1.0`.
# Tuples are one kind and are compared element by element, as `isless` may not compare their elements.
_order_kind(x) = nameof(typeof(x))
_order_kind(::Real) = :Real
_order_kind(::AbstractString) = :AbstractString
_order_kind(::Tuple) = :Tuple
function _order_isless(a, b)
    ka, kb = _order_kind(a), _order_kind(b)
    ka == kb || return isless(ka, kb)
    return !isequal(a, b) && isless(a, b)
end
_group_isless(a, b) = _order_isless(symbolic_group(a), symbolic_group(b))
_order_isless(::Tuple{}, ::Tuple{}) = false
_order_isless(::Tuple{}, ::Tuple) = true
_order_isless(::Tuple, ::Tuple{}) = false
function _order_isless(a::Tuple, b::Tuple)
    x, y = first(a), first(b)
    _order_isless(x, y) && return true
    _order_isless(y, x) && return false
    return _order_isless(Base.tail(a), Base.tail(b))
end

# The tags of a field, `nothing` or a tuple of the tags appended by `add_tag`, such as `:left` and `:right`.
struct Tags{T}
    tags::T
end
Base.hash(id::Tags, h::UInt) = hash(id.tags, h)
Base.hash(::Tags{Nothing}, h::UInt) = h
Base.:(==)(a::Tags, b::Tags) = a.tags == b.tags
Base.isless(a::Tags, b::Tags) = _order_isless(tagset(a), tagset(b))
tagset(id::Tags) = id.tags
tagset(::Tags{Nothing}) = ()
tagset(x) = tagset(tags(x))

function _symbolic_name_with_tags(name, tagged)
    tags = tagset(tagged)
    isempty(tags) && return string(name)
    return string(name, "(", join(tags, ","), ")")
end

add_tag(t::Tags, tag) = Tags((t.tags..., tag))
add_tag(::Tags{Nothing}, tag) = Tags((tag,))

"""
    tag_label(label, tag)

Return the group label `label` with `tag` appended. Any label works, e.g. `1` or `:a`.
"""
tag_label(label, tag) = Tags((label, tag))
tag_label(label::Tags, tag) = add_tag(label, tag)

function _resolve_sector_permutations_and_weights end # for Combinatorics extension
function cache_operator! end # for OhMyThreads extension

## Files
include("spaces.jl")
include("hilbert_space.jl")
include("subspace.jl")
include("sectors.jl")
include("product_space.jl")

include("constraints.jl")
include("constrained_space.jl")

include("tensor_product.jl")
include("permutation_symmetry.jl")
include("embedding.jl")
include("reshape.jl")
include("partial_transpose.jl")

include("generate_constrained_states.jl")
include("operator_constraint.jl")
include("joint_bounds.jl")

include("matrix_representation.jl")
include("sparse_assembly.jl")
include("lazy_matrix_representation.jl")

include("fields.jl")
include("physics/fermions/fock.jl")
include("physics/fermions/phase_factors.jl")
include("physics/fermions/symbolic_fermions.jl")
include("physics/fermions/fermions.jl")
include("physics/fermions/operators.jl")
include("physics/fermions/majoranas.jl")
include("physics/fermions/fixednumberfock.jl")


include("physics/bosons.jl")
include("physics/spin.jl")
include("physics/open_systems.jl")
include("symbolic_states.jl")
include("symmetric_sector.jl")

include("printing.jl")
include("physics/fermions/free_fermions.jl")

import PrecompileTools

PrecompileTools.@setup_workload begin
    m = rand(4, 4)
    PrecompileTools.@compile_workload begin
        @fermions f
        sprint(show, f[1])
        H1 = hilbert_space(f, 1:2)
        H2 = hilbert_space(f, 3:3, ParityConservation())
        subregion([f[3]], H2)
        partial_trace(m + hc, H1 => hilbert_space(f, 1:1, NumberConservation()))
        H = tensor_product(H1, H2)
        c = representation(f[1], H1)
        embed(c, H1 => H)
        representation((1.0*f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
        @boson b
        sprint(show, b)
        @spin s 1 // 2
        sprint(show, s[:z])
        Hb = hilbert_space(b, 2)
        Hs = hilbert_space(s)
        H = tensor_product(H2, Hb, Hs)
        mat = representation(b'b + 1.0 * s[:x] * f[3], H)
        # Small chains, with fermions covering each common coefficient type.
        hamf = sum(f[n + 1]'f[n] + hc for n in 1:2) + sum(f[n]'f[n] * f[n + 1]'f[n + 1] for n in 1:2)
        Hf = hilbert_space(f, 1:3, NumberConservation())
        representation(hamf, Hf)
        representation(hamf, sector(1, Hf), Matrix)
        hamf = sum(0.5 * f[n + 1]'f[n] + hc for n in 1:2) + sum(0.5 * f[n]'f[n] * f[n + 1]'f[n + 1] for n in 1:2)
        representation(hamf, Hf)
        representation(hamf, sector(1, Hf), Matrix)
        hamf = sum((0.5 + 0.5im) * f[n + 1]'f[n] + hc for n in 1:2) + sum((0.5 + 0.5im) * f[n]'f[n] * f[n + 1]'f[n + 1] for n in 1:2)
        representation(hamf, Hf)
        representation(hamf, sector(1, Hf), Matrix)
        @spins S 1 // 2
        representation(sum(S[k][:]'S[k + 1][:] for k in 1:2), hilbert_space(S, 1:3))
        hilbert_space(f, 1:3, (sum(f[n]'f[n] for n in 1:3) => 1, sum(n * f[n]'f[n] for n in 1:3)))
        hilbert_space(S, 1:3, sum(S[k][:z] for k in 1:3) => 1 // 2)
        @bosons bs
        hamb = sum(bs[k]'bs[k + 1] + hc for k in 1:1) + sum(bs[k]'bs[k] * bs[k]'bs[k] for k in 1:2)
        representation(hamb, hilbert_space(bs, 1:2, 3))
        # Reduced states of a state vector in a constrained space
        partial_trace(rand(dim(Hf)), Hf => subregion([f[1]], Hf))
        Hn = hilbert_space(f, 1:4, NumberConservation(2))
        partial_trace(rand(dim(Hn)), Hn => subregion([f[1], f[2]], Hn))
    end
end

end
