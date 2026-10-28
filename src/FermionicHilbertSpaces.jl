"""
    FermionicHilbertSpaces

Symbolic fermions, bosons and spins, Hilbert spaces with constraints, and the tensor
products, partial traces and matrix representations that connect them.

"""
module FermionicHilbertSpaces

using LinearAlgebra: LinearAlgebra, ColumnNorm, Diagonal, Eigen, Hermitian,
    Symmetric, UniformScaling, cholesky, diag, diagind, dot,
    eigen, eigen!, eigvals, eigvecs, ishermitian, issymmetric,
    mul!, normalize, qr, rank, rmul!, schur, svdvals, tr, I
using LowRankMatrices: LowRankMatrices, LowRankMatrix
import SciMLOperators
using FillArrays: Zeros, Fill, Eye
using OrderedCollections: OrderedDict
using BitPermutations: BitPermutations, AbstractBitPermutation, BitPermutation, bitpermute
using BitIntegers: UInt256, UInt512, UInt1024
using HalfIntegers: HalfIntegers, Half, HalfInt, HalfInteger, half, twice, ishalfinteger
using SparseArrays: SparseArrays, AbstractSparseMatrix, SparseVector, findnz,
    nnz, nonzeros, nzrange, rowvals, sparse, sprandn, spzeros, SparseMatrixCSC
using TestItems: TestItems, @testitem, @testmodule
import TupleTools
using NonCommutativeProducts: NonCommutativeProducts, @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, add!!


export FockNumber, hc, basisstates, dim, state_index, basisstate
export hilbert_space, subregion
export parityoperator, numberoperator, matrix_representation, representation, vector_representation
export Kets, Ket, ProductState, SpinState, BosonicState

export partial_trace, generalized_kron, tensor_product, embed, partial_transpose
export @fermions, @majoranas, @boson, @bosons, @spin, @spins
export BosonField, SpinField
export NoSymmetry, ParityConservation, NumberConservation, constrain_space
export SectorConstraint, FilterConstraint, GenericHilbertSpace
export SectorHilbertSpace, quantumnumbers, sector, sectors, indices, factors
export SingleParticle, BdG
export symmetric_sector

public AbstractSubspace, SubspaceIndex, subspace_states, subspace_index, rewrap, subspace_key, show_details

"""
    HC
Represents the Hermitian conjugate.
"""
struct HC end
Base.:+(m, ::HC) = (m + m')
Base.:-(m, ::HC) = (m - m')
"""
    hc
Adding this is equivalent to adding the hermitian conjugate.
"""
const hc = HC()

abstract type AbstractSym end

# Factors from different groups are ordered by the type name of the group id, then by `isless`.
# The type name depends only on the code, so the order is the same in every session and on every platform.
function _group_isless(a, b)
    ga, gb = symbolic_group(a), symbolic_group(b)
    na, nb = nameof(typeof(ga)), nameof(typeof(gb))
    na == nb ? isless(ga, gb) : isless(na, nb)
end

# The tags that identify a group of symbolic operators. `tags` is `nothing` or a tuple. For `@fermions` and
# `@majoranas` the first element is the tuple of names given to the macro, so the names determine the group.
# Elements appended by `add_tag` are symbols such as `:left` and `:right`.
# Tags are ordered by `isless` of their tag tuples. For groups created by the macros this is the lexicographic order
# of the macro's names, e.g. the operators of `@fermions a` sort before those of `@fermions b`.
struct Tags{T}
    tags::T
end
Base.hash(id::Tags, h::UInt) = hash(id.tags, h)
Base.:(==)(a::Tags, b::Tags) = a.tags == b.tags
Base.isless(a::Tags, b::Tags) = isless(tagset(a), tagset(b))
tagset(id::Tags) = id.tags
tagset(::Tags{Nothing}) = ()
tagset(x) = tagset(tags(x))

function _symbolic_name_with_tags(name, tagged; skip_first=0)
    tags = tagset(tagged)
    tags = skip_first > 0 ? tags[(skip_first+1):end] : tags
    isempty(tags) && return string(name)
    return string(name, "(", join(tags, ","), ")")
end

add_tag(t::Tags, tag) = Tags((t.tags..., tag))
add_tag(::Tags{Nothing}, tag) = Tags((tag,))

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

include("matrix_representation.jl")
include("lazy_matrix_representation.jl")

include("physics/fermions/fock.jl")
include("physics/fermions/phase_factors.jl")
include("physics/fermions/symbolic_fermions.jl")
include("physics/fermions/fermions.jl")
include("physics/fermions/operators.jl")
include("physics/fermions/majoranas.jl")
include("physics/fermions/fixednumberfock.jl")
include("physics/fermions/bdg.jl")


include("physics/bosons.jl")
include("physics/spin.jl")
include("physics/open_systems.jl")
include("symbolic_states.jl")

include("printing.jl")

function __init__()
    NonCommutativeProducts.enable_autosort!()
end


import PrecompileTools

PrecompileTools.@setup_workload begin
    m = rand(4, 4)
    NonCommutativeProducts.enable_autosort!()
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
    end
end

end
