module FermionicHilbertSpaces

using LinearAlgebra, SparseArrays
import SciMLOperators
import FillArrays: Zeros, Fill
import OrderedCollections: OrderedDict
using TestItems
using BitPermutations
using TupleTools
using NonCommutativeProducts
import NonCommutativeProducts: @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, add!!


export FockNumber, hc, basisstates, dim, state_index, basisstate
export hilbert_space, subregion
export parityoperator, numberoperator, matrix_representation, representation, vector_representation
export Kets

export partial_trace, generalized_kron, tensor_product, embed
export @fermions, @majoranas, @boson, @bosons, @spin, @spins
export BosonField, SpinField
export NoSymmetry, ParityConservation, NumberConservation, constrain_space
export SectorHilbertSpace, quantumnumbers, sector, sectors, indices, factors
export majorana_hilbert_space, single_particle_hilbert_space, bdg_hilbert_space
export symmetric_sector

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


struct Tags{T}
    tags::T
end
Base.hash(id::Tags, h::UInt) = hash(id.tags, h)
Base.:(==)(a::Tags, b::Tags) = a.tags == b.tags
Base.isless(a::Tags, b::Tags) = isless(hash(a.tags), hash(b.tags))
tagset(id::Tags) = id.tags
tagset(::Tags{Nothing}) = ()
tagset(x) = tagset(tags(x))

function _symbolic_name_with_tags(name, tagged; skip_first = 0)
    tags = tagset(tagged)
    tags = skip_first > 0 ? tags[skip_first+1:end] : tags
    isempty(tags) && return string(name)
    return string(name, "(", join(tags, ","), ")")
end

add_tag(t::Tags, tag) = Tags((t.tags..., tag))
add_tag(::Tags{Nothing}, tag) = Tags((tag,))

struct TypedIterator{T,I}
    iter::I
end

# Constructor with explicit type
TypedIterator{T}(iter) where T = TypedIterator{T,typeof(iter)}(iter)
# Iterator interface
Base.iterate(ti::TypedIterator) = iterate(ti.iter)
Base.iterate(ti::TypedIterator, state) = iterate(ti.iter, state)
Base.IteratorSize(::Type{<:TypedIterator{T,I}}) where {T,I} = Base.IteratorSize(I)
Base.IteratorEltype(::Type{<:TypedIterator}) = Base.HasEltype()
Base.eltype(::Type{<:TypedIterator{T}}) where T = T
Base.length(ti::TypedIterator) = length(ti.iter)
Base.size(ti::TypedIterator) = size(ti.iter)
Base.keys(ti::TypedIterator) = keys(ti.iter)

function _resolve_sector_permutations_and_weights end # for Combinatorics extension

## Files
include("spaces.jl")
include("hilbert_space.jl")
include("sectors.jl")
include("product_space.jl")

include("constraints.jl")
include("constrained_space.jl")

include("tensor_product.jl")
include("permutation_symmetry.jl")
include("embedding.jl")
include("reshape.jl")

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

PrecompileTools.@compile_workload begin
    m = rand(4, 4)
    PrecompileTools.@compile_workload begin
        @fermions f
        H1 = hilbert_space(f, 1:2)
        H2 = hilbert_space(f, 3:3, ParityConservation())
        subregion([f[3]], H2)
        partial_trace(m + hc, H1 => hilbert_space(f, 1:1, NumberConservation()))
        H = tensor_product(H1, H2)
        c = matrix_representation(f[1], H1)
        embed(c, H1 => H)
        matrix_representation((f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
        @majoranas γ
        (γ[1] * γ[2] + 1.0 + γ[1])^2
        @boson b
        @spin s 1 // 2
        Hb = hilbert_space(b, 2)
        Hs = hilbert_space(s)
        H = tensor_product(H2, Hb, Hs)
        mat = matrix_representation(b'b + s[:x] * f[3], H)
    end
end

end
