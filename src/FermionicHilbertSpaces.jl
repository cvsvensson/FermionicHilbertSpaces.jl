module FermionicHilbertSpaces

using LinearAlgebra, SparseArrays
using FlexiGroups: group
import Dictionaries: dictionary, Dictionary, sortkeys!, getindices
import FillArrays: Zeros, Fill
import OrderedCollections: OrderedDict
using TestItems
using BitPermutations
using TupleTools
using NonCommutativeProducts
import NonCommutativeProducts: @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, add!!
import UUIDs: uuid4


export FockNumber, JordanWignerOrdering, hc, basisstates, dim
export hilbert_space, subregion
export parityoperator, numberoperator, fermions, majoranas, matrix_representation

export partial_trace, generalized_kron, tensor_product, embed
export @fermions, @majoranas, @boson, @spin, @spins
export NoSymmetry, ParityConservation, NumberConservation, constrain_space
export majorana_hilbert_space, single_particle_hilbert_space, bdg_hilbert_space

## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractFockHilbertSpace{F<:AbstractFockState} <: AbstractHilbertSpace{F} end
abstract type AbstractAtomicHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractProductHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractClusterHilbertSpace{B} <: AbstractProductHilbertSpace{B} end
abstract type AbstractFermionicClusterHilbertSpace{B} <: AbstractClusterHilbertSpace{B} end
atomic_factors(H::AbstractAtomicHilbertSpace) = (H,)
factors(H::AbstractClusterHilbertSpace) = atomic_factors(H)
factors(H::AbstractAtomicHilbertSpace) = (H,)

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

## Files
include("fock.jl")
include("phase_factors.jl")
# include("symmetry.jl")
include("hilbert_space.jl")
include("generate_constrained_states.jl")
include("kron_product_space.jl")
include("constrained_space.jl")

include("fermions.jl")

include("operators.jl")
include("tensor_product.jl")
include("embedding.jl")
include("reshape.jl")

include("symbolics/muladd.jl")
include("symbolics/symbolic_fermions.jl")

include("fixednumberfock.jl")

include("majoranas.jl")
include("bosons.jl")
include("spin.jl")
include("qubit.jl")

include("bdg.jl")

# include("sectors.jl")



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
        partial_trace(m + hc, H1 => hilbert_space(f, 1:1, NumberConservation()))
        H = tensor_product(H1, H2)
        c = matrix_representation(f[1], H1)
        embed(c, H1 => H)
        matrix_representation((f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
        @majoranas γ
        (γ[1] * γ[2] + 1.0 + γ[1])^2
    end
end

end
