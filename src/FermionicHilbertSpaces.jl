module FermionicHilbertSpaces

using LinearAlgebra, SparseArrays
using FlexiGroups: group
import Dictionaries: dictionary, Dictionary, sortkeys!, getindices
import FillArrays: Zeros
import OrderedCollections: OrderedDict
using TestItems
using BitPermutations
using TupleTools
using NonCommutativeProducts
import NonCommutativeProducts: @nc_eager, Swap, NCAdd, NCMul, NCterms, AddTerms, add!!
import UUIDs: uuid4


export FockNumber, JordanWignerOrdering, hc, basisstates, dim
export FockHilbertSpace, SymmetricFockHilbertSpace, SimpleFockHilbertSpace, hilbert_space, subregion
export parityoperator, numberoperator, fermions, majoranas, matrix_representation

export partial_trace, generalized_kron, tensor_product, embed, extend
export @fermions, @majoranas
export number_conservation, NoSymmetry, ParityConservation, NumberConservation
export majorana_hilbert_space, single_particle_hilbert_space, bdg_hilbert_space

## Some types
abstract type AbstractHilbertSpace end
abstract type AbstractFockHilbertSpace <: AbstractHilbertSpace end
abstract type AbstractBasisState end

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
include("fixednumberfock.jl")
include("phase_factors.jl")
include("symmetry.jl")
include("hilbert_space.jl")
include("product_space.jl")
include("operators.jl")
include("tensor_product.jl")
include("embedding.jl")
include("reshape.jl")
include("generate_constrained_states.jl")


include("qubit.jl")

include("symbolics/muladd.jl")
include("symbolics/symbolic_fermions.jl")
include("symbolics/symbolic_majoranas.jl")

include("majorana_hilbert_space.jl")
include("bdg.jl")

include("sectors.jl")

include("spin.jl")


import PrecompileTools

PrecompileTools.@compile_workload begin
    m = rand(4, 4)
    PrecompileTools.@compile_workload begin
        H1 = hilbert_space(1:2)
        H2 = hilbert_space(3:3, ParityConservation())
        c1 = fermions(H1)
        c2 = fermions(H2)
        partial_trace(m + hc, H1 => hilbert_space(1:1))
        H = tensor_product(H1, H2)
        extend(c1[1], H1 => H2)
        embed(c1[1], H1 => H)
        @fermions f
        FermionicHilbertSpaces.eval_in_basis((f[1] * f[2]' + 1 + f[1])^2 * 2.0 + hc, c1)
        matrix_representation((f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
        @majoranas γ
        (γ[1] * γ[2] + 1.0 + γ[1])^2
    end
end

end
