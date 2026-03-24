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
export parityoperator, numberoperator, matrix_representation

export partial_trace, generalized_kron, tensor_product, embed
export @fermions, @majoranas, @boson, @spin, @spins
export NoSymmetry, ParityConservation, NumberConservation, constrain_space
export BlockHilbertSpace, quantumnumbers, sector, sectors, indices
export majorana_hilbert_space, single_particle_hilbert_space, bdg_hilbert_space

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
include("spaces.jl")
include("state_splitter.jl")
include("hilbert_space.jl")
include("sectors.jl")
include("product_space.jl")
include("tensor_product.jl")

include("embedding.jl")
include("reshape.jl")

include("constraints.jl")
include("constrained_space.jl")
include("generate_constrained_states.jl")

include("matrix_representation.jl")

include("physics/fermions/fock.jl")
include("physics/fermions/phase_factors.jl")
include("physics/fermions/symbolic_fermions.jl")
include("physics/fermions/fermions.jl")
include("physics/fermions/fixednumberfock.jl")
include("physics/fermions/operators.jl")


include("physics/majoranas.jl")
include("physics/bosons.jl")
include("physics/spin.jl")
include("physics/bdg.jl")

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
