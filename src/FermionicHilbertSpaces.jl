module FermionicHilbertSpaces

using LinearAlgebra, SparseArrays
using SplitApplyCombine: group, sortkeys!
using Dictionaries: dictionary, Dictionary
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

export partial_trace, fermionic_kron, tensor_product, embed, extend
export @fermions, @majoranas
export NumberConservation, NoSymmetry, ParityConservation, IndexConservation
export majorana_hilbert_space, single_particle_hilbert_space, bdg_hilbert_space

## Symbolics extension
"""
    fermion_to_majorana(expr)
Convert symbolic fermions to symbolic majoranas.
"""
function fermion_to_majorana end
"""
    majorana_to_fermion(expr)
Convert symbolic majoranas to symbolic fermions.
"""
function majorana_to_fermion end

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
include("operators.jl")
include("tensor_product.jl")
include("embedding.jl")
include("reshape.jl")

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
