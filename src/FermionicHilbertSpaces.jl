module FermionicHilbertSpaces

using LinearAlgebra, SparseArrays
using SplitApplyCombine: group, sortkeys!
using Dictionaries: dictionary, Dictionary
import FillArrays: Zeros
import OrderedCollections: OrderedDict
using TestItems
using TermInterface


export FockNumber, JordanWignerOrdering, hc, focknumbers
export FockHilbertSpace, SymmetricFockHilbertSpace, SimpleFockHilbertSpace, hilbert_space
export parityoperator, numberoperator, fermions, majoranas, matrix_representation

export partial_trace, fermionic_kron, tensor_product, embedding, extension
export @fermions, @majoranas
export FermionConservation, NoSymmetry, ParityConservation, IndexConservation
export project_on_parity, project_on_parities

## Symbolics extension
function fermion_to_majorana end
function majorana_to_fermion end


abstract type AbstractHilbertSpace end
abstract type AbstractFockHilbertSpace <: AbstractHilbertSpace end
struct HC end
Base.:+(m, ::HC) = (m + m')
Base.:-(m, ::HC) = (m - m')
const hc = HC()

## Files
include("fock.jl")
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
        extension(c1[1], H1 => H2)
        embedding(c1[1], H1 => H)
        @fermions f
        FermionicHilbertSpaces.eval_in_basis((f[1] * f[2]' + 1 + f[1])^2 * 2.0 + hc, c1)
        matrix_representation((f[1] * f[2]' + 1 + f[1])^2 * 2.0, H1)
        @majoranas γ
        (γ[1] * γ[2] + 1.0 + γ[1])^2
    end
end

end
