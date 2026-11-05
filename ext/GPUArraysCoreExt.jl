module GPUArraysCoreExt

using FermionicHilbertSpaces: FermionicHilbertSpaces, NoScalarIndexingAlg, FactorizedPartialTransposeAlg, FactorizedKronAlg,
    partial_trace, embed, change_basis, partial_transpose, generalized_kron, tensor_product
using GPUArraysCore: AnyGPUArray

FermionicHilbertSpaces.default_alg(::typeof(partial_trace), ::AnyGPUArray, args...) = NoScalarIndexingAlg()
FermionicHilbertSpaces.default_alg(::Union{typeof(embed), typeof(reshape), typeof(change_basis)}, ::AnyGPUArray) = NoScalarIndexingAlg()
FermionicHilbertSpaces.default_alg(::typeof(partial_transpose), ::AnyGPUArray) = FactorizedPartialTransposeAlg()
FermionicHilbertSpaces.default_alg(::Union{typeof(generalized_kron), typeof(tensor_product)}, ::AnyGPUArray) = FactorizedKronAlg()

end
