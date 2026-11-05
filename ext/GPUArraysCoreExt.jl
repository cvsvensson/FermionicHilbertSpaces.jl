module GPUArraysCoreExt

using FermionicHilbertSpaces: FermionicHilbertSpaces, GatherAlg, FactorizedPartialTransposeAlg, FactorizedKronAlg,
    partial_trace, embed, partial_transpose, generalized_kron, tensor_product
using GPUArraysCore: AnyGPUArray

FermionicHilbertSpaces.default_alg(::typeof(partial_trace), ::AnyGPUArray, args...) = GatherAlg()
FermionicHilbertSpaces.default_alg(::Union{typeof(embed), typeof(reshape)}, ::AnyGPUArray) = GatherAlg()
FermionicHilbertSpaces.default_alg(::typeof(partial_transpose), ::AnyGPUArray) = FactorizedPartialTransposeAlg()
FermionicHilbertSpaces.default_alg(::Union{typeof(generalized_kron), typeof(tensor_product)}, ::AnyGPUArray) = FactorizedKronAlg()

end
