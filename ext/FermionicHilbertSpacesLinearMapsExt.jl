module FermionicHilbertSpacesLinearMapsExt

using FermionicHilbertSpaces
using FermionicHilbertSpaces: LazyOperator, LazyRepr

using LinearMaps
using LinearAlgebra
import LinearAlgebra: kron


struct LazyOperatorMap{L,T} <: LinearMaps.LinearMap{T}
    op::L
end
Base.size(L::LazyOperatorMap) = size(L.op)
Base.size(L::LazyOperatorMap, i::Int) = size(L.op, i)
Base.eltype(::LazyOperatorMap{L,T}) where {L,T} = T
Base.adjoint(L::LazyOperatorMap) = LinearMap(adjoint(L.op))
Base.transpose(L::LazyOperatorMap) = LinearMap(transpose(L.op))
LinearMaps.LinearMap(op::L) where {L<:LazyOperator} = LazyOperatorMap{L,eltype(op)}(op)
LinearMaps.MulStyle(::LazyOperatorMap) = LinearMaps.FiveArg()
function LinearMaps._unsafe_mul!(y, A::LazyOperatorMap, x::AbstractVector, α=true, β=false)
    return LinearAlgebra.mul!(y, A.op, x, α, β)
end

function FermionicHilbertSpaces._factorized_term_matrix_representation(ops::Vector, H, ::LazyRepr; kwargs...)
    LinearMap(LazyOperator(ops, H; kwargs...))
end

LinearAlgebra.kron(L::LazyOperator) = LinearMap(L)
LinearAlgebra.kron(L1::LazyOperator, L2::LazyOperator) = kron(LinearMap(L1), LinearMap(L2))
LinearAlgebra.kron(L1::LazyOperator, L2) = kron(LinearMap(L1), L2)
LinearAlgebra.kron(L1, L2::LazyOperator) = kron(L1, LinearMap(L2))

end
