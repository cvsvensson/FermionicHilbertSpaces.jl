# Based on https://github.com/JuliaArrays/ReadOnlyArrays.jl
using Base: @propagate_inbounds

struct BasisArray{T,N,A,B} <: AbstractArray{T,N}
    parent::A
    basis::B
    function BasisArray(parent::AbstractArray{T,N}, basis::B) where {T,N,B}
        new{T,N,typeof(parent),B}(parent, basis)
    end
end

BasisArray{T}(parent::AbstractArray{T,N}, basis) where {T,N} = BasisArray(parent, basis)

BasisArray{T,N}(parent::AbstractArray{T,N}, basis) where {T,N} = BasisArray(parent, basis)

BasisArray{T,N,P}(parent::P, basis) where {T,N,P<:AbstractArray{T,N}} = BasisArray(parent, basis)

#--------------------------------------
# aliases

const BVector{T,P} = BasisArray{T,1,P}

BVector(parent::AbstractVector, basis) = BasisArray(parent, basis)

const BMatrix{T,P} = BasisArray{T,2,P}

BMatrix(parent::AbstractMatrix, basis) = BasisArray(parent, basis)

#--------------------------------------

Base.size(x::BasisArray, args...) = size(x.parent, args...)

@propagate_inbounds function Base.getindex(x::BasisArray, args...)
    getindex(x.parent, args...)
end
Base.setindex!(x::BasisArray, args...) = setindex!(x.parent, args...)

Base.IndexStyle(::Type{<:BasisArray{T,N,P}}) where {T,N,P} = IndexStyle(P)

Base.iterate(x::BasisArray, args...) = iterate(x.parent, args...)

Base.length(x::BasisArray) = length(x.parent)

Base.similar(x::BasisArray) = BasisArray(similar(x.parent), x.basis)
Base.similar(x::BasisArray, dims::Union{Integer,AbstractUnitRange}...) = BasisArray(similar(x.parent, dims...), x.basis)
Base.similar(x::BasisArray, ::Type{T}, dims::Union{Integer,AbstractUnitRange}...) where {T} = BasisArray(similar(x.parent, T, dims...), x.basis)

Base.axes(x::BasisArray) = axes(x.parent)

function Base.IteratorSize(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    Base.IteratorSize(P)
end

function Base.IteratorEltype(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    Base.IteratorEltype(P)
end

function Base.eltype(::Type{<:BasisArray{T,N,P}}) where {T,N,P}
    eltype(P)
end

Base.firstindex(x::BasisArray) = firstindex(x.parent)

Base.lastindex(x::BasisArray) = lastindex(x.parent)

Base.strides(x::BasisArray) = strides(x.parent)

function Base.unsafe_convert(p::Type{Ptr{T}}, x::BasisArray) where {T}
    Base.unsafe_convert(p, x.parent)
end

Base.stride(x::BasisArray, i::Int) = stride(x.parent, i)

Base.parent(x::BasisArray) = x.parent

function Base.:*(x::BasisArray, y::BasisArray)
    x.basis == y.basis && return BasisArray(x.parent * y.parent, x.basis)
    H = combine_spaces(x.basis, y.basis)
    embed(x, H) * embed(y, H)
end
Base.:*(x::BasisArray, y::Number) = BasisArray(x.parent .* y, x.basis)
Base.:*(x::Number, y::BasisArray) = BasisArray(x .* y.parent, y.basis)

function Base.:+(x::BasisArray, y::BasisArray)
    x.basis == y.basis && return BasisArray(x.parent + y.parent, x.basis)
    H = combine_spaces(x.basis, y.basis)
    embed(x, H) + embed(y, H)
end
Base.:+(x::BasisArray, y::AbstractArray) = BasisArray(x.parent + y, x.basis)
Base.:+(x::AbstractArray, y::BasisArray) = BasisArray(x + y.parent, y.basis)

Base.:-(x::BasisArray, y::AbstractArray) = x + (-y)
Base.:-(x::AbstractArray, y::BasisArray) = x + (-y)
Base.:-(x::BasisArray, y::BasisArray) = (-x) + (-y)
Base.:-(x::BasisArray) = BasisArray(-x.parent, x.basis)

Base.adjoint(b::BasisArray) = BasisArray(adjoint(b.parent), b.basis)
Base.transpose(b::BasisArray) = BasisArray(transpose(b.parent), b.basis)
LinearAlgebra.Hermitian(b::BasisArray) = BasisArray(Hermitian(b.parent), b.basis)
Base.zero(x::BasisArray) = BasisArray(zero(x.parent), x.basis)
Base.one(x::BasisArray) = BasisArray(one(x.parent), x.basis)

partial_trace(m::Union{BMatrix,BVector}, Hsub::AbstractHilbertSpace) = BasisArray(partial_trace(m.parent, m.basis => Hsub), Hsub)
embed(m::BMatrix, H::AbstractHilbertSpace; kwargs...) = BasisArray(embed(m.parent, m.basis => H; kwargs...), H)
function extend(m::BMatrix, Hbar::AbstractHilbertSpace; kwargs...)
    Hout = combine_spaces(m.basis, Hbar)
    BasisArray(extend(m.parent, m.basis => Hbar, Hout), Hout)
end
tensor_product(ms::Union{NTuple{N,<:BMatrix},Vector{<:BMatrix}}, H::AbstractHilbertSpace) where N = BasisArray(tensor_product(ms, map(m -> m.basis, ms) => H))

function combine_spaces(H1::AbstractFockHilbertSpace, H2::AbstractFockHilbertSpace)
    # Find the intersection of the spaces
    intersection = intersect(keys(H1.jw), keys(H2.jw))
    if length(intersection) == 0
        return tensor_product(H1, H2)
    end
    H1complabels = setdiff(keys(H1.jw), keys(H2.jw),)
    H2complabels = setdiff(keys(H2.jw), keys(H1.jw))
    H1comp = subregion(H1complabels, H1)
    H2comp = subregion(H2complabels, H2)
    H1intersect = subregion(intersection, H1)
    H2intersect = subregion(intersection, H2)
    #combine basis states of the intersecting region
    states = union(basisstates(H1intersect), basisstates(H2intersect))
    Hintersect = hilbert_space(intersection, states)
    return tensor_product(H1comp, Hintersect, H2comp)
end
combine_spaces(H1::AbstractFockHilbertSpace, H2::AbstractFockHilbertSpace, Hs...) = combine_spaces(H1, combine_spaces(H2, Hs...))

@testitem "BasisArray" begin
    import FermionicHilbertSpaces: BasisArray
    using LinearAlgebra
    H1 = hilbert_space(1:2)
    H2 = hilbert_space(2:3)
    H3 = hilbert_space(3:4)
    m1 = BasisArray(rand(dim(H1), dim(H1)), H1)
    m2 = BasisArray(rand(dim(H2), dim(H2)), H2)
    m3 = BasisArray(rand(dim(H3), dim(H3)), H3)

    @test m1 * m1 isa BasisArray
    @test m1 - 2m1 isa BasisArray
    @test m1 - I isa BasisArray
    @test m1 + I isa BasisArray
    @test m1 + rand(dim(H1), dim(H1)) isa BasisArray
    @test m1 + 2 * m1 == 3 * m1

    @test extend(m1, H3) isa BasisArray
    @test partial_trace(extend(m1, H3), H1) == dim(H3) * m1
    @test m1 * m2 * m3 isa BasisArray
    @test m1 + m2 + m3 isa BasisArray
end