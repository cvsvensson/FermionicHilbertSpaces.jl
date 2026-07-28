function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    if isnothing(complement)
        return m
    end
    return generalized_kron((m, I), (Hsub, complement), H; kwargs...)
end

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces. 
"""
embed(m, Hs::PairWithHilbertSpaces; kwargs...) = embed(m, first(Hs), last(Hs); kwargs...)

"""
    embed(Hsub => H; kwargs...)

Create a callable embedding operation from `Hsub` into `H`.

The operation can be applied directly as `op(m)` and converted to a sparse matrix map with `sparse(op)`.
"""
struct EmbedMap{HS,H,C,K,M}
    Hsub::HS
    H::H
    complement::C
    kwargs::K
    map::M
end

function EmbedMap(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    map = partial_trace_map(H, Hsub; complement, kwargs...)'
    EmbedMap(Hsub, H, complement, kwargs, map)
end

function (op::EmbedMap)(m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match the size of `Hsub`"))
    reshape(op.map * vec(m), dim(op.H), dim(op.H))
end

function (op::EmbedMap)(out::AbstractMatrix, m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match the size of `Hsub`"))
    size(out) == (dim(op.H), dim(op.H)) || throw(DimensionMismatch("The output matrix must have size ($(dim(op.H)), $(dim(op.H))), got $(size(out))"))
    mul!(vec(out), op.map, vec(m))
    out
end

function (op::EmbedMap)(v::AbstractVector)
    length(v) == dim(op.Hsub)^2 || throw(DimensionMismatch("The input vector must have length $(dim(op.Hsub)^2), got $(length(v))"))
    op.map * v
end

function (op::EmbedMap)(out::AbstractVector, v::AbstractVector)
    length(v) == dim(op.Hsub)^2 || throw(DimensionMismatch("The input vector must have length $(dim(op.Hsub)^2), got $(length(v))"))
    length(out) == dim(op.H)^2 || throw(DimensionMismatch("The output vector must have length $(dim(op.H)^2), got $(length(out))"))
    mul!(out, op.map, v)
    out
end

function (op::EmbedMap)(m::UniformScaling)
    mfull = m.λ * I(dim(op.Hsub))
    reshape(op.map * vec(mfull), dim(op.H), dim(op.H))
end
SparseArrays.sparse(op::EmbedMap) = op.map
Base.adjoint(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, op.map')
Base.transpose(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, op.map')
Base.adjoint(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, op.map')
Base.transpose(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, op.map')

embed(Hs::PairWithHilbertSpaces; kwargs...) = EmbedMap(first(Hs), last(Hs); kwargs...)

@testitem "Partial trace, embed" begin
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsub = hilbert_space(f, [2, 4])
    Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
    pt = partial_trace(H => Hsub)
    emb = embed(Hsub => H)
    @test pt'.map == emb.map

    msub = rand(dim(Hsub), dim(Hsub))
    mfull = emb(msub)
    @test pt(mfull) ≈ msub * dim(Hcomp)
end