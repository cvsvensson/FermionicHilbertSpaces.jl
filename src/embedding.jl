function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isnothing(complement) && return change_basis(m, Hsub, H; kwargs...)
    return generalized_kron((m, I), (Hsub, complement), H; kwargs...)
end

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).
"""
embed(m, Hs::PairWithHilbertSpaces; kwargs...) = embed(m, first(Hs), last(Hs); kwargs...)

struct EmbedMap{HS,H,C,K,M}
    Hsub::HS
    H::H
    complement::C
    kwargs::K
    map::M
end

function EmbedMap(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), skipmissing=false, kwargs...)
    map = if isnothing(complement)
        basis_transformation_map(Hsub, H; skipmissing, kwargs...)
    else
        _adjoint_superop(partial_trace_map(H, Hsub; complement, alg, skipmissing=skipmissing, kwargs...))
    end
    EmbedMap(Hsub, H, complement, kwargs, map)
end

function (op::EmbedMap)(m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    _devectorize_pt_output(op.map * _vectorize_pt_density_matrix(m), dim(op.H))
end
function (op::EmbedMap)(m::SparseMatrixCSC)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    _embed_sparse_matrix(op.map, m, dim(op.H))
end

_embed_sparse_matrix(map, m::SparseMatrixCSC, D) = _devectorize_pt_output(map * m[:], D)
# map * vec(m) for a sparse map of size D^2 × n^2 and a sparse n × n matrix m. Multiplying with the
# sparse vector m[:] would allocate a dense vector of length D^2, so the output entries are
# collected and assembled into a D × D matrix instead.
function _embed_sparse_matrix(map::SparseMatrixCSC, m::SparseMatrixCSC, D)
    n = size(m, 1)
    rows = rowvals(map)
    vals = nonzeros(map)
    T = promote_type(eltype(map), eltype(m))
    J1s = Int[]
    J2s = Int[]
    Vs = T[]
    for (k1, k2, x) in zip(findnz(m)...)
        for p in nzrange(map, k1 + (k2 - 1) * n)
            J2, J1 = divrem(rows[p] - 1, D) .+ 1
            push!(J1s, J1)
            push!(J2s, J2)
            push!(Vs, vals[p] * x)
        end
    end
    return sparse(J1s, J2s, Vs, D, D)
end

function (op::EmbedMap)(out::AbstractMatrix, m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    size(out) == (dim(op.H), dim(op.H)) || throw(DimensionMismatch("The output matrix must have size ($(dim(op.H)), $(dim(op.H))), got $(size(out))"))
    mul!(vec(out), op.map, _vectorize_pt_density_matrix(m))
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
    _devectorize_pt_output(op.map * _vectorize_pt_density_matrix(mfull), dim(op.H))
end
SparseArrays.sparse(op::EmbedMap) = op.map
Base.adjoint(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, _adjoint_superop(op.map))
Base.transpose(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, _adjoint_superop(op.map))
Base.adjoint(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, _adjoint_superop(op.map))
Base.transpose(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, _adjoint_superop(op.map))

"""
    embed(Hsub => H; kwargs...)

Create a callable embedding operation from `Hsub` into `H`.

The operation can be applied directly as `op(m)` and converted to a sparse matrix map with `sparse(op)`.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).
"""
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

@testitem "Partial trace of a UniformScaling" begin
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:3)
    Hsub = hilbert_space(f, [2])
    @test partial_trace(2I, H, Hsub) ≈ 2 * partial_trace(I(dim(H)), H, Hsub)
end