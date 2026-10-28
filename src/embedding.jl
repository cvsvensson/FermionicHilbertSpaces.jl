function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isnothing(complement) && return change_basis(m, Hsub, H; kwargs...)
    return generalized_kron((m, I), (Hsub, complement), H; kwargs...)
end

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).

Unless `skipmissing=true`, throws an `ArgumentError` if a nonzero element of `m` combined with
a state of `complement` gives a state that is not in `H`, as happens when embedding into a
space with a constraint, such as a fixed particle number.
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
        skipmissing || _check_embed_support(H, Hsub, complement, state_mapper(H, (Hsub, complement)))
        _adjoint_superop(partial_trace_map(H, Hsub; complement, alg, skipmissing=true, kwargs...))
    end
    EmbedMap(Hsub, H, complement, kwargs, map)
end

function _check_embed_support(H, Hsub, complement, mapper)
    for fbar in basisstates(complement), f in basisstates(Hsub), fullf in first(combine_states((f, fbar), mapper))
        iszero(state_index(fullf, H)) && throw(ArgumentError("The state `$fullf` is not in the full Hilbert space. Pass `skipmissing=true` to drop such states."))
    end
    return nothing
end

function (op::EmbedMap)(m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    _devectorize_pt_output(op.map * _vectorize_pt_density_matrix(m), dim(op.H))
end
function (op::EmbedMap)(m::SparseMatrixCSC)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    _embed_sparse_matrix(op.map, m, dim(op.H))
end

# map * vec(m) as a sparse D × D matrix, avoiding the dense length-D^2 vector that map * m[:] would allocate
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
Unless `skipmissing=true`, throws an `ArgumentError` if a state of `Hsub` combined with a state
of the complement is not in `H`.
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