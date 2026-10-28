function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isnothing(complement) && return change_basis(m, Hsub, H; kwargs...)
    return generalized_kron((m, I), (Hsub, complement), H; kwargs...)
end

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).

The result is the compression `P (m ⊗ I) P` onto `H`, so that `tr(embed(A) * ρ) ==
tr(A * partial_trace(ρ))` for `ρ` on `H`. Unless `skipmissing=true`, throws an `ArgumentError`
if `m ⊗ I` maps a state of `H` outside `H` with nonzero amplitude. Elements of `m` on states of
`Hsub` that no state of `H` contains are dropped, and states of `H` whose part on `Hsub` is not
in `Hsub` get zero.
"""
embed(m, Hs::PairWithHilbertSpaces; kwargs...) = embed(m, first(Hs), last(Hs); kwargs...)

struct EmbedMap{HS,H,C,K,M,S}
    Hsub::HS
    H::H
    complement::C
    kwargs::K
    map::M
    support::S # `(; trace, embed)` supports that inputs are checked against, or `nothing`
end

function EmbedMap(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), skipmissing=false, kwargs...)
    map = if isnothing(complement)
        basis_transformation_map(Hsub, H; skipmissing=true, kwargs...)
    else
        _adjoint_superop(partial_trace_map(H, Hsub; complement, alg, kwargs...))
    end
    EmbedMap(Hsub, H, complement, kwargs, map, _map_support(H, Hsub, complement, skipmissing))
end
function (op::EmbedMap)(args...)
    in, n = last(args), dim(op.Hsub)
    m = in isa AbstractVector && length(in) == n^2 ? reshape(in, n, n) : in
    m isa AbstractVector || _check_support(m, isnothing(op.support) ? nothing : op.support.embed)
    _apply(op, args...)
end

# For substates `i` and `j` of `Hsub` with nonzero `m[i, j]`, `m ⊗ I` keeps the states of `H` in `H`
# when `allowed[cls[j], cls[i]]`: when every complement part that `j` has in `H`, `i` has too.
struct EmbedSupport
    cls::Vector{Int}
    allowed::BitMatrix
end
function _embed_support(splits, complement, nsub)
    reach = [Int[] for _ in 1:nsub]
    for (_, s) in splits, (K, _, fbar, _) in s
        iszero(K) || push!(reach[K], state_index(fbar, complement))
    end
    foreach(sort!, reach)
    allequal(reach) && return nothing
    classes = unique(reach)
    return EmbedSupport(indexin(reach, classes), BitMatrix([issubset(a, b) for a in classes, b in classes]))
end
# Call `f(i, j, m[i, j])` for the stored elements of `m`
function _foreach_stored(f, m::SparseMatrixCSC)
    rows, vals = rowvals(m), nonzeros(m)
    for j in axes(m, 2), p in nzrange(m, j)
        f(rows[p], j, vals[p])
    end
end
_foreach_stored(f, m) = foreach(I -> f(Tuple(I)..., m[I]), CartesianIndices(m))
function _check_support(m::AbstractMatrix, s::EmbedSupport)
    size(m) == (length(s.cls), length(s.cls)) || return nothing
    _foreach_stored(m) do i, j, x
        s.allowed[s.cls[j], s.cls[i]] || iszero(x) ||
            throw(ArgumentError("The element ($i, $j) of `m` maps a state of `H` outside `H`. Pass `skipmissing=true` to drop such terms."))
    end
end
_check_support(::UniformScaling, ::EmbedSupport) = nothing

function _apply(op::EmbedMap, m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    _devectorize_pt_output(op.map * _vectorize_pt_density_matrix(m), dim(op.H))
end
function _apply(op::EmbedMap, m::SparseMatrixCSC)
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

function _apply(op::EmbedMap, out::AbstractMatrix, m::AbstractMatrix)
    size_compatible(m, op.Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(op.Hsub)), got $(size(m))."))
    size(out) == (dim(op.H), dim(op.H)) || throw(DimensionMismatch("The output matrix must have size ($(dim(op.H)), $(dim(op.H))), got $(size(out))"))
    mul!(vec(out), op.map, _vectorize_pt_density_matrix(m))
    out
end

function _apply(op::EmbedMap, v::AbstractVector)
    length(v) == dim(op.Hsub)^2 || throw(DimensionMismatch("The input vector must have length $(dim(op.Hsub)^2), got $(length(v))"))
    op.map * v
end

function _apply(op::EmbedMap, out::AbstractVector, v::AbstractVector)
    length(v) == dim(op.Hsub)^2 || throw(DimensionMismatch("The input vector must have length $(dim(op.Hsub)^2), got $(length(v))"))
    length(out) == dim(op.H)^2 || throw(DimensionMismatch("The output vector must have length $(dim(op.H)^2), got $(length(out))"))
    mul!(out, op.map, v)
    out
end

function _apply(op::EmbedMap, m::UniformScaling)
    mfull = m.λ * I(dim(op.Hsub))
    _devectorize_pt_output(op.map * _vectorize_pt_density_matrix(mfull), dim(op.H))
end
SparseArrays.sparse(op::EmbedMap) = op.map
Base.adjoint(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, _adjoint_superop(op.map), op.support)
Base.transpose(op::PartialTraceMap) = EmbedMap(op.Hsub, op.H, op.complement, op.kwargs, _adjoint_superop(op.map), op.support)
Base.adjoint(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, _adjoint_superop(op.map), op.support)
Base.transpose(op::EmbedMap) = PartialTraceMap(op.H, op.Hsub, op.complement, op.kwargs, _adjoint_superop(op.map), op.support)

"""
    embed(Hsub => H; kwargs...)

Create a callable embedding operation from `Hsub` into `H`.

The operation can be applied directly as `op(m)` and converted to a sparse matrix map with `sparse(op)`.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).
Applying it throws as [`embed`](@ref) does, unless `skipmissing=true`; `sparse(op)` is not checked.
"""
embed(Hs::PairWithHilbertSpaces; kwargs...) = EmbedMap(first(Hs), last(Hs); kwargs...)

@testitem "Embedding into a number sector" begin
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:12, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, 1:4), H)
    emb = embed(Hsub => H)
    @test embed(Matrix(I, dim(Hsub), dim(Hsub)), Hsub => H) == I
    hopping = sum(f[n]' * f[n+1] + hc for n in 1:3)
    m = matrix_representation(hopping, Hsub)
    @test embed(m, Hsub => H) ≈ matrix_representation(hopping, H)
    @test emb(m) ≈ matrix_representation(hopping, H)
    # these map states of H outside H
    for op in (f[1]', f[1]' * f[1] + f[1]')
        m = matrix_representation(op, Hsub; projection=true)
        @test_throws ArgumentError embed(m, Hsub => H)
        @test_throws ArgumentError emb(m)
        compression = matrix_representation(op, H; projection=true)
        @test embed(m, Hsub => H; skipmissing=true) ≈ compression
        @test embed(Hsub => H; skipmissing=true)(m) ≈ compression
    end
    # embed is the adjoint of partial_trace
    A = rand(ComplexF64, dim(Hsub), dim(Hsub))
    ρ = rand(ComplexF64, dim(H), dim(H))
    @test tr(embed(A, Hsub => H; skipmissing=true) * ρ) ≈ tr(A * partial_trace(ρ, H => Hsub))
end

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