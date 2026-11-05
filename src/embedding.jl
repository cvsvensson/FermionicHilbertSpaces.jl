function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement = Base.inferencebarrier(complementary_subsystem(H, Hsub)), projection = false, alg = default_alg(embed, m), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    size_compatible(m, Hsub) || throw(ArgumentError("The size of `m` must match `dim(Hsub)` = $(dim(Hsub)), got $(size(m))."))
    !isnothing(complement) && m isa AbstractVector && throw(ArgumentError("A state vector can only be embedded into a space with the same modes, such as from a sector into its parent space. A state on a subsystem does not fix the state of the rest; use `tensor_product` of states or `embed` a density matrix instead."))
    alg isa GatherAlg && return _embed_by_gather(m, Hsub, H, complement; projection, kwargs...)
    alg isa GenericAlg || throw(ArgumentError("`embed` supports `alg = GenericAlg()` or `GatherAlg()`, got $alg."))
    if isnothing(complement)
        m isa AbstractVector && return change_basis(m, Hsub, H; projection, kwargs...)
        if !(projection || m isa UniformScaling)
            dropped = last(basis_transformation(Hsub, H))
            iszero(m[dropped, setdiff(axes(m, 2), dropped)]) ||
                throw(ArgumentError("`m` maps a state of `H` outside `H` with nonzero amplitude. Pass `projection=true` to drop such terms."))
        end
        return change_basis(m, Hsub, H; projection = true, kwargs...)
    end
    return generalized_kron((m, I), (Hsub, complement), H; projection, kwargs...)
end

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces.
`Hsub => H` reads input => output; see [Pair arguments](@ref pair_convention).

The result is the compression `P (m ⊗ I) P` onto `H`, so that `tr(embed(A, Hsub => H) * ρ) ==
tr(A * partial_trace(ρ, H => Hsub))` for `ρ` on `H`. Unless `projection=true`, throws an `ArgumentError`
if `m ⊗ I` maps a state of `H` outside `H` with nonzero amplitude. Columns of `m` for states of
`Hsub` that no state of `H` contains are dropped, and states of `H` whose part on `Hsub` is not
in `Hsub` get zero.

With `alg = GatherAlg()`, `m` can be a GPU array (see [`GatherAlg`](@ref)).

If `m` is a vector, `Hsub` and `H` must have the same modes, as for a sector `Hsub` of `H`. The
result is `m` placed at the indices [`indices`](@ref)`(Hsub, H)` and zero elsewhere.

```jldoctest
julia> using FermionicHilbertSpaces

julia> @fermions f;

julia> H = hilbert_space(f, 1:2, NumberConservation());

julia> embed([1, 2], sector(1, H) => H)
4-element Vector{Int64}:
 0
 1
 2
 0
```
"""
embed(m, Hs::PairWithHilbertSpaces; kwargs...) = embed(m, first(Hs), last(Hs); kwargs...)

function _embed_by_gather(m::AbstractVecOrMat, Hsub, H, complement; projection = false, phase_factors = true)
    if isnothing(complement)
        src = first(basis_transformation(Hsub, H)) * (1:dim(Hsub))
        splits = [((K, 1, 1),) for K in src]
    else
        splits = _full_partial_trace_splits(H, Hsub, complement, _partial_trace_mapper(H, Hsub, complement), phase_factors)
    end
    P, B, S, present = _embed_gather_table(splits, dim(Hsub), isnothing(complement) ? 1 : dim(complement))
    projection || _check_embed_into(m, present)
    return _gather_embed(m, P, B, S)
end
_embed_by_gather(::UniformScaling, args...; kwargs...) = throw(ArgumentError("`UniformScaling` is not supported with `alg = GatherAlg()`."))

# `P[J, r]`, `B[J, r]` and `S[J, r]` are the indices in `Hsub` and the complement, and the weight,
# of the r-th split of the J-th state of `H`, padded with index 1 and weight 0.
# `present[K, b]` is whether some state of `H` splits into (K, b).
function _embed_gather_table(splits, dsub, nb)
    R = maximum(ss -> count(!iszero ∘ first, ss), splits; init = 0)
    P = ones(Int, length(splits), R)
    B = ones(Int, size(P))
    S = zeros(fieldtype(eltype(eltype(splits)), 3), size(P))
    present = falses(dsub, nb)
    for (J, ss) in pairs(splits)
        r = 0
        for (K, b, s) in ss
            iszero(K) && continue
            r += 1
            P[J, r], B[J, r], S[J, r] = K, b, s
            present[K, b] = true
        end
    end
    return P, B, S, present
end

# m ⊗ I maps (K2, b) in `H` to (K1, b) outside `H` exactly when present[K2, b] and !present[K1, b]
_embed_outside(::AbstractMatrix, present) = (.!present) * transpose(present) .> 0
_embed_outside(::AbstractVector, present) = vec(.!present)
function _check_embed_into(m, present)
    all(present) && return nothing
    any(_adapt_like(m, Array(_embed_outside(m, present))) .& .!iszero.(m)) &&
        throw(ArgumentError("`m` maps a state of `H` outside `H` with nonzero amplitude. Pass `projection=true` to drop such terms."))
    return nothing
end

function _gather_embed(m::AbstractMatrix, P, B, S)
    D, R = size(P)
    P, B, S = _adapt_like(m, P), _adapt_like(m, B), _adapt_like(m, S)
    row(A) = reshape(A, D, R, 1, 1)
    col(A) = reshape(A, 1, 1, D, R)
    LL = row(P) .+ size(m, 1) .* (col(P) .- 1)
    WW = conj.(row(S)) .* col(S) .* (row(B) .== col(B))
    return reshape(_gather_reduce(WW, (2, 4), m, LL), D, D)
end
_gather_embed(v::AbstractVector, P, B, S) = vec(_gather_reduce(_adapt_like(v, S), 2, v, _adapt_like(v, P)))

struct EmbedMap{HS,H,C,K,M}
    Hsub::HS
    H::H
    complement::C
    kwargs::K
    map::M
end

# The map is the compression onto `H` and checks nothing, so `projection` has no effect.
function EmbedMap(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement = Base.inferencebarrier(complementary_subsystem(H, Hsub)), alg = default_partial_trace_alg(Hsub, H, complement), projection = false, kwargs...)
    map = if isnothing(complement)
        basis_transformation_map(Hsub, H; projection = true, kwargs...)
    else
        _adjoint_superop(partial_trace_map(H, Hsub; complement, alg, kwargs...))
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
    size_compatible(out, op.H) || throw(DimensionMismatch("The output matrix must have size ($(dim(op.H)), $(dim(op.H))), got $(size(out))"))
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
It is the compression onto `H` and, unlike [`embed`](@ref), does not check its input.
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
    m = representation(hopping, Hsub)
    @test embed(m, Hsub => H) ≈ representation(hopping, H)
    @test emb(m) ≈ representation(hopping, H)
    # this maps states of H outside H
    m = representation(f[1]', Hsub; projection=true)
    @test_throws ArgumentError embed(m, Hsub => H)
    compression = representation(f[1]', H; projection=true)
    @test embed(m, Hsub => H; projection = true) ≈ compression
    @test emb(m) ≈ compression # the map is the compression and does not check
    # with the same atoms: |00⟩⟨11| maps no state of the space with at most one particle outside it, |11⟩⟨00| does
    Hf, Hc = hilbert_space(f, 1:2), hilbert_space(f, 1:2, NumberConservation(0:1))
    m = zeros(4, 4)
    m[1, 4] = 1
    @test iszero(embed(m, Hf => Hc)) && iszero(embed(Hf => Hc)(m))
    @test_throws ArgumentError embed(permutedims(m), Hf => Hc)
    # embed is the adjoint of partial_trace
    A = rand(ComplexF64, dim(Hsub), dim(Hsub))
    ρ = rand(ComplexF64, dim(H), dim(H))
    @test tr(embed(A, Hsub => H; projection = true) * ρ) ≈ tr(A * partial_trace(ρ, H => Hsub))
end

@testitem "Embed by gather" begin
    import FermionicHilbertSpaces: GatherAlg
    alg = GatherAlg()
    @fermions f
    @spins s
    @bosons b
    Hcons = hilbert_space(f, 1:6, NumberConservation(3))
    Hprod = tensor_product(hilbert_space(f, 1:3), hilbert_space(s[1], 1 // 1), hilbert_space(b[1], 3))
    Hpar = hilbert_space(f, 1:4, ParityConservation())
    for (Hsub, H) in (
            (hilbert_space(f, [2, 4]), hilbert_space(f, 1:5)),
            (subregion(hilbert_space(f, [1, 4]), Hcons), Hcons),
            (tensor_product(hilbert_space(f, 2:2), hilbert_space(b[1], 3)), Hprod),
            (sector(-1, Hpar), Hpar),
        )
        m = rand(ComplexF64, dim(Hsub), dim(Hsub))
        @test embed(m, Hsub => H; alg, projection = true) ≈ embed(m, Hsub => H; projection = true)
    end
    v = rand(ComplexF64, dim(sector(-1, Hpar)))
    @test embed(v, sector(-1, Hpar) => Hpar; alg) ≈ embed(v, sector(-1, Hpar) => Hpar)

    # f[1]' maps states of the number sector outside it
    H = hilbert_space(f, 1:6, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, 1:3), H)
    @test_throws ArgumentError embed(representation(f[1]', Hsub, Matrix; projection = true), Hsub => H; alg)
    m = representation(f[2]' * f[1] + hc, Hsub, Matrix)
    @test embed(m, Hsub => H; alg) ≈ embed(m, Hsub => H)
    Hf, Hc = hilbert_space(f, 1:2), hilbert_space(f, 1:2, NumberConservation(0:1))
    @test_throws ArgumentError embed(rand(4, 4), Hf => Hc; alg)
    @test_throws ArgumentError embed(rand(4), Hf => Hc; alg)
    @test_throws ArgumentError embed(rand(4, 4), Hf => hilbert_space(f, 1:3); alg = :other)
end

@testitem "Embedding a sector vector" begin
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:4, ParityConservation())
    Hsec = sector(-1, H)
    v = rand(ComplexF64, dim(Hsec))
    w = embed(v, Hsec => H)
    @test w[indices(Hsec, H)] == v && norm(w) ≈ norm(v)
    @test_throws ArgumentError embed(rand(4), hilbert_space(f, 1:2) => H)
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