struct _QR end
struct _EIG end

function permute_state(state, mapper, perm::AbstractVector)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = substates[perm]
    newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
    return newstate
end
function permute_state(state, mapper, perm::NTuple)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = TupleTools.permute(substates, Tuple(perm))
    newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
    return newstate
end

"""
    permutation_operator(H, Hs, perm)

Gives the matrix representation on `H` of a permutation `perm` of the partition `Hs`.
"""
function permutation_operator(H::AbstractHilbertSpace, Hs, perm, ::Type{T}=Int) where T
    P = zeros(T, dim(H), dim(H))
    add_permutation_operator!(P, (basisstates(H), Base.Fix2(state_index, H)), state_mapper(H, Hs), perm, one(T))
end
function add_permutation_operator!(P, (states, state_index), mapper, perm, weight=1)
    _perm = if unique_split_state(first(states), mapper) isa Tuple
        Tuple(perm)
    else
        perm
    end
    return _add_permutation_operator!(P, (states, state_index), mapper, _perm, weight)
end

function _add_permutation_operator!(P, (states, state_index), mapper, perm, weight)
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        iszero(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
        push_inds_amps!(P, i, j, weight)
    end
    return P
end
"""
    permutation_projector(H, Hs, perms, weights=nothing, T = Float64; normalize=true)

Build a group-averaged projector/operator from permutation operators:
`P = sum(weights[g] * R_g for g)`.

`Hs` may be either a full partition of `H` (covering all atomic factors) or a proper
subsystem list (covering only a subset of atomic factors). In the subsystem case the
projector is first constructed on `Hsub = tensor_product(Hs...)` and then embedded
back into `H` via `embed(Psub, Hsub => H)`.
"""
function permutation_projector(H::AbstractHilbertSpace, Hs, perms, weights=nothing, ::Type{T}=_default_projector_eltype(weights); normalize=true, sparse=true) where T
    isempty(perms) && throw(ArgumentError("At least one permutation is required"))

    # Subsystem branch: Hs does not partition H, treat as a proper subsystem.
    if !ispartition(Hs, H)
        Hsub = subregion(tensor_product(Hs...), H)
        issubsystem(Hsub, H) || throw(ArgumentError("Hs is neither a partition nor a subsystem of H"))
        Psub = permutation_projector(Hsub, Hs, perms, weights, T; normalize)
        return embed(Psub, Hsub => H)
    end

    ws = if isnothing(weights)
        ones(T, length(perms))
    else
        length(weights) == length(perms) || throw(ArgumentError("weights must match perms length"))
        weights
    end

    d = dim(H)
    accum = if sparse
        if 4 * length(perms) > d
            zeros(T, d, d)
        else
            L = d * length(perms)
            map(v -> sizehint!(v, L), (Int[], Int[], T[]))
        end
    else
        zeros(T, d, d)
    end
    mapper = state_mapper(H, Hs)
    states = basisstates(H)
    si = Base.Fix2(state_index, H)
    for (perm, w) in zip(perms, ws)
        add_permutation_operator!(accum, (states, si), mapper, perm, w)
    end
    P = if sparse
        SparseArrays.sparse(finalize!(accum, H))
    else
        accum
    end
    
    if normalize
        tr2 = sum(abs2, P)
        tr1 = tr(P)
        iszero(tr2) || (P *= (tr1 / tr2))
    end
    return P
end

function _default_projector_eltype(weights)
    isnothing(weights) && return Int
    eltype(weights) <: Number && return eltype(weights)
    typeof(first(weights)) <: Number && return typeof(first(weights))
    return Float64
end

"""
    symmetric_sector(H, Hs, sector=:symmetric, T=Float64; normalize=true, cutoff=0.9, atol=1e-10)

Construct the projector onto symmetric sectors of `H` under permutations of the partition `Hs`. 

If `Combinatorics.jl` is loaded, `sector` can be `:symmetric` or `:antisymmetric` to automatically generate the full set of permutations and weights for the symmetric or antisymmetric representation. Otherwise, `sector` must be a tuple of `(perms, weights)` to specify the permutations and their corresponding weights directly.

`orth_method` selects what is returned. With `_QR()` (the default) or `_EIG()`, it is a
matrix with orthonormal columns spanning the sector, from column-pivoted QR
decompositions or from eigendecompositions of the diagonal blocks of the projector. With
`nothing`, it is the projector itself.
Both decompositions assume that the projector is proportional to an orthogonal
projector, which holds when `perms` form a group and `weights` are proportional to a
character of it or to its complex conjugate. Each diagonal block `B` contributes at most
`round(Int, real(tr(B))^2 / sum(abs2, B))` columns, which is its rank in that case.

`Hs` may be either a full partition of `H` or a proper subsystem list; in the latter
case the sector is constructed on `tensor_product(Hs...)` and embedded back into `H`.
"""
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector=:symmetric, ::Type{T}=Float64;
    orth_method=_QR(), kwargs...) where T
    isempty(Hs) && throw(ArgumentError("At least one factor space is required"))

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("Generated weights must match generated permutations"))
    P = permutation_projector(H, Hs, perms, weights, T; kwargs...)
    isnothing(orth_method) && return P
    _sector_basis(P, orth_method)
end
_resolve_sector_permutations_and_weights(Hs, (perms, weights), T) = (perms, weights) # for direct input of perms and weights

# Orthonormal basis for the range of `P`. `P` is block diagonal over the connected
# components of its sparsity pattern, so each block is decomposed separately.
function _sector_basis(P, method)
    F = float(eltype(P))
    # Let `P` be p times an orthogonal projector. Its columns have norm at most p, its
    # nonzero eigenvalues are p, and for an n×n block every pivot of the `_QR` method of
    # `_block_basis` inside the block's rank is at least p/sqrt(n), while
    # rounding errors are of order eps*p. sqrt(eps) times the largest column norm lies
    # between the two for any block with fewer than about 1/eps states.
    # Each block also keeps at most `_trace_rank` columns, which is its rank when the block
    # is proportional to a projector.
    tol = sqrt(eps(real(F))) * _max_column_norm(P)
    blocks = _connected_blocks(P)
    loc = zeros(Int, size(P, 1))
    bases = map(blocks) do idx
        B = _dense_block(P, idx, loc)
        _block_basis(B, method, tol, _trace_rank(B))
    end
    Q = zeros(F, size(P, 1), sum(B -> size(B, 2), bases; init=0))
    col = 0
    for (idx, B) in zip(blocks, bases)
        Q[idx, col+1:col+size(B, 2)] = B
        col += size(B, 2)
    end
    return Q
end

function _max_column_norm(P::SparseMatrixCSC)
    vals = nonzeros(P)
    z = zero(real(eltype(P)))
    column_norm(j) = sqrt(sum(k -> abs2(vals[k]), nzrange(P, j); init=z))
    return maximum(column_norm, axes(P, 2); init=z)
end
_max_column_norm(P::AbstractMatrix) = maximum(norm, eachcol(P); init=zero(real(eltype(P))))
function _trace_rank(B)
    s = sum(abs2, B)
    return iszero(s) ? 0 : round(Int, real(tr(B))^2 / s)
end

# Column-pivoted Gram-Schmidt QR of `B`, stopped after `r` columns or once every residual
# column norm (the next diagonal entry of R) is at most `tol`. Returns the orthonormal
# columns found. Stopping early keeps the cost proportional to the rank of `B`.
function _block_basis(B, ::_QR, tol, r)
    m, n = size(B)
    kmax = min(m, n, r)
    Q = zeros(eltype(B), m, kmax)
    norms = zeros(real(eltype(B)), n)
    w = zeros(eltype(B), n)
    k = 0
    while k < kmax
        for c in 1:n
            norms[c] = sum(abs2, view(B, :, c))
        end
        j = argmax(norms)
        sqrt(norms[j]) > tol || break
        v = B[:, j]
        Qk = view(Q, :, 1:k)
        v .-= Qk * (Qk' * v)  # reorthogonalize against the columns found so far
        v ./= norm(v)
        k += 1
        Q[:, k] = v
        mul!(w, B', v)
        mul!(B, v, w', -1, 1)  # B -= v * v' * B
    end
    return Q[:, 1:k]
end
# Eigenvectors of `Hermitian(B)` for the at most `r` eigenvalues of largest magnitude,
# among those larger than `tol` in magnitude.
function _block_basis(B, ::_EIG, tol, r)
    E = eigen!(Hermitian(B))
    order = sortperm(E.values; by=abs, rev=true)
    keep = filter(i -> abs(E.values[i]) > tol, first(order, r))
    return E.vectors[:, keep]
end

function _dense_block(P::SparseMatrixCSC, idx, loc)
    loc[idx] = eachindex(idx)
    B = zeros(float(eltype(P)), length(idx), length(idx))
    rows, vals = rowvals(P), nonzeros(P)
    for (jl, j) in enumerate(idx), k in nzrange(P, j)
        iszero(vals[k]) || (B[loc[rows[k]], jl] = vals[k])
    end
    return B
end
_dense_block(P::AbstractMatrix, idx, loc) = Matrix{float(eltype(P))}(P[idx, idx])

# Index sets of the connected components of the graph with an edge i-j for each
# nonzero P[i, j], each sorted, ordered by their smallest index.
function _connected_blocks(P::SparseMatrixCSC)
    parent = collect(1:size(P, 2))
    rows, vals = rowvals(P), nonzeros(P)
    for j in axes(P, 2), k in nzrange(P, j)
        iszero(vals[k]) || _union!(parent, rows[k], j)
    end
    return _components(parent)
end
function _connected_blocks(P::AbstractMatrix)
    parent = collect(1:size(P, 2))
    for j in axes(P, 2), i in axes(P, 1)
        iszero(P[i, j]) || _union!(parent, i, j)
    end
    return _components(parent)
end
function _find_root!(parent, i)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end
function _union!(parent, i, j)
    a, b = _find_root!(parent, i), _find_root!(parent, j)
    a != b && (parent[a] = b)
    return parent
end
function _components(parent)
    label = zeros(Int, length(parent))
    comps = Vector{Int}[]
    for i in eachindex(parent)
        r = _find_root!(parent, i)
        if iszero(label[r])
            push!(comps, Int[])
            label[r] = length(comps)
        end
        push!(comps[label[r]], i)
    end
    return comps
end



@testitem "Permutation symmetry" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, permutation_projector, symmetric_sector
    using Combinatorics: permutations

    permutation_sign(perm) = isodd(sum(perm[i] > perm[j] for i in 1:(length(perm)-1) for j in (i+1):length(perm))) ? -1.0 : 1.0

    @fermions f
    H1 = hilbert_space(f, 1:1)
    H2 = hilbert_space(f, 2:2)
    H = tensor_product(H1, H2)

    idperm = [1, 2]
    swapperm = [2, 1]

    Rswap = permutation_operator(H, [H1, H2], swapperm)
    @test Rswap * Rswap ≈ I

    perms = [idperm, swapperm]
    Ptriv = permutation_projector(H, [H1, H2], perms)
    Psign = permutation_projector(H, [H1, H2], perms, (1.0, -1.0))

    @test Ptriv * Ptriv ≈ Ptriv
    @test Psign * Psign ≈ Psign
    @test Ptriv * Psign ≈ 0I

    sympw = ([idperm, swapperm], [1.0, 1.0])
    antisymw = ([idperm, swapperm], [1.0, -1.0])
    Ptriv = symmetric_sector(H, [H1, H2], sympw)
    Psign = symmetric_sector(H, [H1, H2], antisymw)

    @test Ptriv == symmetric_sector(H, [H1, H2], :symmetric)
    @test Psign == symmetric_sector(H, [H1, H2], :antisymmetric)

    @test size(Psign, 2) == 1
    @test size(Ptriv, 2) == 3

    Ttriv = symmetric_sector(H, [H1, H2], sympw; orth_method=nothing)
    Tsign = symmetric_sector(H, [H1, H2], antisymw; orth_method=nothing)
    @test Ptriv' * Ptriv ≈ I
    @test Ptriv * Ptriv' ≈ Ttriv
    @test Psign' * Psign ≈ I
    @test Psign * Psign' ≈ Tsign

    @test Ttriv' * Ttriv ≈ Ttriv
    @test Tsign' * Tsign ≈ Tsign

    H3 = hilbert_space(f, 3:3)
    H123 = tensor_product(H1, H2, H3)
    Hs3 = [H1, H2, H3]
    perms3 = permutations(1:3)
    ws3 = [permutation_sign(p) for p in perms3]
    Pmanual3 = permutation_projector(H123, Hs3, perms3, ws3)
    Pauto3 = symmetric_sector(H123, Hs3, :antisymmetric, orth_method=nothing)
    @test Pmanual3 ≈ Pauto3

    @test_throws ArgumentError symmetric_sector(H, [H1, H2], :invalid)

    # Subsystem tests: Hs covers only a subset of H
    Hs_sub = [H1, H2]
    Hsub = tensor_product(H1, H2)

    # Subsystem result should equal manually embedded projector
    Psub_manual_sym = permutation_projector(Hsub, Hs_sub, [idperm, swapperm])
    Psub_embed_sym = embed(Psub_manual_sym, Hsub => H123)
    Psub_auto_sym = permutation_projector(H123, Hs_sub, [idperm, swapperm])
    @test Psub_auto_sym ≈ Psub_embed_sym

    Psub_manual_anti = permutation_projector(Hsub, Hs_sub, [idperm, swapperm], (1.0, -1.0))
    Psub_embed_anti = embed(Psub_manual_anti, Hsub => H123)
    Psub_auto_anti = permutation_projector(H123, Hs_sub, [idperm, swapperm], (1.0, -1.0))
    @test Psub_auto_anti ≈ Psub_embed_anti

    # symmetric_sector subsystem path
    Qsym = symmetric_sector(H123, Hs_sub, :symmetric)
    Qanti = symmetric_sector(H123, Hs_sub, :antisymmetric)

    # Columns should be orthonormal
    @test Qsym' * Qsym ≈ I
    @test Qanti' * Qanti ≈ I

    # Q*Q' should equal the full space projector (orth_method=nothing)
    Tsym_sub = symmetric_sector(H123, Hs_sub, :symmetric; orth_method=nothing)
    Tanti_sub = symmetric_sector(H123, Hs_sub, :antisymmetric; orth_method=nothing)
    @test Qsym * Qsym' ≈ Tsym_sub
    @test Qanti * Qanti' ≈ Tanti_sub

    # Invalid input: Hs not a subsystem of H
    @test_throws ArgumentError permutation_projector(H, [H1, H3], [idperm, swapperm])
end

@testitem "Permutation symmetry with a 120-state orbit" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_projector, symmetric_sector, AdditiveConstraint, _QR, _EIG
    using Combinatorics: permutations, parity
    # The orbit of the states with substates {-2,-1,0,1,2} has 5! = 120 states. The rank
    # tolerance must keep its sector vectors, although its projector columns have norm 1/sqrt(120)
    @spins S 2
    H = hilbert_space(S, 1:5, AdditiveConstraint(0; maps=s -> s.m))
    Hs = factors(H)
    perms = collect(permutations(1:5))
    for (ws, rank) in ((ones(120), 12), ([(-1.0)^parity(p) for p in perms], 1))
        P = permutation_projector(H, Hs, perms, ws, Float64)
        @test tr(P) ≈ rank
        for orth_method in (_QR(), _EIG()) # weights scaled so that P is half a projector
            @test size(symmetric_sector(H, Hs, (perms, ws ./ 240); orth_method, normalize=false), 2) == rank
        end
        for kwargs in ((; sparse=true), (; sparse=false), (; orth_method=_EIG()), (; normalize=false))
            Q = symmetric_sector(H, Hs, (perms, ws); kwargs...)
            @test size(Q, 2) == rank
            @test Q' * Q ≈ I
            @test Q * Q' ≈ P
        end
    end
end

@testitem "Permutation symmetry with complex weights" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_projector, symmetric_sector, _QR, _EIG
    # Irreducible representations of the cyclic group acting on 4 modes. The orbits of the
    # 16 Fock states have sizes 1, 1, 2, 4, 4, 4, so the sector dimensions are 6, 3, 4, 3.
    @fermions f
    H = hilbert_space(f, 1:4)
    Hs = factors(H)
    perms = [circshift(1:4, j) for j in 0:3]
    for (k, rank) in zip(0:3, (6, 3, 4, 3))
        ws = [cis(2pi * k * j / 4) for j in 0:3]
        P = permutation_projector(H, Hs, perms, ws, ComplexF64)
        for orth_method in (_QR(), _EIG())
            Q = symmetric_sector(H, Hs, (perms, ws), ComplexF64; orth_method)
            @test size(Q, 2) == rank
            @test Q' * Q ≈ I
            @test Q * Q' ≈ P
        end
    end
end
