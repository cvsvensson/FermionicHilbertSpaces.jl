struct _QR end
struct _EIG end

function permute_state(state, mapper, perm::AbstractVector)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = substates[perm]
    # permuted = TupleTools.permute(substates, Tuple(perm))
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
function permutation_operator(H::AbstractHilbertSpace, Hs, perm, ::Type{T}=Float64) where T
    P = zeros(T, dim(H), dim(H))
    add_permutation_operator!(P, (basisstates(H), Base.Fix2(state_index, H)), state_mapper(H, Hs), perm, one(T))
end
function add_permutation_operator!(P, (states, state_index), mapper, perm, weight=1)
    if unique_split_state(first(states), mapper) isa Tuple
        return _add_permutation_operator_tuple!(P, (states, state_index), mapper, Tuple(perm), weight)
    else
        return _add_permutation_operator!(P, (states, state_index), mapper, perm, weight)
    end
end

function _add_permutation_operator_tuple!(P, (states, state_index), mapper, perm, weight=1)
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        iszero(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
        P[i, j] += weight
    end
    return P
end
function _add_permutation_operator!(P, (states, state_index), mapper, perm, weight=1)
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        iszero(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
        P[i, j] += weight
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
function permutation_projector(H::AbstractHilbertSpace, Hs, perms, weights=nothing, ::Type{T}=Float64; normalize=true) where T
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
    P = zeros(T, d, d)
    mapper = state_mapper(H, Hs)
    states = basisstates(H)
    si = Base.Fix2(state_index, H)
    for (perm, w) in zip(perms, ws)
        add_permutation_operator!(P, (states, si), mapper, perm, w)
    end

    if normalize
        tr2 = sum(abs2, P)
        tr1 = tr(P)
        iszero(tr2) || (P *= (tr1 / tr2))
    end
    return P
end

"""
    symmetric_sector(H, Hs, sector=:symmetric, T=Float64; normalize=true, cutoff=0.9, atol=1e-10)

Construct symmetry-adapted objects from symbolic sector selectors.

Supported sectors are `:symmetric` and `:antisymmetric`, generated over the full
permutation group `S_n` for `n = length(Hs)`. This requires the `Combinatorics.jl`
weak extension to be available.

`Hs` may be either a full partition of `H` or a proper subsystem list; in the latter
case the sector is constructed on `tensor_product(Hs...)` and embedded back into `H`.
"""
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector=:symmetric, ::Type{T}=Float64;
    orth_method=_QR(), kwargs...) where T
    isempty(Hs) && throw(ArgumentError("At least one factor space is required"))

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("Generated weights must match generated permutations"))
    P = permutation_projector(H, Hs, perms, weights, T; kwargs...)
    _remove_columns(P, orth_method)
end
_resolve_sector_permutations_and_weights(Hs, (perms, weights), T) = (perms, weights) # for direct input of perms and weights

_remove_columns(P::AbstractMatrix, alg) = _remove_columns(Matrix(P), alg) # ensure we have a dense matrix for the decomposition
function _remove_columns(P::Matrix, ::_QR)
    F = qr(P, ColumnNorm())
    big_cols = findall(>(0.1) ∘ abs, diag(F.R))
    return F.Q[:, big_cols]
end
function _remove_columns(P::Matrix, ::_EIG)
    E = eigen(Hermitian(P))
    keep = findall(>(0.9) ∘ abs, E.values)  # λ ≈ 1 subspace
    return E.vectors[:, keep]  # already orthonormal
end
function _remove_columns(P::AbstractMatrix, ::Nothing)
    return P
end


@testitem "Permutation symmetry" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, permutation_projector, symmetric_sector
    using Combinatorics: permutations

    permutation_sign(perm) = isodd(sum(perm[i] > perm[j] for i in 1:length(perm)-1 for j in i+1:length(perm))) ? -1.0 : 1.0

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
