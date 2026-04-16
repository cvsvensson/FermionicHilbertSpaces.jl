struct Orbit end
struct Projector end
struct _QR end
struct _EIG end

function permute_state(state, mapper, perm)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = substates[perm]
    # permuted = TupleTools.permute(substates, Tuple(perm))
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
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        ismissing(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
        P[i, j] += weight
    end
    return P
end
"""
    permutation_projector(H, Hs, perms; weights=nothing, normalize=true)

Build a group-averaged projector/operator from permutation operators:
`P = sum(weights[g] * R_g for g)`.
"""
permutation_projector(H::AbstractHilbertSpace, args...; kwargs...) = get_projector(Projector(), H, args...; kwargs...)
function get_projector(::Projector, H::AbstractHilbertSpace, Hs, perms, weights, ::Type{T}=Float64; normalize=true) where T
    isempty(perms) && throw(ArgumentError("At least one permutation is required"))

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
    symmetric_sector(H, Hs, sector=:symmetric, T=Float64; method=:states, normalize=true, cutoff=0.9, atol=1e-10)

Construct symmetry-adapted objects from symbolic sector selectors.

Supported sectors are `:symmetric` and `:antisymmetric`, generated over the full
permutation group `S_n` for `n = length(Hs)`. This requires the `Combinatorics.jl`
weak extension to be available.

Set `method` to:
- `:states` (default): return orbit-based symmetry basis states.
- `:transformation`: return an orthonormal basis from projector eigenspace.
- `:projector`: return the permutation projector/operator itself.
"""
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector=:symmetric, ::Type{T}=Float64;
    method=Projector(), orth_method=_QR(), kwargs...) where T
    isempty(Hs) && throw(ArgumentError("At least one factor space is required"))

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("Generated weights must match generated permutations"))
    P = get_projector(method, H, Hs, perms, weights; kwargs...)
    _remove_columns(P, orth_method)
end
_resolve_sector_permutations_and_weights(Hs, (perms, weights), T) = (perms, weights) # for direct input of perms and weights

function get_projector(::Orbit, H::AbstractHilbertSpace, Hs, perms, weights, ::Type{T}=Float64; atol=sqrt(eps(T)), orth=_QR()) where T
    d = dim(H)
    mapper = state_mapper(H, Hs)
    P = zeros(T, d, d)
    for (j, state) in enumerate(basisstates(H))
        substates = collect(unique_split_state(state, mapper)) #only support unique splits for now
        for (perm, w) in zip(perms, weights)
            permuted = substates[perm]
            newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
            i = state_index(newstate, H)
            P[i, j] += w
        end
    end
    return P
end

function _remove_columns(P, ::_QR)
    F = qr(P)
    big_cols = findall(>(0.1) ∘ abs, diag(F.R))
    return F.Q[:, big_cols]
end
function _remove_columns(P, ::_EIG)
    E = eigen(Hermitian((P)))
    keep = findall(>(0.9) ∘ abs, E.values)  # λ ≈ 1 subspace
    return E.vectors[:, keep]  # already orthonormal
end
function _remove_columns(P, ::Nothing)
    return P
end


@testitem "Permutation symmetry" begin
    using LinearAlgebra
    using Combinatorics: permutations
    using FermionicHilbertSpaces: permutation_operator, permutation_projector, symmetry_basis_transformation, symmetry_basis_states, symmetric_sector

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
    Psign = permutation_projector(H, [H1, H2], perms; weights=(1.0, -1.0))

    @test Ptriv * Ptriv ≈ Ptriv
    @test Psign * Psign ≈ Psign
    @test Ptriv * Psign ≈ 0I

    Ttriv = symmetry_basis_states(H, [H1, H2], [idperm, swapperm]; weights=[1.0, 1.0])
    Tsign = symmetry_basis_states(H, [H1, H2], [idperm, swapperm]; weights=[1.0, -1.0])

    Ttriv2 = symmetry_basis_transformation(H, [H1, H2], perms; weights=(1.0, 1.0))
    Tsign2 = symmetry_basis_transformation(H, [H1, H2], perms; weights=(1.0, -1.0))
    Tsym = symmetric_sector(H, [H1, H2], :symmetric)
    Tanti = symmetric_sector(H, [H1, H2], :antisymmetric)
    Panti = symmetric_sector(H, [H1, H2], :antisymmetric; method=:projector)
    @test size(Tsign, 2) == 1
    @test Tsign' * Tsign ≈ I
    @test Tsign' * Rswap * Tsign ≈ -I
    @test Tsym' * Tsym ≈ I
    @test Tanti' * Tanti ≈ I
    @test Tsym * Tsym' ≈ Ptriv
    @test Tanti * Tanti' ≈ Psign
    @test Panti ≈ Psign

    H3 = hilbert_space(f, 3:3)
    H123 = tensor_product(H1, H2, H3)
    Hs3 = [H1, H2, H3]
    perms3 = permutations(1:3)
    ws3 = [permutation_sign(p) for p in perms3]
    Pmanual3 = permutation_projector(H123, Hs3, perms3; weights=ws3)
    Pauto3 = symmetric_sector(H123, Hs3, :antisymmetric; method=:projector)
    @test Pmanual3 ≈ Pauto3

    @test_throws ArgumentError symmetric_sector(H, [H1, H2], :invalid)
end