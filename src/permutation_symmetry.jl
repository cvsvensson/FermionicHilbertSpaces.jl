"""
    permute_state(state, mapper, perm)

"""
function permute_state(state, mapper, perm)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = substates[perm]
    newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
    return newstate
end

"""
    permutation_operator(H, Hs, perm)

Build the matrix representation `R_perm` of a partition permutation in basis `H`.
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
function permutation_projector(H::AbstractHilbertSpace, Hs, perms, ::Type{T}=Float64; weights=nothing, normalize=true) where T
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
        tr2 = tr(P^2)
        tr1 = tr(P)
        iszero(tr2) || (P *= (tr1 / tr2))
    end
    return P
end

"""
    symmetry_basis_transformation(H, Hs, perms; weights=nothing, cutoff=0.9)

Construct a basis transformation matrix `T` from the image of the group-averaged projector.
Columns of `T` form an orthonormal basis for the projected subspace.
"""
function symmetry_basis_transformation(H::AbstractHilbertSpace, Hs, perms, ::Type{T}=Float64;
    weights=nothing, cutoff=0.9) where T
    P = permutation_projector(H, Hs, perms, T; weights=weights, normalize=true)
    # P is Hermitian; eigenvalues cluster at 0 and 1
    E = eigen(Hermitian(Matrix(P)))
    keep = findall(>(cutoff), E.values)  # λ ≈ 1 subspace
    isempty(keep) && return Matrix{T}(undef, dim(H), 0)
    return E.vectors[:, keep]  # already orthonormal
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
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector::Symbol=:symmetric, ::Type{T}=Float64;
    method::Symbol=:states, normalize::Bool=true, cutoff=0.9, atol=1e-10) where T
    isempty(Hs) && throw(ArgumentError("At least one factor space is required"))

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("Generated weights must match generated permutations"))

    if method === :states
        return symmetry_basis_states(H, Hs, perms, T; weights=weights, atol=atol)
    elseif method === :transformation
        return symmetry_basis_transformation(H, Hs, perms, T; weights=weights, cutoff=cutoff)
    elseif method === :projector
        return permutation_projector(H, Hs, perms, T; weights=weights, normalize=normalize)
    end

    throw(ArgumentError("Unknown method :$(method). Expected one of :states, :transformation, :projector"))
end

function symmetry_basis_states(H::AbstractHilbertSpace, Hs, perms, ::Type{T}=Float64;
    weights=nothing, atol=1e-10) where T
    isempty(perms) && throw(ArgumentError("At least one permutation is required"))
    mapper = state_mapper(H, Hs)
    d = dim(H)
    ws = if isnothing(weights)
        Fill(one(T), length(perms))
    else
        length(weights) == length(perms) || throw(ArgumentError("weights must match perms length"))
        weights
    end

    visited = falses(d)
    result_cols = Vector{T}[]

    for (j, state_j) in enumerate(basisstates(H))
        visited[j] && continue
        visited[j] = true

        v = zeros(T, d)
        for (perm, w) in zip(perms, ws)
            newstate = permute_state(state_j, mapper, perm)
            i = state_index(newstate, H)
            ismissing(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
            visited[i] = true   # mark orbit member
            v[i] += w
        end

        n = norm(v)
        n > atol && push!(result_cols, v ./ n)
    end

    isempty(result_cols) && return Matrix{T}(undef, d, 0)
    return reduce(hcat, result_cols)
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