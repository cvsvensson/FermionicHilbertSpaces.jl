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
        float.(collect(weights))
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


function symmetry_basis_states(H::AbstractHilbertSpace, Hs, perms, ::Type{T}=Float64;
    weights=nothing, atol=1e-10) where T
    mapper = state_mapper(H, Hs)
    d = dim(H)
    ws = isnothing(weights) ? ones(d) : collect(weights)

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
    using FermionicHilbertSpaces: permute_state, permutation_operator, permutation_projector, symmetry_basis_transformation, symmetry_basis_states
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
    @test size(Tsign, 2) == 1
    @test Tsign' * Tsign ≈ I
    @test Tsign' * Rswap * Tsign ≈ -I
end