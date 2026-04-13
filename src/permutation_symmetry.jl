#_as_tuple_spaces(Hs) = Hs isa Tuple ? Hs : Tuple(Hs)

function _validate_permutation(perm, n::Int)
    length(perm) == n || throw(ArgumentError("Permutation length must match number of partition blocks"))
    all(1 <= p <= n for p in perm) || throw(ArgumentError("Permutation entries must be in 1:$n"))
    length(unique(perm)) == n || throw(ArgumentError("Permutation must contain each index exactly once"))
    return Tuple(perm)
end

"""
    permute_state(state, mapper, perm)

"""
function permute_state(state, mapper, perm)
    splits, wsplits = split_state(state, mapper)
    outstates = typeof(state)[]
    outamps = Int[]

    for (substates, wsplit) in zip(splits, wsplits)
        permuted = substates[perm]
        states2, w2 = combine_states(permuted, mapper)
        for (state2, amp2) in zip(states2, w2)
            push!(outstates, state2)
            push!(outamps, wsplit * amp2)
        end
    end
    return outstates, outamps
end

"""
    permutation_operator(H, Hs, perm)

Build the matrix representation `R_perm` of a partition permutation in basis `H`.
"""
function permutation_operator(H::AbstractHilbertSpace, Hs, perm)
    spaces = Hs
    d = dim(H)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, d)
    sizehint!(J, d)
    sizehint!(V, d)
    mapper = state_mapper(H, spaces)
    for (j, state) in enumerate(basisstates(H))
        states2, amps2 = permute_state(state, mapper, perm)
        for (state2, amp2) in zip(states2, amps2)
            i = state_index(state2, H)
            ismissing(i) && throw(ArgumentError("Permutation maps a basis state outside the constrained space"))
            push!(I, i)
            push!(J, j)
            push!(V, amp2)
        end
    end
    return SparseArrays.sparse!(I, J, V, d, d)
end

"""
    permutation_projector(H, Hs, perms; weights=nothing, normalize=true)

Build a group-averaged projector/operator from permutation operators:
`P = sum(weights[g] * R_g for g)`.
"""
function permutation_projector(H::AbstractHilbertSpace, Hs, perms; weights=nothing, normalize=true)
    # perms_t = map(p -> _validate_permutation(p, length(Hs)), perms)
    isempty(perms) && throw(ArgumentError("At least one permutation is required"))

    ws = if isnothing(weights)
        ones(Float64, length(perms))
    else
        length(weights) == length(perms) || throw(ArgumentError("weights must match perms length"))
        float.(collect(weights))
    end

    d = dim(H)
    P = spzeros(Float64, d, d)
    for (perm, w) in zip(perms, ws)
        P .+= w .* permutation_operator(H, Hs, perm)
    end

    if normalize
        tr2 = tr(P^2)
        tr1 = tr(P)
        iszero(tr2) || (P *= (tr1 / tr2))
    end
    return P
end

function _independent_columns(M::AbstractMatrix; atol=1e-10)
    isempty(M) && return Int[]
    F = qr(Matrix(M))
    rdiag = abs.(diag(F.R))
    r = count(>(atol), rdiag)
    return collect(1:r)
end

"""
    symmetry_basis_transformation(H, Hs, perms; weights=nothing, atol=1e-10)

Construct a basis transformation matrix `T` from the image of the group-averaged projector.
Columns of `T` form an orthonormal basis for the projected subspace.
"""
function symmetry_basis_transformation(H::AbstractHilbertSpace, Hs, perms; weights=nothing, atol=1e-10)
    P = permutation_projector(H, Hs, perms; weights=weights, normalize=true)
    cols = _independent_columns(P; atol=atol)
    isempty(cols) && return Matrix{Float64}(undef, dim(H), 0)
    Q, _ = qr(Matrix(P[:, cols]))
    return Matrix(Q)
end

@testitem "Permutation symmetry" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permute_state, permutation_operator, permutation_projector, symmetry_basis_transformation
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

    Ttriv = symmetry_basis_transformation(H, [H1, H2], perms; weights=(1.0, 1.0)) #These are wrong
    Tsign = symmetry_basis_transformation(H, [H1, H2], perms; weights=(1.0, -1.0)) #there are wrong
    @test size(Tsign, 2) == 1
    @test Tsign' * Tsign ≈ I
    @test Tsign' * Rswap * Tsign ≈ -I
end