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
    substates = unique_split_state(first(states), mapper)
    return _add_permutation_operator!(P, (states, state_index), mapper, _match_perm(perm, substates), weight)
end
function add_permutation_operators!(P, (states, state_index), mapper, perms, weights)
    substates = unique_split_state(first(states), mapper)
    for (perm, w) in zip(perms, weights)
        _add_permutation_operator!(P, (states, state_index), mapper, _match_perm(perm, substates), w)
    end
    return P
end

# Convert `perm` to the container type that `permute_state` uses for `substates`.
function _match_perm(perm, ::NTuple{N,Any}) where N
    length(perm) == N || throw(ArgumentError("The permutation `$perm` must have length $N, the number of factors."))
    return ntuple(i -> perm[i], Val(N))
end
_match_perm(perm, substates) = perm

function _add_permutation_operator!(P, (states, state_index), mapper, perm, weight)
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        iszero(i) && throw(ArgumentError("The permutation maps the basis state `$state` outside the space."))
        push_inds_amps!(P, i, j, weight)
    end
    return P
end
"""
    permutation_projector(H, Hs, perms[, weights[, T]]; normalize=true, sparse=true)

Build a group-averaged projector/operator from permutation operators:
`P = sum(weights[g] * R_g for g)`, where `R_g` permutes the factors `Hs` according to
`perms[g]`. Without `weights`, every weight is one.

`T` is the element type of `P`. It defaults to `Int` without `weights`, and otherwise to
the element type of `weights`. For floating point and complex element types, `P` is
replaced by zero if its largest entry is at most `sqrt(eps)` times the largest weight, so
that a sector absent from `H` gives an exactly zero `P` instead of roundoff.

Keywords:
- `normalize`: rescale `P` by `conj(tr(P)) / sum(abs2, P)`, which makes `P^2 == P` when `P`
  is proportional to a projector.
- `sparse`: return a `SparseMatrixCSC` if `true`, and a `Matrix` otherwise.

`Hs` may be either a full partition of `H` (covering all atomic factors) or a proper
subsystem list (covering only a subset of atomic factors). In the subsystem case the
projector is first constructed on `Hsub = tensor_product(Hs...)` and then embedded
back into `H` via `embed(Psub, Hsub => H)`.
"""
function permutation_projector(H::AbstractHilbertSpace, Hs, perms, weights=nothing, ::Type{T}=_default_projector_eltype(weights); normalize=true, sparse=true) where T
    isempty(perms) && throw(ArgumentError("`perms` must contain at least one permutation."))

    # Subsystem branch: Hs does not partition H, treat as a proper subsystem.
    if !ispartition(Hs, H)
        Hsub = subregion(tensor_product(Hs...), H)
        issubsystem(Hsub, H) || throw(ArgumentError("`Hs` must be a partition or a subsystem of `H`."))
        Psub = permutation_projector(Hsub, Hs, perms, weights, T; normalize, sparse)
        P = embed(Psub, Hsub => H)
        return sparse ? P : Matrix(P)
    end

    ws = if isnothing(weights)
        ones(T, length(perms))
    else
        length(weights) == length(perms) || throw(ArgumentError("`weights` and `perms` must have the same length, got $(length(weights)) and $(length(perms))."))
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
    add_permutation_operators!(accum, (states, si), mapper, perms, ws)
    P = if sparse
        SparseArrays.sparse(finalize!(accum, H))
    else
        accum
    end
    
    _is_negligible(P, ws) && (P = zero(P))
    if normalize
        tr2 = sum(abs2, P)
        tr1 = tr(P)
        iszero(tr2) || (P *= (conj(tr1) / tr2))
    end
    return P
end

# Floating point sums of weights that cancel exactly leave roundoff of order
# eps * sqrt(length(ws)) * maximum(abs, ws). For weights c * conj(χ(g)) of an irrep χ of
# dimension d, a nonzero P has a diagonal entry of at least |c| = maximum(abs, ws) / d.
# Comparing with sqrt(eps) * maximum(abs, ws) separates the two cases for any
# realistic group order and irrep dimension. Other element types are exact or symbolic.
function _is_negligible(P::AbstractArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}, ws)
    scale = maximum(abs, ws)
    return maximum(abs, P; init=zero(real(eltype(P)))) <= sqrt(eps(real(eltype(P)))) * scale
end
_is_negligible(P, ws) = false

function _default_projector_eltype(weights)
    isnothing(weights) && return Int
    eltype(weights) <: Number && return eltype(weights)
    typeof(first(weights)) <: Number && return typeof(first(weights))
    return Float64
end

"""
    symmetric_sector(H, Hs, sector=:symmetric, T=default; orth_method=:qr, check=true, normalize=true, sparse=true)

Return a basis for the sector of `H` selected by `sector`, under permutations of the
factors `Hs`.

`sector` is either a tuple `(perms, weights)`, with permutations of `1:length(Hs)` and
one weight per permutation, or `:symmetric` or `:antisymmetric`, which stand for all
permutations with weight one or the permutation sign.
`T` is the element type of the result. It defaults to `Float64` for `:symmetric` and
`:antisymmetric`, and for a tuple `(perms, weights)` to the floating point version of the
element type of `weights`, for example `ComplexF64` for complex weights.

For `:symmetric` and `:antisymmetric` with `Hs` a partition of `H` and `orth_method` not
`nothing`, the basis is built directly from the orbits of the basis states under
permutations, with one column per orbit that contributes to the sector. The basis then
has element type `float(T)`, and the keywords `normalize` and `sparse` have no effect.
In the remaining cases, the projector is built with
`permutation_projector(H, Hs, perms, weights, T; normalize, sparse)`, and `:symmetric` and
`:antisymmetric` require `Combinatorics.jl` to be loaded.
`orth_method` selects what is returned:
- `:qr`: a matrix with orthonormal columns spanning the sector, from a QR decomposition
  of the projector.
- `:eig`: a matrix with orthonormal columns spanning the sector, from the eigenvectors of
  the projector with nonzero eigenvalue.
- `nothing`: the projector itself.

`:qr` and `:eig` find the numerical rank with a tolerance relative to the scale of the
projector, so they assume that the projector is proportional to an orthogonal projector.
This holds when `perms` form a group and `weights` are a positive multiple of a character
of it or of its complex conjugate.

With `check=true` and a tuple `(perms, weights)`, `symmetric_sector` throws an
`ArgumentError` if `perms` does not contain the identity permutation or its weight is not
real and positive, and, for `orth_method` not `nothing`, if the projector is not
Hermitian. `check=false` skips these checks.

`Hs` may be either a full partition of `H` or a proper subsystem list; in the latter
case the sector is constructed on `tensor_product(Hs...)` and embedded back into `H`.
"""
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector=:symmetric, ::Type{T}=_default_sector_eltype(sector);
    orth_method=:qr, check=true, kwargs...) where T
    isempty(Hs) && throw(ArgumentError("`Hs` must contain at least one factor space."))
    method = _orth_method(orth_method)
    if sector isa Symbol && !isnothing(method) && ispartition(Hs, H)
        return _orbit_sector_basis(H, Hs, _sector_parity(sector), T; kwargs...)
    end

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("The generated weights and permutations must have the same length, got $(length(weights)) and $(length(perms))."))
    check && sector isa Tuple && _check_identity_weight(perms, weights)
    P = permutation_projector(H, Hs, perms, weights, T; kwargs...)
    isnothing(method) && return P
    check && sector isa Tuple && _check_hermitian(P)
    _sector_basis(P, method)
end

const _CHECK_HINT = "Pass `check=false` to skip this check."

function _check_identity_weight(perms, weights)
    found = false
    for (perm, w) in zip(perms, weights)
        _isidentity(perm) || continue
        found = true
        _is_positive_weight(w) || throw(ArgumentError("The weight of the identity permutation is $w, but for a positive multiple of a character it is real and positive. $_CHECK_HINT"))
    end
    found || throw(ArgumentError("`perms` does not contain the identity permutation, so it is not a group. $_CHECK_HINT"))
    return nothing
end
_isidentity(perm) = all(i -> perm[i] == i, eachindex(perm))

# Weights of other types, such as symbolic ones, are not checked.
_is_positive_weight(w) = true
_is_positive_weight(w::Union{Integer,Rational,AbstractFloat}) = w > 0
function _is_positive_weight(w::Complex{<:Union{Integer,Rational,AbstractFloat}})
    return abs(imag(w)) <= sqrt(eps(float(real(typeof(w))))) * abs(w) && real(w) > 0
end

_check_hermitian(P) = nothing
function _check_hermitian(P::AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})
    maximum(abs, P - P'; init=zero(real(eltype(P)))) <= _rank_tol(P) ||
        throw(ArgumentError("The projector is not Hermitian, so `weights` are not a multiple of a character of the group formed by `perms`, or of its complex conjugate. $_CHECK_HINT"))
    return nothing
end
_default_sector_eltype(sector) = Float64
function _default_sector_eltype((perms, weights)::Tuple)
    T = _default_projector_eltype(weights)
    return T <: Union{Real,Complex} ? float(T) : T
end
_orth_method(method::Union{_QR,_EIG,Nothing}) = method
function _orth_method(method)
    method === :qr && return _QR()
    method === :eig && return _EIG()
    throw(ArgumentError("`orth_method` must be `:qr`, `:eig` or `nothing`, got `$(repr(method))`."))
end
_resolve_sector_permutations_and_weights(Hs, (perms, weights), T) = (perms, weights) # for direct input of perms and weights

_sector_parity(sector::Symbol) = sector === :symmetric ? 1 : sector === :antisymmetric ? -1 :
    throw(ArgumentError("The sector must be `:symmetric` or `:antisymmetric`, got `:$sector`."))

# Orthonormal basis of the symmetric (`parity = 1`) or antisymmetric (`parity = -1`)
# sector of `H` under all permutations of the partition `Hs`.
# `permute_state` returns one basis state and discards the amplitude that `combine_states`
# returns, so every permutation acts as a permutation matrix on the basis. The
# group-averaged projector is then a sum over orbits of basis states, and each orbit
# contributes one column with entries `sign(g) / sqrt(length(orbit))`, where `g` maps the
# orbit's first state to the entry's state. In the antisymmetric case an orbit whose
# states are fixed by an odd permutation contributes nothing.
# The orbits are found with the adjacent transpositions, which generate all permutations.
function _orbit_sector_basis(H, Hs, parity, ::Type{T}; normalize=true, sparse=true) where T
    # `normalize` and `sparse` only affect the projector, which is never built here.
    states = basisstates(H)
    mapper = state_mapper(H, Hs)
    n = length(Hs)
    substates = unique_split_state(first(states), mapper)
    gens = [_match_perm([1:k-1; k + 1; k; k+2:n], substates) for k in 1:n-1]
    parent, rel, odd = _signed_orbits(states, Base.Fix2(state_index, H), mapper, gens, parity)

    d = length(parent)
    label = zeros(Int, d)  # column of each orbit root, -1 for dropped orbits
    firstsign = Int[]
    sizes = Int[]
    roots = Vector{Int}(undef, d)
    signs = Vector{Int}(undef, d)
    for x in 1:d
        r, s = _find_signed!(parent, rel, x)
        roots[x], signs[x] = r, s
        if iszero(label[r])
            if odd[r]
                label[r] = -1
            else
                push!(firstsign, s)
                push!(sizes, 0)
                label[r] = length(sizes)
            end
        end
        label[r] > 0 && (sizes[label[r]] += 1)
    end
    F = float(T)
    Q = zeros(F, d, length(sizes))
    for x in 1:d
        c = label[roots[x]]
        c > 0 && (Q[x, c] = signs[x] * firstsign[c] / sqrt(real(F)(sizes[c])))
    end
    return Q
end

# Signed union-find over the basis states: `parent` and `rel` encode a forest where the
# amplitude of state `x` in its orbit's column is `rel[x]` times that of `parent[x]`. Each
# generator application `i = g(j)` requires amplitude(i) = parity * amplitude(j). An orbit
# whose constraints contradict each other is marked in `odd`: such an orbit contains a
# closed walk with an odd number of transpositions, so a state in it is fixed by an odd
# permutation.
function _signed_orbits(states, state_index, mapper, gens, parity)
    d = length(states)
    parent = collect(1:d)
    rel = ones(Int, d)
    odd = falses(d)
    for (j, state) in enumerate(states), g in gens
        i = state_index(permute_state(state, mapper, g))
        iszero(i) && throw(ArgumentError("The permutation maps the basis state `$state` outside the space."))
        ri, si = _find_signed!(parent, rel, i)
        rj, sj = _find_signed!(parent, rel, j)
        if ri == rj
            si == parity * sj || (odd[ri] = true)
        else
            parent[ri] = rj
            rel[ri] = si * parity * sj
            odd[rj] |= odd[ri]
        end
    end
    return parent, rel, odd
end

# Root of `x` and the sign of `x` relative to the root, compressing the path to the root.
function _find_signed!(parent, rel, x)
    r, s = x, 1
    while parent[r] != r
        s *= rel[r]
        r = parent[r]
    end
    y, sy = x, s
    while parent[y] != r && parent[y] != y
        next, snext = parent[y], sy * rel[y]
        parent[y], rel[y] = r, sy
        y, sy = next, snext
    end
    return r, s
end

# For `P = p * Π` with `Π` an orthogonal projector, the pivots inside the rank are at least
# `p / sqrt(dim)` and the nonzero eigenvalues are `p`, while roundoff is of order `eps * p`.
# A tolerance of `sqrt(eps)` times the largest column norm (at most `p`) separates the two.
_rank_tol(P) = sqrt(eps(real(float(eltype(P))))) * sqrt(maximum(sum(abs2, P; dims=1); init=zero(real(eltype(P)))))

function _sector_basis(P::AbstractMatrix, ::_QR)
    F = qr(Matrix{float(eltype(P))}(P), ColumnNorm())
    k = count(>(_rank_tol(P)) ∘ abs, diag(F.R))
    return Matrix(F.Q)[:, 1:k]
end
function _sector_basis(P::SparseMatrixCSC, ::_QR)
    F = qr(P; tol=_rank_tol(P))
    return F.Q[invperm(F.prow), 1:rank(F)]
end
function _sector_basis(P::AbstractMatrix, ::_EIG)
    E = eigen!(Hermitian(Matrix{float(eltype(P))}(P)))
    tol = sqrt(eps(real(eltype(E.values)))) * maximum(abs, E.values; init=zero(eltype(E.values)))
    return E.vectors[:, abs.(E.values) .> tol]
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

    # The bases span the same sector; the basis vectors themselves may differ
    Qtriv = symmetric_sector(H, [H1, H2], :symmetric)
    Qsign = symmetric_sector(H, [H1, H2], :antisymmetric)
    @test Ptriv * Ptriv' ≈ Qtriv * Qtriv'
    @test Psign * Psign' ≈ Qsign * Qsign'

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

    # orth_method values
    using FermionicHilbertSpaces: _QR, _EIG
    @test symmetric_sector(H, [H1, H2], sympw; orth_method=:qr) == Ptriv
    @test symmetric_sector(H, [H1, H2], sympw; orth_method=_QR()) == Ptriv
    for w in (sympw, antisymw)
        Q = symmetric_sector(H, [H1, H2], w; orth_method=:eig)
        @test Q' * Q ≈ I
        @test Q * Q' ≈ symmetric_sector(H, [H1, H2], w; orth_method=nothing)
        @test symmetric_sector(H, [H1, H2], w; orth_method=_EIG()) == Q
    end
    @test_throws ArgumentError symmetric_sector(H, [H1, H2], sympw; orth_method=:svd)
    @test symmetric_sector(H, [H1, H2], sympw; orth_method=nothing, sparse=false) isa Matrix

    H3 = hilbert_space(f, 3:3)
    H123 = tensor_product(H1, H2, H3)
    Hs3 = [H1, H2, H3]
    perms3 = permutations(1:3)
    ws3 = [permutation_sign(p) for p in perms3]
    Pmanual3 = permutation_projector(H123, Hs3, perms3, ws3)
    Pauto3 = symmetric_sector(H123, Hs3, :antisymmetric, orth_method=nothing)
    @test Pmanual3 ≈ Pauto3
    # Permutations must have one entry per factor
    @test_throws ArgumentError permutation_operator(H123, Hs3, [2, 1])
    @test_throws ArgumentError permutation_projector(H123, Hs3, [[2, 1, 3, 4]])

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

    # The normalize keyword only rescales the projector, so the sector basis must not change
    @test size(symmetric_sector(H123, Hs3, :symmetric; normalize=false), 2) == size(symmetric_sector(H123, Hs3, :symmetric), 2)
    Qanti_unnorm = symmetric_sector(H123, Hs_sub, :antisymmetric; normalize=false)
    @test size(Qanti_unnorm, 2) == size(Qanti, 2)

    # Q*Q' should equal the full space projector (orth_method=nothing)
    Tsym_sub = symmetric_sector(H123, Hs_sub, :symmetric; orth_method=nothing)
    Tanti_sub = symmetric_sector(H123, Hs_sub, :antisymmetric; orth_method=nothing)
    @test Qsym * Qsym' ≈ Tsym_sub
    @test Qanti * Qanti' ≈ Tanti_sub
    @test Qanti_unnorm * Qanti_unnorm' ≈ Tanti_sub

    @test permutation_projector(H123, Hs_sub, [idperm, swapperm]; sparse=false) isa Matrix

    # Symbolic element types: the projector is returned without a rank computation
    using Symbolics: Num, symbolic_to_float, substitute, @variables
    @variables x
    Psym = symmetric_sector(H, [H1, H2], :symmetric, Num; orth_method=nothing)
    @test Float64.(symbolic_to_float.(Matrix(Psym))) ≈ symmetric_sector(H, [H1, H2], :symmetric; orth_method=nothing)
    Pwx = symmetric_sector(H, [H1, H2], ([idperm, swapperm], [1, x]), Num; orth_method=nothing, normalize=false, sparse=false)
    @test Float64.(symbolic_to_float.(Matrix(substitute.(Pwx, Ref(x => -1))))) ≈ symmetric_sector(H, [H1, H2], antisymw; orth_method=nothing, normalize=false, sparse=false)

    # Complex weights give a complex basis without passing the element type
    cw = ([idperm, swapperm], [1.0 + 0im, 1.0 + 0im])
    Qc = symmetric_sector(H, [H1, H2], cw)
    @test eltype(Qc) <: Complex
    @test size(Qc, 2) == 3
    @test eltype(symmetric_sector(H, [H1, H2], ([idperm, swapperm], [1, -1]))) == Float64

    # Irreps absent from the space give an empty basis, not a rank-1 projector from roundoff
    @spins S 1
    N = 4
    spinspace = hilbert_space(S, 1:N, SectorConstraint(sum; maps=st -> st.m))
    translations = [[mod1(i + r, N) for i in 1:N] for r in 0:(N-1)]
    for M in (4, 3) # one state (fully polarized), then four states (one spin lowered)
        sec = sector(M, spinspace)
        dims = map(0:(N-1)) do k
            w = [cis(2π * k * r / N) for r in 0:(N-1)]
            Q = symmetric_sector(sec, factors(sec), (translations, w))
            @test Q' * Q ≈ I
            for method in (:eig, :qr)
                @test size(symmetric_sector(sec, factors(sec), (translations, w); orth_method=method, normalize=false), 2) == size(Q, 2)
            end
            size(Q, 2)
        end
        @test sum(dims) == dim(sec)
        M == 4 && @test dims == [1, 0, 0, 0]
    end

    # Invalid input: Hs not a subsystem of H
    @test_throws ArgumentError permutation_projector(H, [H1, H3], [idperm, swapperm])
end

@testitem "Sector bases from the projector" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_projector, symmetric_sector, AdditiveConstraint
    using Combinatorics: permutations, parity
    function check(H, Hs, (perms, ws), rank)
        P = permutation_projector(H, Hs, perms, ws, eltype(ws))
        for orth_method in (:qr, :eig), sparse in (true, false), normalize in (true, false)
            Q = symmetric_sector(H, Hs, (perms, ws); orth_method, sparse, normalize)
            @test size(Q, 2) == rank
            @test Q' * Q ≈ I
            @test Q * Q' ≈ P
        end
    end
    # An orbit of 5! = 120 states, whose projector columns have small norm
    @spins S 2
    H = hilbert_space(S, 1:5, AdditiveConstraint(0; maps=s -> s.m))
    perms = collect(permutations(1:5))
    check(H, factors(H), (perms, fill(1 / 240, 120)), 12)
    check(H, factors(H), (perms, [(-1.0)^parity(p) for p in perms]), 1)
    # Complex characters of the cyclic group on 4 modes
    @fermions f
    H = hilbert_space(f, 1:4)
    for (k, rank) in zip(0:3, (6, 3, 4, 3))
        check(H, factors(H), ([circshift(1:4, j) for j in 0:3], [cis(2pi * k * j / 4) for j in 0:3]), rank)
    end
end

@testitem "Symmetric and antisymmetric sectors from orbits" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, symmetric_sector, AdditiveConstraint, _QR, _EIG
    using Combinatorics: permutations
    @fermions f
    @spins S 1 // 2
    @bosons b
    sites = [tensor_product(hilbert_space(f[k, :u]), hilbert_space(f[k, :d])) for k in 1:3]
    bosons = [hilbert_space(b[k], 3) for k in 1:3]
    mixed = [tensor_product(hilbert_space(S[k]), hilbert_space(f[10+k])) for k in 1:3]
    Hspins = hilbert_space(S, 1:5, AdditiveConstraint(1 // 2; maps=s -> s.m))
    Hferm = hilbert_space(f, 1:5, NumberConservation(2))
    cases = [(hilbert_space(f, 1:4), factors(hilbert_space(f, 1:4))), (Hferm, factors(Hferm)),
        (Hspins, factors(Hspins)), (tensor_product(sites), sites),
        (tensor_product(bosons), bosons), (tensor_product(mixed), mixed)]
    for (H, Hs) in cases
        # The orbit construction relies on every permutation acting as a permutation matrix
        for perm in permutations(1:length(Hs))
            R = permutation_operator(H, Hs, perm)
            @test all(x -> x == 0 || x == 1, R)
            @test all(==(1), sum(R; dims=1)) && all(==(1), sum(R; dims=2))
        end
        for sector in (:symmetric, :antisymmetric)
            P = symmetric_sector(H, Hs, sector; orth_method=nothing)
            for orth_method in (_QR(), _EIG())
                Q = symmetric_sector(H, Hs, sector; orth_method)
                @test Q isa Matrix{Float64}
                @test size(Q, 2) == round(Int, tr(P))
                @test Q' * Q ≈ I
                @test Q * Q' ≈ P
            end
        end
    end
    # The orbit basis is computed in the requested precision
    Qbig = symmetric_sector(cases[1]..., :symmetric, BigFloat)
    @test eltype(Qbig) == BigFloat && size(Qbig, 2) == 5 && maximum(abs, Qbig' * Qbig - I) < 1e-60
    # Single factor: the sector is the whole space
    H1 = hilbert_space(f, 1:1)
    @test symmetric_sector(H1, [H1], :antisymmetric) ≈ I
    @test_throws ArgumentError symmetric_sector(cases[1]..., :invalid)
end

@testitem "Checks on the weights of symmetric_sector" begin
    @fermions f
    H = hilbert_space(f, 1:3)
    Hs = factors(H)
    cyclic = [circshift(1:3, j) for j in 0:2]
    chi = [cis(2pi * j / 3) for j in 0:2]
    for ws in (cis(pi / 4) .* chi, -ones(3))
        @test_throws ArgumentError symmetric_sector(H, Hs, (cyclic, ws); orth_method=nothing)
    end
    # normalize divides out a complex scale
    @test symmetric_sector(H, Hs, (cyclic, cis(pi / 4) .* chi); orth_method=nothing, check=false) ≈
          symmetric_sector(H, Hs, (cyclic, chi); orth_method=nothing)
    @test_throws ArgumentError symmetric_sector(H, Hs, (cyclic[2:3], chi[2:3]))
    # The Hermitian check runs only when a basis is computed
    @test_throws ArgumentError symmetric_sector(H, Hs, (cyclic, [1.0, 1.0, 0.0]))
    @test symmetric_sector(H, Hs, (cyclic, [1.0, 1.0, 0.0]); orth_method=nothing) isa AbstractMatrix
end
