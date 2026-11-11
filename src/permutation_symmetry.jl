struct _QR end
struct _EIG end

function permute_state(state, mapper, perm::AbstractVector, f = (i, s) -> s)
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = map(i -> f(i, substates[perm[i]]), eachindex(perm))
    newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
    return newstate
end
# `ntuple` rather than `map` (as in `TupleTools.permute`), which allocates for more than 32 factors
function permute_state(state, mapper, perm::NTuple{N}, f = (i, s) -> s) where {N}
    substates = unique_split_state(state, mapper) #only support unique splits for now
    permuted = ntuple(i -> f(i, substates[perm[i]]), Val(N))
    newstate = only(first(combine_states(permuted, mapper))) # only support unique combination for now
    return newstate
end

# The fermionic permutation `R`, with `R * c' * R'` the corresponding operator in the factor that `perm` moves `c` to,
# maps the basis state `x` to `u(x) * uπ(y) * y`, where `y` is the permuted state and `u` and `uπ` are the phase factors
# of the partitions `Hs` and `Hs[invperm(perm)]`. When `u` is `Returns(1)`, as for spaces without fermions, so is `uπ`,
# since permuting `Hs` leaves the kind of each factor's mapper unchanged.
function _permutation_phases(H, Hs, mapper, phase_factors)
    phase_factors || return perm -> (j, i) -> 1
    pfu = phase_factor_u(mapper)
    pfu == Returns(1) && return perm -> (j, i) -> 1
    states = basisstates(H)
    u = map(pfu, states)
    return function (perm)
        uπ = map(phase_factor_u(state_mapper(H, Hs[invperm(collect(perm))])), states)
        return (j, i) -> u[j] * uπ[i]
    end
end

"""
    permutation_operator(H, Hs, perm; phase_factors=true)

Gives the sparse matrix representation on `H` of a permutation `perm` of the partition `Hs`.
For fermionic factors it includes the sign of reordering the fermions, so that it maps `c'` in factor `perm[k]` to the
corresponding operator in factor `k`. With `phase_factors=false`, it permutes the basis states without signs.
"""
function permutation_operator(H::AbstractHilbertSpace, Hs, perm, ::Type{T} = Int; phase_factors = true) where {T}
    entries = (Int[], Int[], T[])
    mapper = state_mapper(H, Hs)
    phase_of = _permutation_phases(H, Hs, mapper, phase_factors)
    add_permutation_operators!(entries, (basisstates(H), Base.Fix2(state_index, H)), mapper, (perm,), (one(T),), phase_of)
    return sparse(entries..., dim(H), dim(H))
end
# `phase_of(perm)` gives the phase function of a permutation, see `_permutation_phases`.
function add_permutation_operators!(P, (states, state_index), mapper, perms, weights, phase_of)
    substates = unique_split_state(first(states), mapper)
    for (perm, w) in zip(perms, weights)
        perm = _match_perm(perm, substates)
        _add_permutation_operator!(P, (states, state_index), mapper, perm, w, phase_of(perm))
    end
    return P
end

# Convert `perm` to the container type that `permute_state` uses for `substates`.
function _match_perm(perm, ::NTuple{N,Any}) where N
    length(perm) == N || throw(ArgumentError("The permutation `$perm` must have length $N, the number of factors."))
    isperm(perm) || throw(ArgumentError("`$perm` is not a permutation."))
    return ntuple(i -> perm[i], Val(N))
end
_match_perm(perm, substates) = perm

function _add_permutation_operator!(P, (states, state_index), mapper, perm, weight, phase)
    for (j, state) in enumerate(states)
        state2 = permute_state(state, mapper, perm)
        i = state_index(state2)
        iszero(i) && throw(ArgumentError("The permutation maps the basis state `$state` outside the space."))
        push_inds_amps!(P, i, j, weight * phase(j, i))
    end
    return P
end
"""
    permutation_projector(H, Hs, perms[, weights[, T]]; normalize=true, sparse=true, phase_factors=true)

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
- `phase_factors`: include the fermionic signs of the permutations, as in [`permutation_operator`](@ref).

`Hs` may be either a full partition of `H` (covering all atomic factors) or a proper
subsystem list (covering only a subset of atomic factors). In the subsystem case the
projector is first constructed on `Hsub = tensor_product(Hs...)` and then embedded
back into `H` via `embed(Psub, Hsub => H)`.
"""
function permutation_projector(H::AbstractHilbertSpace, Hs, perms, weights = nothing, ::Type{T} = _default_projector_eltype(weights); normalize = true, sparse = true, phase_factors = true) where {T}
    isempty(perms) && throw(ArgumentError("`perms` must contain at least one permutation."))

    # Subsystem branch: Hs does not partition H, treat as a proper subsystem.
    if !ispartition(Hs, H)
        Hsub = subregion(tensor_product(Hs...), H)
        issubsystem(Hsub, H) || throw(ArgumentError("`Hs` must be a partition or a subsystem of `H`."))
        Psub = permutation_projector(Hsub, Hs, perms, weights, T; normalize, sparse, phase_factors)
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
    add_permutation_operators!(accum, (states, si), mapper, perms, ws, _permutation_phases(H, Hs, mapper, phase_factors))
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
    symmetric_sector(H, Hs, sector=:symmetric, T=default; orth_method=:qr, check=true, normalize=true, sparse=true, phase_factors=true)

Return a matrix whose orthonormal columns span the sector of `H` selected by `sector`, under permutations
of the factors `Hs` (all or some of the factors of `H`). `sector` is `:symmetric`, `:antisymmetric`, or a tuple
`(perms, weights)` with permutations of `1:length(Hs)` and one weight per permutation. For an orthonormal basis,
`perms` must form a group and `weights` a positive multiple of a character of it.
`T` is the element type, by default `Float64`, or the float type of the weights.

`orth_method` is `:qr` or `:eig`, or `nothing` to return the projector, scaled to `P^2 == P` unless
`normalize=false` and dense if `sparse=false`. `check=false` skips the validation of `(perms, weights)`.
`phase_factors` includes the fermionic signs of the permutations, as in [`permutation_operator`](@ref).
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
    if sector isa Tuple && !isnothing(method) && ispartition(Hs, H)
        return _orbit_block_basis(H, Hs, perms, weights, method, T; check, kwargs...)
    end
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

_check_hermitian(P, tol = nothing) = nothing
function _check_hermitian(P::AbstractMatrix{<:Union{AbstractFloat, Complex{<:AbstractFloat}}}, tol = _rank_tol(P))
    maximum(abs, P - P'; init = zero(real(eltype(P)))) <= tol ||
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
# Every permutation maps each basis state to one basis state times a sign, the fermionic
# phase. The group-averaged projector is then a sum over orbits of basis states, and each
# orbit contributes one column with entries `±1 / sqrt(length(orbit))`, where the sign is
# the phase of the permutation `g` mapping the orbit's first state to the entry's state,
# times `sign(g)` in the antisymmetric case. An orbit containing a state that a
# permutation maps to minus itself, including these signs, contributes nothing.
# The orbits are found with the adjacent transpositions, which generate all permutations.
function _orbit_sector_basis(H, Hs, parity, ::Type{T}; normalize = true, sparse = true, phase_factors = true) where {T}
    # `normalize` and `sparse` only affect the projector, which is never built here.
    states = basisstates(H)
    mapper = state_mapper(H, Hs)
    n = length(Hs)
    substates = unique_split_state(first(states), mapper)
    gens = [_match_perm([1:k-1; k + 1; k; k+2:n], substates) for k in 1:n-1]
    phases = map(_permutation_phases(H, Hs, mapper, phase_factors), gens)
    parent, rel, odd = _signed_orbits(states, Base.Fix2(state_index, H), mapper, gens, phases, parity)

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
    rows = filter(x -> label[roots[x]] > 0, 1:d)
    cols = [label[roots[x]] for x in rows]
    vals = [F(signs[x] * firstsign[c]) / sqrt(real(F)(sizes[c])) for (x, c) in zip(rows, cols)]
    return SparseArrays.sparse(rows, cols, vals, d, length(sizes))
end

# Signed union-find over the basis states: `parent` and `rel` encode a forest where the
# amplitude of state `x` in its orbit's column is `rel[x]` times that of `parent[x]`. Each
# generator application `i = g(j)` with phase `p` requires amplitude(i) = parity * p * amplitude(j).
# An orbit whose constraints contradict each other is marked in `odd`: a state in it is
# mapped to minus itself, so the orbit is absent from the sector.
function _signed_orbits(states, state_index, mapper, gens, phases, parity)
    d = length(states)
    parent = collect(1:d)
    rel = ones(Int, d)
    odd = falses(d)
    for (j, state) in enumerate(states), (g, phase) in zip(gens, phases)
        newstate = permute_state(state, mapper, g)
        i = state_index(newstate)
        iszero(i) && throw(ArgumentError("The permutation maps the basis state `$state` outside the space."))
        p = parity * phase(j, i)
        ri, si = _find_signed!(parent, rel, i)
        rj, sj = _find_signed!(parent, rel, j)
        if ri == rj
            si == p * sj || (odd[ri] = true)
        else
            parent[ri] = rj
            rel[ri] = si * p * sj
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

function _sector_basis(P::AbstractMatrix, ::_QR, tol = _rank_tol(P))
    F = qr(Matrix{float(eltype(P))}(P), ColumnNorm())
    k = count(>(tol) ∘ abs, diag(F.R))
    return Matrix(F.Q)[:, 1:k]
end
function _sector_basis(P::SparseMatrixCSC, ::_QR, tol = _rank_tol(P))
    F = qr(P; tol)
    return F.Q[invperm(F.prow), 1:rank(F)]
end
function _sector_basis(P::AbstractMatrix, ::_EIG, tol = _rank_tol(P))
    E = eigen!(Hermitian(Matrix{float(eltype(P))}(P)))
    return E.vectors[:, abs.(E.values) .> tol]
end

# `P = sum(weights[g] * R_g)` is block diagonal over the orbits of the basis states under the
# group `perms`, and each block is orthonormalized on its own. The rank tolerance comes from
# the whole `P`, since a block that is pure roundoff has no scale of its own. For a
# one-dimensional character each block has rank at most one and is spanned by its column at
# the first state of the orbit, so only that column is computed.
function _orbit_block_basis(H, Hs, perms, weights, method, ::Type{T}; check = true, normalize = true, sparse = true, phase_factors = true) where {T}
    states = basisstates(H)
    mapper = state_mapper(H, Hs)
    substates = unique_split_state(first(states), mapper)
    gs = [_match_perm(perm, substates) for perm in perms]
    phases = map(_permutation_phases(H, Hs, mapper, phase_factors), gs)
    onedim = _has_equal_magnitudes(weights)
    onedim && check && _check_1d_character(perms, weights)

    F = float(T)
    d = length(states)
    si = Base.Fix2(state_index, H)
    owner = zeros(Int, d)
    pos = zeros(Int, d)
    entries = (Int[], Int[], F[])
    orbits = Vector{Int}[]
    blocks = Matrix{F}[]
    for s in eachindex(states)
        iszero(owner[s]) || continue
        _permutation_images!(entries, [s], states, si, mapper, gs, weights, phases)
        orbit = Int[]
        push!(orbits, orbit)
        for t in entries[1]
            iszero(owner[t]) && (push!(orbit, t); owner[t] = length(orbits); pos[t] = length(orbit))
        end
        onedim || _permutation_images!(entries, orbit, states, si, mapper, gs, weights, phases)
        B = zeros(F, length(orbit), onedim ? 1 : length(orbit))
        for (i, j, w) in zip(entries...)
            owner[i] == length(orbits) || throw(ArgumentError("`perms` is not closed under composition, so it is not a group."))
            B[pos[i], j] += w
        end
        push!(blocks, _is_negligible(B, weights) ? zero(B) : B)
    end
    bases = if onedim
        [iszero(B) ? B[:, 1:0] : B / sqrt(sum(abs2, B)) for B in blocks]
    else
        if normalize
            tr2 = sum(B -> sum(abs2, B), blocks)
            iszero(tr2) || (s = conj(sum(tr, blocks)) / tr2; blocks .*= s)
        end
        tol = maximum(_rank_tol, blocks)
        check && foreach(B -> _check_hermitian(B, tol), blocks)
        [_sector_basis(B, method, tol) for B in blocks]
    end
    rows, cols, vals = Int[], Int[], F[]
    c = 0
    for (orbit, basis) in zip(orbits, bases), col in eachcol(basis)
        c += 1
        append!(rows, orbit); append!(cols, Iterators.repeated(c, length(orbit))); append!(vals, col)
    end
    return SparseArrays.sparse(rows, cols, vals, d, c)
end

# The images of `states[inds]`, with column `j` in `entries` for `states[inds[j]]`.
# `::F` forces specialization, which Julia skips for a `Function` argument that is only passed on
function _permutation_images!(entries, inds, states, state_index::F, mapper, gs, weights, phases) where {F}
    foreach(empty!, entries)
    for (g, w, phase) in zip(gs, weights, phases)
        _add_permutation_operator!(entries, ((states[x] for x in inds), state_index), mapper, g, w, (j, i) -> phase(inds[j], i))
    end
    return entries
end

# A positive multiple of a character has weights of equal magnitude exactly when the
# representation is a multiple of a one-dimensional one, since |χ(g)| equals the dimension
# only when g acts as a scalar.
const _NumericWeight = Union{Integer, Rational, AbstractFloat}
_has_equal_magnitudes(weights) = all(w -> w isa Union{_NumericWeight, Complex{<:_NumericWeight}}, weights) &&
    all(w -> isapprox(abs(w), abs(first(weights)); rtol = sqrt(eps(float(real(typeof(w)))))), weights)

function _check_1d_character(perms, weights)
    ps = map(collect, perms)
    ws = collect(weights)
    index = Dict(p => g for (g, p) in enumerate(ps))
    c = abs(first(ws))
    tol = sqrt(eps(float(typeof(c)))) * c^2
    for (g, p) in enumerate(ps), (h, q) in enumerate(ps)
        k = get(index, p[q], 0)
        iszero(k) && throw(ArgumentError("`perms` is not closed under composition, so it is not a group. $_CHECK_HINT"))
        abs(ws[g] * ws[h] - c * ws[k]) <= tol ||
            throw(ArgumentError("`weights` are not a positive multiple of a one-dimensional character of the group formed by `perms`. $_CHECK_HINT"))
    end
    return nothing
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

    # The swap maps |11⟩ to -|11⟩, which is antisymmetric. Without phase factors it is symmetric.
    @test size(Psign, 2) == 2
    @test size(Ptriv, 2) == 2
    @test size(symmetric_sector(H, [H1, H2], :symmetric; phase_factors = false), 2) == 3

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
    @test_throws ArgumentError permutation_operator(H123, Hs3, [1, 1, 2])

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
    @test size(Qc, 2) == 2
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

@testitem "Permutation of more than 32 factors" begin
    using FermionicHilbertSpaces: permutation_operator
    @bosons b
    n = 36
    H = hilbert_space(b, 1:n, 2, NumberConservation(1))
    perm = circshift(1:n, 1)
    R = permutation_operator(H, factors(H), perm)
    @test all(R * representation(b[perm[k]]' * b[perm[k]], H) * R' == representation(b[k]' * b[k], H) for k in 1:n)
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
    for (k, rank) in zip(0:3, (4, 4, 4, 4))
        check(H, factors(H), ([circshift(1:4, j) for j in 0:3], [cis(2pi * k * j / 4) for j in 0:3]), rank)
    end
    # The two-dimensional irrep of S3, whose character is the number of fixed points minus one:
    # twice in the orbit of (1, 0, -1), and absent from the single state with M = 3
    @spins S 1
    perms = collect(permutations(1:3))
    chi = [count(i -> p[i] == i, 1:3) - 1.0 for p in perms]
    for (M, rank) in ((0, 4), (3, 0))
        Hm = hilbert_space(S, 1:3, AdditiveConstraint(M; maps = s -> s.m))
        check(Hm, factors(Hm), (perms, chi), rank)
    end
end

@testitem "Symmetric and antisymmetric sectors from orbits" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, symmetric_sector, AdditiveConstraint, _QR, _EIG
    using Combinatorics: permutations
    using SparseArrays: SparseMatrixCSC
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
        # The orbit construction relies on every permutation acting as a signed permutation matrix
        for perm in permutations(1:length(Hs))
            R = permutation_operator(H, Hs, perm)
            @test all(x -> x in (-1, 0, 1), R)
            @test all(==(1), sum(abs, R; dims = 1)) && all(==(1), sum(abs, R; dims = 2))
        end
        for sector in (:symmetric, :antisymmetric)
            P = symmetric_sector(H, Hs, sector; orth_method=nothing)
            for orth_method in (_QR(), _EIG())
                Q = symmetric_sector(H, Hs, sector; orth_method)
                @test Q isa SparseMatrixCSC{Float64}
                @test size(Q, 2) == round(Int, tr(P))
                @test Q' * Q ≈ I
                @test Q * Q' ≈ P
            end
        end
    end
    # The orbit basis is computed in the requested precision
    Qbig = symmetric_sector(cases[1]..., :symmetric, BigFloat)
    @test eltype(Qbig) == BigFloat && size(Qbig, 2) == 2 && maximum(abs, Qbig' * Qbig - I) < 1.0e-60
    # Single factor: the sector is the whole space
    H1 = hilbert_space(f, 1:1)
    @test symmetric_sector(H1, [H1], :antisymmetric) ≈ I
    @test_throws ArgumentError symmetric_sector(cases[1]..., :invalid)
end

@testitem "Sectors of one-dimensional characters from orbits" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_projector, AdditiveConstraint
    # Momentum sectors of a ring of spins 1/2 with total Sz = 0. Both vanish on the orbit of the
    # Néel states, whose stabilizer contains the translation by two sites.
    @spins S 1 // 2
    N = 6
    H = hilbert_space(S, 1:N, AdditiveConstraint(0; maps = s -> s.m))
    translations = [circshift(1:N, j) for j in 0:(N - 1)]
    for k in (1, 2)
        ws = [cis(2pi * k * j / N) for j in 0:(N - 1)]
        Q = symmetric_sector(H, factors(H), (translations, ws))
        @test Q' * Q ≈ I
        @test Q * Q' ≈ permutation_projector(H, factors(H), translations, ws)
    end
    @test_throws ArgumentError symmetric_sector(H, factors(H), (translations, [cis(pi * j / N) for j in 0:(N - 1)]))
    @test_throws ArgumentError symmetric_sector(H, factors(H), (translations[1:3], ones(3)))
end

@testitem "Fermionic permutations" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, symmetric_sector
    @fermions c
    sites = [hilbert_space(c, [(k, :u), (k, :d)]) for k in 1:3]
    H = tensor_product(sites)
    perm = [2, 3, 1]
    R = permutation_operator(H, sites, perm)
    @test all(R * representation(c[perm[k], s]', H, Matrix) * R' ≈ representation(c[k, s]', H, Matrix) for k in 1:3, s in (:u, :d))
    # The site-exchange symmetric and antisymmetric sectors block-diagonalize a Hubbard dimer
    H2 = tensor_product(sites[1:2])
    ham = sum(c[1, s]' * c[2, s] + hc for s in (:u, :d)) + 3 * sum(c[k, :u]' * c[k, :u] * c[k, :d]' * c[k, :d] for k in 1:2)
    M = representation(ham, H2, Matrix)
    Qs, Qa = symmetric_sector(H2, sites[1:2], :symmetric), symmetric_sector(H2, sites[1:2], :antisymmetric)
    @test norm(Qs' * M * Qa) < 1.0e-12
end

@testitem "Fermionic translation sectors" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_projector
    # The fermionic translations commute with the boundary hopping of a ring
    @fermions c
    N = 10
    H = hilbert_space(c, 1:N)
    ham = sum(c[k]' * c[mod1(k + 1, N)] + hc + c[k]' * c[k] * c[mod1(k + 1, N)]' * c[mod1(k + 1, N)] for k in 1:N)
    M = representation(ham, H, Matrix)
    translations = [circshift(1:N, j) for j in 0:(N - 1)]
    sectors = [symmetric_sector(H, factors(H), (translations, [cis(2pi * k * j / N) for j in 0:(N - 1)])) for k in 0:(N - 1)]
    @test sort(reduce(vcat, [eigvals(Hermitian(Matrix(Q' * M * Q))) for Q in sectors])) ≈ eigvals(Hermitian(M))
    # A two-dimensional irrep of the dihedral group, which is built block by block
    perms = [translations; reverse.(translations)]
    chi = [[2cos(2pi * j / N) for j in 0:(N - 1)]; zeros(N)]
    Q = symmetric_sector(H, factors(H), (perms, chi))
    @test Q * Q' ≈ permutation_projector(H, factors(H), perms, chi)
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
    Qs, Q = (symmetric_sector(H, Hs, (cyclic, ws); orth_method = :eig, check = false) for ws in (cis(pi / 4) .* chi, chi))
    @test Qs * Qs' ≈ Q * Q'
    @test_throws ArgumentError symmetric_sector(H, Hs, (cyclic[2:3], chi[2:3]))
    # The Hermitian check runs only when a basis is computed
    @test_throws ArgumentError symmetric_sector(H, Hs, (cyclic, [1.0, 1.0, 0.0]))
    @test symmetric_sector(H, Hs, (cyclic, [1.0, 1.0, 0.0]); orth_method=nothing) isa AbstractMatrix
end
