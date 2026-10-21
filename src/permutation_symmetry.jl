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
- `normalize`: rescale `P` by `tr(P) / sum(abs2, P)`, which makes `P^2 == P` when `P` is
  proportional to a projector.
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
    P = sparse ? _to_sparse(finalize!(accum, H)) : accum
    _is_negligible(P, ws) && (P = zero(P))
    if normalize
        tr2 = sum(abs2, P)
        tr1 = tr(P)
        iszero(tr2) || (P = _rmul!!(P, tr1 / tr2))
    end
    return P
end
_to_sparse(P::SparseMatrixCSC) = P
_to_sparse(P) = SparseArrays.sparse(P)
_rmul!!(P, c) = promote_type(eltype(P), typeof(c)) == eltype(P) ? rmul!(P, c) : P * c

# Floating point sums of weights that cancel exactly leave roundoff of order
# eps * sqrt(length(ws)) * maximum(abs, ws). For weights c * conj(χ(g)) of an irrep χ of
# dimension d, a nonzero P has a diagonal entry of at least |c| = maximum(abs, ws) / d.
# Comparing with sqrt(eps) * maximum(abs, ws) separates the two cases for any
# realistic group order and irrep dimension. Other element types are exact or symbolic.
function _is_negligible(P::AbstractArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}, ws)
    scale = maximum(abs, ws)
    return _max_abs(P) <= sqrt(eps(real(eltype(P)))) * scale
end
_max_abs(P::SparseMatrixCSC) = maximum(abs, nonzeros(P); init=zero(real(eltype(P))))
_max_abs(P) = maximum(abs, P; init=zero(real(eltype(P))))
_is_negligible(P, ws) = false

function _default_projector_eltype(weights)
    isnothing(weights) && return Int
    eltype(weights) <: Number && return eltype(weights)
    typeof(first(weights)) <: Number && return typeof(first(weights))
    return Float64
end

"""
    symmetric_sector(H, Hs, sector=:symmetric, T=default; orth_method=:qr, normalize=true, sparse=true)

Return a basis for the sector of `H` selected by `sector`, under permutations of the
factors `Hs`.

`sector` is either a tuple `(perms, weights)`, with permutations of `1:length(Hs)` and
one weight per permutation, or, if `Combinatorics.jl` is loaded, `:symmetric` or
`:antisymmetric`, which use all permutations with weight one or the permutation sign.
`T` is the element type of the projector. It defaults to `Float64` for `:symmetric` and
`:antisymmetric`, and for a tuple `(perms, weights)` to the floating point version of the
element type of `weights`, for example `ComplexF64` for complex weights.

The projector is built with `permutation_projector(H, Hs, perms, weights, T; normalize, sparse)`.
`orth_method` selects what is returned:
- `:qr`: a matrix with orthonormal columns spanning the sector, from a QR decomposition
  of the projector.
- `:eig`: a matrix with orthonormal columns spanning the sector, from the eigenvectors of
  the projector with eigenvalue one. The projector is converted to a dense matrix.
- `nothing`: the projector itself.

`Hs` may be either a full partition of `H` or a proper subsystem list; in the latter
case the sector is constructed on `tensor_product(Hs...)` and embedded back into `H`.
"""
function symmetric_sector(H::AbstractHilbertSpace, Hs, sector=:symmetric, ::Type{T}=_default_sector_eltype(sector);
    orth_method=:qr, kwargs...) where T
    isempty(Hs) && throw(ArgumentError("`Hs` must contain at least one factor space."))
    method = _orth_method(orth_method)

    perms, weights = _resolve_sector_permutations_and_weights(Hs, sector, T)
    length(perms) == length(weights) || throw(ArgumentError("The generated weights and permutations must have the same length, got $(length(weights)) and $(length(perms))."))
    P = permutation_projector(H, Hs, perms, weights, T; kwargs...)
    isnothing(method) && return P

    tr2 = sum(abs2, P)
    rank = iszero(tr2) ? 0 : round(Int, real(tr(P))^2 / tr2)
    _remove_columns!(P, method, rank)
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

function _remove_columns!(P::Matrix, ::_QR, M::Int=round(Int, tr(P)))
    F = try
        qr!(P, ColumnNorm())
    catch e
        qr(P, ColumnNorm())
    end
    big_cols = findall(>(0.1) ∘ abs, diag(F.R))
    return F.Q[:, big_cols]
end
_remove_columns!(P::AbstractMatrix, ::_EIG, M::Int) = _remove_columns!(Matrix(P), _EIG(), M) # ensure we have a dense matrix for the decomposition
function _remove_columns!(P::Matrix, ::_EIG, M::Int)
    E = try
        eigen!(Hermitian(P))
    catch e
        eigen(Hermitian(P))
    end
    keep = findall(>(0.9) ∘ abs, E.values)  # λ ≈ 1 subspace
    return E.vectors[:, keep]  # already orthonormal
end
function _remove_columns!(P::SparseMatrixCSC, ::_QR, M::Int)
    F = qr(P; tol=0.1)
    F.Q[invperm(F.prow), 1:M]
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
    cw = ([idperm, swapperm], [1.0 + 0im, cis(0.3)])
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
