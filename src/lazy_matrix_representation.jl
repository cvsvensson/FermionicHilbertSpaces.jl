"""
    LazyOperator{O,S,T,C,P}

A matrix-free (lazy) representation of a symbolic operator on a Hilbert space.
Acts directly on vectors and matrices without constructing a sparse matrix.

`representation(op, space, :lazy)` returns a `SciMLOperators.FunctionOperator`
that wraps a `LazyOperator`. The bare `LazyOperator` is available as
`LazyOperator(op, space, chunking)`.
"""
mutable struct LazyOperator{O,S,T,C,P}
    const op::O
    const space::S
    const chunking::C
    const precomp::P
    const projection::Bool
    const conjugate::Bool
    const ishermitian::Bool
    cache::Any
end

function _show_lazy_operator_expression(io::IO, L::LazyOperator)
    show(IOContext(io, :compact => true), L.op)
    L.conjugate && print(io, "^*")
end

function Base.show(io::IO, L::LazyOperator)
    _show_lazy_operator_expression(io, L)
    print(io, " acting on ")
    show(IOContext(io, :compact => true), L.space)
end

function LazyOperator(_op, space::S, chunking::C=NoChunking(); projection=false, conjugate=false, ishermitian=_ishermitian(_op), T=mat_eltype(_op), concretizer=NoConcretizer()
) where {S,C<:AbstractChunkingStrategy}
    op = __concretize(_op, concretizer)
    precomp = lazy_precomputation(op, space)
    O = typeof(op)
    P = typeof(precomp)
    LazyOperator{O,S,T,C,P}(op, space, chunking, precomp, projection, conjugate, ishermitian, nothing)
end
function LazyOperator(op::O, space::S, chunking::C, precomp::P; projection=false, conjugate=false, ishermitian=_ishermitian(op), T=mat_eltype(op)) where {O,S,C<:AbstractChunkingStrategy,P}
    LazyOperator{O,S,T,C,P}(op, space, chunking, precomp, projection, conjugate, ishermitian, nothing)
end

# Data that `lazy_mul!` needs to apply `op` on `space`, computed once when the `LazyOperator`
# is constructed. For an `NCAdd` it is a vector with one `(term, coeff, precomp)` tuple per
# entry `term => coeff` of `op.dict`.
lazy_precomputation(op, space) = precomputation_before_operator_application(op, space)
function lazy_precomputation(op::NCAdd, space)
    map(collect(op.dict)) do (term, coeff)
        (term, coeff, precomputation_before_operator_application(term, space))
    end
end

function _lazy_output_prototype(L::LazyOperator, input::AbstractVector)
    size(L, 2) == length(input) || throw(DimensionMismatch("The input must have length $(size(L, 2)), got $(length(input))"))
    T = promote_type(eltype(L), eltype(input))
    return similar(input, T, size(L, 1))
end

function _lazy_output_prototype(L::LazyOperator, input::AbstractMatrix)
    size(input, 1) == size(L, 2) || throw(DimensionMismatch("The input must have $(size(L, 2)) rows, got size $(size(input))"))
    T = promote_type(eltype(L), eltype(input))
    return similar(input, T, (size(L, 1), size(input, 2)))
end


# The default cache, `similar(input, T, dims)`, fails for lengths that do not fit in an `Int`.
function scimloperator(L::LazyOperator, input=Vector{eltype(L)}(undef, dim(L.space)), output=input; kwargs...)
    SciMLOperators.FunctionOperator(L, input, output; ishermitian=ishermitian(L), op_adjoint=adjoint(L), isconstant=true, T=eltype(L), islinear=true, batch=true, cache=(similar(input), similar(output)), kwargs...)
end

(L::LazyOperator)(w, v, u, p, t, α, β) = mul!(w, L, v, α, β)
(L::LazyOperator)(w, v, u, p, t) = mul!(w, L, v)
(L::LazyOperator)(v, u, p, t) = L * v

_ishermitian(x::NCMul) = iszero(x - hc)
_ishermitian(x::NCAdd) = iszero(x - hc)
function _ishermitian(x::ProductOperator)
    _ishermitian(prod(x.ops))
end

Base.size(L::LazyOperator) = (d = dim(L.space); (d, d))
Base.size(L::LazyOperator, i::Int) = size(L)[i]
Base.eltype(::LazyOperator{O,S,T,C,P}) where {O,S,T,C,P} = T
Base.conj(L::LazyOperator) = LazyOperator(L.op, L.space, L.chunking, L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
function Base.adjoint(L::LazyOperator)
    LazyOperator(L.op, TransposedSpace(L.space), L.chunking, L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
end
function Base.transpose(L::LazyOperator)
    LazyOperator(L.op, TransposedSpace(L.space), L.chunking, L.precomp; projection=L.projection, conjugate=L.conjugate, ishermitian=L.ishermitian)
end
function Base.adjoint(L::LazyOperator{<:ProductOperator{C}}) where C
    newop = ProductOperator{C}(L.op.ops, map(TransposedSpace, L.op.spaces), L.op.inds)
    LazyOperator(newop, TransposedSpace(L.space), L.chunking, L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
end
function Base.transpose(L::LazyOperator{<:ProductOperator{C}}) where C
    newop = ProductOperator{C}(L.op.ops, map(TransposedSpace, L.op.spaces), L.op.inds)
    LazyOperator(newop, TransposedSpace(L.space), L.chunking, L.precomp; projection=L.projection, conjugate=L.conjugate, ishermitian=L.ishermitian)
end
function Base.transpose(L::SciMLOperators.FunctionOperator{<:Any,<:Any,<:Any,<:Any,<:LazyOperator})
    isnothing(L.cache) && return scimloperator(transpose(L.op))
    return scimloperator(transpose(L.op), first(L.cache))
end
LinearAlgebra.ishermitian(L::LazyOperator) = L.ishermitian

function _eager_matrix_representation(L::LazyOperator{<:Union{NCMul,ProductOperator}})
    return _term_matrix_representation(L.op, L.space, EagerSparseRepr(), L.chunking; projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:NCAdd})
    return _matrix_representation_single_space(L.op, L.space, EagerSparseRepr(), L.chunking; projection=L.projection)
end

function Base.convert(::Type{AbstractMatrix}, L::LazyOperator)
    sparse(L)
end

function SparseArrays.sparse(L::LazyOperator)
    M = _eager_matrix_representation(L)
    L.conjugate && (M = conj(M))
    return M
end

function LinearAlgebra.mul!(y::AbstractVector, L::LazyOperator, x::AbstractVector, α=true, β=false)
    return lazy_mul!(y, L, x, α, β)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, L::LazyOperator, X::AbstractMatrix, α=true, β=false)
    size(X, 1) == size(L, 2) || throw(DimensionMismatch("The input must have $(size(L, 2)) rows, got size $(size(X))"))
    expected_size = (size(L, 1), size(X, 2))
    size(Y) == expected_size || throw(DimensionMismatch("The output must have size $expected_size, got $(size(Y))"))
    lazy_mul!(Y, L, X, α, β)
    return Y
end

function _apply_single_term!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, projection, ::Union{NoChunking,TermChunking})
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for (n, state) in enumerate(basisstates(space))
        xn = x[n] * coeff
        iszero(xn) && continue
        state, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(state, space)
            if !_skip_outside(outind, projection)
                amp = (conjugate ? conj(_amp) : _amp)
                y[outind] += amp * xn
            end
        end
    end
end

function _apply_single_term!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, coeff, conjugate, projection, ::Union{NoChunking,TermChunking})
    _foreach_term_entry((i, v) -> (y[i] += v), x, space, term, precomp, coeff, conjugate, projection)
end

# Calls `f(outind, value)` for each output entry of `coeff * term` applied to the sparse `x`.
function _foreach_term_entry(f, x::SparseArrays.AbstractSparseVector, space, term, precomp, _coeff, conjugate, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for (n, xn) in zip(SparseArrays.nonzeroinds(x), nonzeros(x))
        iszero(xn) && continue
        state = basisstate(n, space)
        newstate, _amp = _apply_local_operators(term, state, space, precomp)
        iszero(_amp) && continue
        outind = state_index(newstate, space)
        _skip_outside(outind, projection) && continue
        f(outind, (conjugate ? conj(_amp) : _amp) * coeff * xn)
    end
end

function _apply_single_term!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, projection, ::Union{NoChunking,TermChunking})
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for (n, state) in enumerate(basisstates(space))
        newstate, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(newstate, space)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !_skip_outside(outind, projection)
                @views y[outind, :] .+= amp .* x[n, :]
            end
        end
    end
end

function _apply_single_term!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, _coeff, conjugate, projection, ::Union{NoChunking,TermChunking})
    coeff = (conjugate ? conj(_coeff) : _coeff)
    rows = rowvals(x)
    vals = nonzeros(x)
    for col in axes(x, 2)
        for ptr in nzrange(x, col)
            n = rows[ptr]
            xn = vals[ptr]
            state = basisstate(n, space)
            newstate, _amp = _apply_local_operators(term, state, space, precomp)
            if !iszero(_amp)
                outind = state_index(newstate, space)
                amp = (conjugate ? conj(_amp) : _amp) * coeff
                if !_skip_outside(outind, projection)
                    y[outind, col] += amp * xn
                end
            end
        end
    end
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCMul}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection, L.chunking)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:ProductOperator}, x::AbstractVecOrMat, α, β) where T
    # This is for productspaces, where ops is a list of operators applying to each factor space
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection, L.chunking)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCAdd}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    op = L.op
    for (term, coeff, precomp) in L.precomp
        _apply_single_term!(y, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection, L.chunking)
    end
    if !iszero(op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(op.coeff) : op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

function Base.:*(L::LazyOperator, x::AbstractVecOrMat)
    T = promote_type(eltype(L), eltype(x))
    y = similar(x, T)
    return mul!(y, L, x)
end

# `sparsevec` sums the collected values with equal index.
function Base.:*(L::LazyOperator, x::SparseArrays.AbstractSparseVector)
    size(L, 2) == length(x) || throw(DimensionMismatch("The input must have length $(size(L, 2)), got $(length(x))"))
    T = promote_type(eltype(L), eltype(x))
    outinds = typeof(size(L, 1))[]
    amps = T[]
    push_entry = (i, v) -> (push!(outinds, i); push!(amps, v))
    for (term, coeff, precomp) in _lazy_terms(L)
        _foreach_term_entry(push_entry, x, L.space, term, precomp, coeff, L.conjugate, L.projection)
    end
    if L.op isa NCAdd && !iszero(L.op.coeff)
        scalar_coeff = L.conjugate ? conj(L.op.coeff) : L.op.coeff
        append!(outinds, SparseArrays.nonzeroinds(x))
        append!(amps, scalar_coeff .* nonzeros(x))
    end
    return SparseArrays.sparsevec(outinds, amps, size(L, 1))
end
_lazy_terms(L::LazyOperator{<:NCAdd}) = L.precomp
_lazy_terms(L::LazyOperator) = ((L.op, true, L.precomp),)
_sparse_input(H) = (d = dim(H); spzeros(mat_eltype(H), typeof(d), d))
get_input(::LazyRepr{Missing}, H) = dim(H) < 1e4 ? Vector{mat_eltype(H)}(undef, dim(H)) : _sparse_input(H)
function get_input(rep::LazyRepr{<:AbstractArray}, H)
    dim(H) == size(rep.input, 1) || throw(DimensionMismatch("The input must have $(dim(H)) rows, got size $(size(rep.input))"))
    return rep.input
end
function get_input(rep::LazyRepr{Symbol}, H)
    rep.input == :dense && return Vector{mat_eltype(H)}(undef, dim(H))
    rep.input == :sparse && return _sparse_input(H)
end

function _term_matrix_representation(op::Union{NCMul,ProductOperator}, H::AbstractHilbertSpace, rep::LazyRepr, chunking; kwargs...)
    scimloperator(LazyOperator(op, H, chunking; kwargs...), get_input(rep, H))
end

function _matrix_representation_single_space(op::NCAdd, H, rep::LazyRepr, chunking; kwargs...)
    scimloperator(LazyOperator(op, H, chunking; kwargs...), get_input(rep, H))
end

function __matrix_representation(op::NCAdd, bases, space, repr::LazyRepr, chunking; kwargs...)
    chunking isa TermChunking && @warn "TermChunking is not supported for LazyRepr. Ignoring chunking strategy."
    mats = Any[_matrix_representation(term, bases, space, repr, chunking; kwargs...) for term in NCterms(op)]
    if !iszero(op.coeff)
        push!(mats, op.coeff * _matrix_representation(missing, bases, space, repr, NoChunking(); kwargs...))
    end
    return _lazy_sum(mats)
end

# On an unconstrained product space, the terms that act on a single group are summed into one
# lazy operator on that group, which is embedded with identities on the other groups. The
# scalar coefficient goes into the first of these. Terms acting on several groups are
# represented one by one, as in the generic method.
function __matrix_representation(op::NCAdd, bases, space::ProductSpace, repr::LazyRepr, chunking; kwargs...)
    spaces = groups(space)
    local_terms = [eltype(op.dict)[] for _ in spaces]
    mats = []
    for (term, coeff) in op.dict
        n = _single_group_index(term, bases)
        if isnothing(n)
            push!(mats, _matrix_representation(NCMul(coeff, term.factors), bases, space, repr, chunking; kwargs...))
        else
            push!(local_terms[n], term => coeff)
        end
    end
    scalar = op.coeff
    local_mats = []
    for (n, terms) in enumerate(local_terms)
        isempty(terms) && continue
        # factor vectors with a common supertype of the factors of the group
        S = mapreduce(((term, _),) -> mapreduce(typeof, typejoin, term.factors), typejoin, terms)
        local_op = NCAdd(scalar, Dict(NCMul(term.coeff, convert(Vector{S}, term.factors)) => coeff for (term, coeff) in terms))
        scalar = zero(scalar)
        local_mat = _matrix_representation_single_space(local_op, spaces[n], repr, chunking; T=mat_eltype(op), kwargs...)
        factor_mats = [m == n ? local_mat : SciMLOperators.IdentityOperator(dim(spaces[m])) for m in eachindex(spaces)]
        push!(local_mats, foldl(kron, Iterators.reverse(factor_mats)))
    end
    if !iszero(scalar)
        push!(mats, scalar * _matrix_representation(missing, bases, space, repr, NoChunking(); kwargs...))
    end
    return _lazy_sum(vcat(local_mats, mats))
end

# The index of the group in `bases` that all factors of `term` belong to, or `nothing` if
# the factors belong to several groups or there are no factors.
function _single_group_index(term, bases)
    isempty(term.factors) && return nothing
    group = symbolic_group(first(term.factors))
    all(fac -> symbolic_group(fac) == group, term.factors) || return nothing
    return findfirst(==(group), bases)
end

_as_scimloperator(m::SciMLOperators.AbstractSciMLOperator) = m
_as_scimloperator(m::AbstractMatrix) = SciMLOperators.MatrixOperator(m)
function _lazy_sum(mats)
    length(mats) == 1 && return only(mats)
    return SciMLOperators.AddedOperator(Tuple(map(_as_scimloperator, mats)))
end

@testitem "lazy matrix_representation" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    @fermions f
    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im
    L = representation(op, Hf, :lazy)
    M = representation(op, Hf)
    v = randn(dim(Hf))
    w = randn(ComplexF64, dim(Hf))
    Vm = randn(dim(Hf), 3)
    Wm = randn(ComplexF64, dim(Hf), 3)
    α = 0.7
    β = -0.2
    @test SciMLOperators.isconstant(L)
    @test SciMLOperators.islinear(L)
    @test !SciMLOperators.isconvertible(L)
    @test SciMLOperators.has_mul(L)
    @test SciMLOperators.has_mul!(L)
    @test SciMLOperators.has_adjoint(L)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) isa SciMLOperators.FunctionOperator
    @test adjoint(L) isa SciMLOperators.FunctionOperator
    @test transpose(L) * v ≈ transpose(M) * v
    @test L * Vm ≈ M * Vm
    t = 0
    u = p = nothing
    @test L(v, u, p, t) ≈ M * v
    @test L(Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t) ≈ M * v
    @test L(copy(Wm), Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t, α, β) ≈ α .* (M * v) .+ β .* w
    # a zero β overwrites the output, including non-finite entries
    @test L(fill(NaN + 0im, dim(Hf)), v, u, p, t, α, 0.0) ≈ α .* (M * v)
    @test L(fill(NaN + 0im, dim(Hf), 3), Vm, u, p, t, α, 0.0) ≈ α .* (M * Vm)
    @test L(copy(Wm), Vm, u, p, t, α, β) ≈ α .* (M * Vm) .+ β .* Wm
    vs = sprandn(dim(Hf), 0.5)
    ws = sprandn(ComplexF64, dim(Hf), 0.5)
    Xs = sprandn(dim(Hf), 3, 0.5)
    Ys = sprandn(ComplexF64, dim(Hf), 3, 0.5)
    @test L * vs ≈ M * vs
    @test L' * vs ≈ M' * vs
    @test transpose(L) * vs ≈ transpose(M) * vs
    @test L * Xs ≈ M * Xs
    @test L' * Xs ≈ M' * Xs
    @test transpose(L) * Xs ≈ transpose(M) * Xs
    @test L(copy(ws), vs, u, p, t, α, β) ≈ α .* (M * vs) .+ β .* ws
    @test L(copy(Ys), Xs, u, p, t, α, β) ≈ α .* (M * Xs) .+ β .* Ys
    @test (L * vs) isa SparseVector
    @test (L * Xs) isa SparseMatrixCSC
    @test (L + one(L)) * v ≈ (M + I) * v
    @test (2 * L) * v ≈ 2 * (M * v)
    @test concretize(L) == M
    @test !ishermitian(L)
    @test ishermitian(representation(op + hc, Hf, :lazy))

    import FermionicHilbertSpaces: LazyRepr
    @test representation(op, Hf, :lazy).cache[1] isa Vector
    @test representation(op, Hf, LazyRepr(:dense)).cache[1] isa Vector
    @test representation(op, Hf, LazyRepr(:sparse)).cache[1] isa SparseVector
    dense_cache = zeros(ComplexF64, dim(Hf))
    sparse_cache = spzeros(ComplexF64, dim(Hf))
    symbol_cache = [:a for _ in 1:dim(Hf)]
    @test representation(op, Hf, LazyRepr(dense_cache)).cache[1] isa typeof(dense_cache)
    @test representation(op, Hf, LazyRepr(sparse_cache)).cache[1] isa typeof(sparse_cache)
    @test representation(op, Hf, LazyRepr(symbol_cache)).cache[1] isa typeof(symbol_cache)

    @test_throws DimensionMismatch representation(op, Hf, LazyRepr(zeros(dim(Hf) + 1)))

    @spin s 1 // 2
    Hs = hilbert_space(s)
    H = tensor_product(Hf, Hs)
    op = 5 * f[1]' * f[2] * s[:x] + 2 * s[:z] + f[1] + 1im
    L = representation(op, H, :lazy)
    M = representation(op, H)
    v = randn(dim(H))
    Vm = randn(dim(H), 2)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v
    @test L * Vm ≈ M * Vm

    Hcons = constrain_space(H, NumberConservation(2, [Hf]))
    L = representation(op, Hcons, :lazy; projection=true)
    M = representation(op, Hcons; projection=true)
    v = randn(dim(Hcons))
    Vm = randn(ComplexF64, dim(Hcons), 2)
    @test L * v ≈ M * v
    @test L * Vm ≈ M * Vm
    vcons_s = sprandn(ComplexF64, dim(Hcons), 0.5)
    Vcons_s = sprandn(ComplexF64, dim(Hcons), 2, 0.5)
    @test L * vcons_s ≈ M * vcons_s
    @test L * Vcons_s ≈ M * Vcons_s
    @test L' * Vcons_s ≈ M' * Vcons_s
    @test transpose(L) * Vcons_s ≈ transpose(M) * Vcons_s
    @test (L * vcons_s) isa SparseVector
    @test (L * Vcons_s) isa SparseMatrixCSC
    @test L' * vcons_s ≈ M' * vcons_s
    @test transpose(L) * vcons_s ≈ transpose(M) * vcons_s
    Lnoproj = representation(op, Hcons, :lazy)
    @test_throws ArgumentError Lnoproj * sparse(ones(dim(Hcons)))
    @test_throws ArgumentError Lnoproj * ones(dim(Hcons))
    @test_throws ArgumentError Lnoproj * ones(dim(Hcons), 2)
    @test Matrix(L) ≈ M
    @test Matrix(L') ≈ M'
    @test Matrix(transpose(L)) ≈ transpose(M)
    @test Matrix(transpose(L)') ≈ conj(M)

    # @test !ishermitian(L)
    # @test ishermitian(representation(op + hc, Hcons; lazy=true, projection=true)) #SciMLOperators doesn't check termwise ishermitian for an added operator
end

@testitem "lazy matrix_representation with dimensions wider than Int" begin
    using SparseArrays
    import FermionicHilbertSpaces: focknbr_from_site_indices, state_index
    @fermions f
    op = 0.3 * f[1]' * f[2] + 0.7im * f[2]' * f[3] + hc + 0.5 * f[1]' * f[1] + 0.2
    Hsmall = hilbert_space(f, 1:3)
    M = representation(op, Hsmall)
    for N in (63, 200, 1100)
        H = hilbert_space(f, 1:N)
        @test dim(H) == big(2)^N
        # the states of `Hsmall` with mode N occupied
        high = focknbr_from_site_indices((N,)).f
        inds = [state_index(FockNumber(high | (k - 1)), H) for k in 1:dim(Hsmall)]
        v = randn(ComplexF64, dim(Hsmall))
        x = sparsevec(inds, v, dim(H))
        L = representation(op, H, :lazy)
        for (Lop, Mop) in ((L, M), (L', M'), (transpose(L), transpose(M)))
            y = Lop * x
            @test y isa SparseVector && length(y) == dim(H) && nnz(y) == length(inds)
            @test [y[i] for i in inds] ≈ Mop * v
        end
    end
end

@testitem "lazy matrix_representation on product spaces" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: open_system, dissipator
    function lazy_matches(op, H)
        L = representation(op, H, :lazy)
        M = representation(op, H)
        v = randn(ComplexF64, dim(H))
        V = randn(ComplexF64, dim(H), 2)
        w = randn(ComplexF64, dim(H))
        Lc = cache_operator(L, v)
        y = copy(w)
        mul!(y, Lc, v, 0.3, -0.7)
        return L * v ≈ M * v && L' * v ≈ M' * v && transpose(L) * v ≈ transpose(M) * v &&
               L * V ≈ M * V && y ≈ 0.3 * (M * v) - 0.7 * w && concretize(L) ≈ M
    end

    @fermions c
    N = 3
    Hopen, _, _, left, right = open_system(c, 1:N)
    ham = sum((0.3n) * c[n]' * c[n] for n in 1:N) + sum(0.2im * c[n]' * c[n + 1] + hc for n in 1:(N - 1))
    liouvillian = -1im * (left(ham) - right(ham)) + sum(dissipator(sqrt(0.1n) * c[n]) for n in 1:N)
    @test lazy_matches(liouvillian + 0.5, Hopen)
    @test lazy_matches(left(ham) + 2, Hopen)
    @test lazy_matches(left(c[1]) * right(c[1]') + left(c[2]') * right(c[3]), Hopen)

    @fermions f
    @boson b
    @spin s 1 // 2
    Hf = hilbert_space(f, 1:2)
    H = tensor_product(Hf, hilbert_space(b, 3), hilbert_space(s))
    op = f[1]' * f[2] + hc + 0.3 * b' * b + 0.2 * (b' + b) * f[1]' * f[1] + 0.7 * s[:z] +
         0.1 * s[:x] * (b + b') + 0.05 * f[2]' * f[2] * s[:z] * b' * b + f[1] * s[:x] + 0.4
    @test lazy_matches(op, H)

    # terms of one group with factors of different concrete types, and a complex coefficient
    H2 = tensor_product(hilbert_space(f, [1, :a]), hilbert_space(b, 3))
    @test lazy_matches(1im * f[1]' * f[:a] + hc + 0.5im * f[1]' * f[1] + b' * b, H2)
end

@testitem "lazy matrix_representation chunking" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    using OhMyThreads
    import FermionicHilbertSpaces: TermChunking, StateChunking, NoChunking, cache_operator!

    scheduler = StaticScheduler(; nchunks=4)
    scheduler = SerialScheduler()

    @fermions f
    Hf = hilbert_space(f, 1:4)
    op = f[1]' * f[2] + 2im * f[3]' * f[4] + hc + 1.5

    M = representation(op, Hf)
    Ln = representation(op, Hf, :lazy; chunking=NoChunking())
    Lt = representation(op, Hf, :lazy; chunking=TermChunking(scheduler))
    Ls = representation(op, Hf, :lazy; chunking=StateChunking(scheduler))

    v = randn(ComplexF64, dim(Hf))
    V = randn(ComplexF64, dim(Hf), 3)
    vs = sprandn(ComplexF64, dim(Hf), 0.5)
    Vs = sprandn(ComplexF64, dim(Hf), 3, 0.5)

    cache_operator(Lt, v)
    cache_operator(Ls, V)
    @test Lt.op.cache isa Channel
    @test Ls.op.cache isa Channel
    @test size(fetch(Ls.op.cache)) == size(V)

    @test Lt * v ≈ M * v
    @test Ls * v ≈ M * v
    @test mul!(fill(NaN + 0im, dim(Hf)), Lt, v, 0.5, 0.0) ≈ 0.5 * (M * v)
    @test mul!(fill(NaN + 0im, dim(Hf)), Ls, v, 0.5, 0.0) ≈ 0.5 * (M * v)
    @test Lt * V ≈ M * V
    @test Ls * V ≈ M * V
    @test Lt * vs ≈ M * vs
    @test Ls * vs ≈ M * vs
    @test Lt * Vs ≈ M * Vs
    @test Ls * Vs ≈ M * Vs
    @test concretize(Lt) ≈ M
    @test concretize(Ls) ≈ M

    @spin s 1 // 2
    Hs = hilbert_space(s)
    Hprod = tensor_product(Hf, Hs)
    op_prod = f[1]' * f[2] * s[:x] + f[3]' * f[4] * s[:z] + hc + 1im
    Mprod = representation(op_prod, Hprod)
    Ltprod = representation(op_prod, Hprod, :lazy; chunking=TermChunking(scheduler))
    Lsprod = representation(op_prod, Hprod, :lazy; chunking=StateChunking(scheduler))
    vprod = randn(ComplexF64, dim(Hprod))
    @test Ltprod * vprod ≈ Mprod * vprod
    @test Lsprod * vprod ≈ Mprod * vprod

    Hcons = constrain_space(Hprod, NumberConservation(2, [Hf]))
    vcons = randn(ComplexF64, dim(Hcons))
    Mcons = representation(op_prod, Hcons; projection=true)
    Mv = Mcons * vcons
    for concretizer in [FermionicHilbertSpaces.NoConcretizer(), FermionicHilbertSpaces.TupleConcretizer(), FermionicHilbertSpaces.VecConcretizer()]
        Lcons = representation(op_prod, Hcons, :lazy; projection=true, chunking=NoChunking(), concretizer)
        Ltcons = representation(op_prod, Hcons, :lazy; projection=true, chunking=TermChunking(scheduler), concretizer)
        Lscons = representation(op_prod, Hcons, :lazy; projection=true, chunking=StateChunking(scheduler), concretizer)
        @test Lcons * vcons ≈ Mv
        @test Ltcons * vcons ≈ Mv
        @test Lscons * vcons ≈ Mv
    end
end

