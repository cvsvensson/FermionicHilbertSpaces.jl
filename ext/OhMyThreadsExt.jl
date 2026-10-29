module OhMyThreadsExt
using SparseArrays: SparseArrays, findnz, nonzeros, nzrange, rowvals
using OhMyThreads: OhMyThreads, Scheduler, chunks, tforeach, tmap
using FermionicHilbertSpaces: FermionicHilbertSpaces
using LinearAlgebra: LinearAlgebra, rmul!
using NonCommutativeProducts: NCAdd, NCMul, NCterms
import SciMLOperators: cache_operator

import OhMyThreads: Schedulers.chunking_args, Consecutive
import FermionicHilbertSpaces: EagerDenseRepr, EagerSparseRepr, TermChunking, StateChunking, ProductOperator, ProductSpace, _matrix_representation, __matrix_representation, _terms_matrix, _precomputed_terms, _blocked_sparse_matrix, _narrowed, _add_identity_term, matrix_accumulator, add_identity!!, NoChunking, chunked_operator_indices_and_amplitudes_generic!, LazyOperator, lazy_mul!, _apply_single_term!, _apply_local_operators, basisstate, state_index, _skip_outside, dim, _lazy_output_prototype, cache_operator!

function _reduce_add!(y, partials)
    isempty(partials) && return y
    # Build a lazy nested broadcast tree: p1 .+ p2 .+ p3 .+ ...
    bc = reduce((acc, p) -> Base.broadcasted(+, acc, p), partials)
    # Materialize in one fused pass: y[i] += p1[i] + p2[i] + ...
    broadcast!(+, y, y, bc)
    return y
end

const ThreadedChunking = Union{TermChunking,StateChunking}

# Consecutive ranges of basis-state indices, one per chunk of the scheduler, and the single range `1:0` when the space is empty
function _state_chunk_data(space, chunking::ThreadedChunking)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    return isempty(basis_chunks) ? [1:0] : basis_chunks
end

function _term_chunk_data(terms, chunking::TermChunking)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    term_chunks = collect(chunks(terms; n, split=Consecutive()))
    return term_chunks
end

function get_nchunks(scheduler)
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    return n
end

cache_operator(L::LazyOperator, input::AbstractVecOrMat) = cache_operator!(L, input)
function cache_operator!(L::LazyOperator, input::AbstractVecOrMat)
    _cache_matches(L, L.cache, input) && return L
    y = _lazy_output_prototype(L, input)
    nbuffers = Threads.nthreads()
    chnl = Channel{typeof(y)}(nbuffers)
    foreach(_ -> put!(chnl, similar(y)), 1:nbuffers)
    L.cache = chnl
    return L
end

function _cache_matches(L, cache, input::AbstractVecOrMat)
    cache isa Channel || return false
    isready(cache) || return false
    buf = fetch(cache)  # peek without taking
    promote_type(eltype(buf), eltype(input)) == eltype(buf) || return false
    return size(buf) == (size(L, 1), size(input)[2:end]...)
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:TermChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    op = L.op
    term_chunks = _term_chunk_data(L.precomp, L.chunking)
    cache_operator!(L, y)
    chnl = L.cache
    scheduler = L.chunking.scheduler
    lk = ReentrantLock()
    tforeach(term_chunks; scheduler) do local_terms
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for (term, coeff, precomp) in local_terms
            _apply_single_term!(local_mat, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection, NoChunking())
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
        local_mat
    end

    if !iszero(op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(op.coeff) : op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCMul,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    cache_operator!(L, y)
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection, L.chunking, L.cache)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:ProductOperator,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    cache_operator!(L, y)
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection, L.chunking, L.cache)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    cache_operator!(L, y)
    for (term, coeff, precomp) in L.precomp
        _apply_single_term!(y, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection, L.chunking, L.cache)
    end
    if !iszero(L.op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(L.op.coeff) : L.op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

function _apply_single_term!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking, chnl::Channel)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    lk = ReentrantLock()
    tforeach(basis_chunks; scheduler) do local_cols
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for n in local_cols
            xn = x[n] * coeff
            iszero(xn) && continue
            state = basisstate(n, space)
            state, _amp = _apply_local_operators(term, state, space, precomp)
            if !iszero(_amp)
                outind = state_index(state, space)
                if !_skip_outside(outind, projection)
                    amp = (conjugate ? conj(_amp) : _amp)
                    local_mat[outind] += amp * xn
                end
            end
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
        local_mat
    end
    return y
end

function _apply_single_term!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking, chnl::Channel)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    inds, vals = findnz(x)
    lk = ReentrantLock()
    tforeach(basis_chunks; scheduler) do local_cols
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        lo = first(local_cols)
        hi = last(local_cols)
        for (n, xn) in zip(inds, vals)
            (n < lo || n > hi) && continue
            state = basisstate(n, space)
            state, _amp = _apply_local_operators(term, state, space, precomp)
            if !iszero(_amp)
                outind = state_index(state, space)
                amp = (conjugate ? conj(_amp) : _amp) * coeff
                if !_skip_outside(outind, projection)
                    local_mat[outind] += amp * xn
                end
            end
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
        local_mat
    end
    return y
end

function _apply_single_term!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking, chnl::Channel)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    lk = ReentrantLock()
    tforeach(basis_chunks; scheduler) do local_cols
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for n in local_cols
            state = basisstate(n, space)
            newstate, _amp = _apply_local_operators(term, state, space, precomp)
            if !iszero(_amp)
                outind = state_index(newstate, space)
                amp = (conjugate ? conj(_amp) : _amp) * coeff
                if !_skip_outside(outind, projection)
                    @views local_mat[outind, :] .+= amp .* x[n, :]
                end
            end
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
        local_mat
    end
    return y
end

function _apply_single_term!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking, chnl::Channel)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    rows = rowvals(x)
    vals = nonzeros(x)
    lk = ReentrantLock()
    tforeach(basis_chunks; scheduler) do local_cols
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        lo = first(local_cols)
        hi = last(local_cols)
        for col in axes(x, 2)
            for ptr in nzrange(x, col)
                n = rows[ptr]
                (n < lo || n > hi) && continue
                xn = vals[ptr]
                state = basisstate(n, space)
                newstate, _amp = _apply_local_operators(term, state, space, precomp)
                if !iszero(_amp)
                    outind = state_index(newstate, space)
                    amp = (conjugate ? conj(_amp) : _amp) * coeff
                    if !_skip_outside(outind, projection)
                        local_mat[outind, col] += amp * xn
                    end
                end
            end
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
        local_mat
    end
    return y
end


# Each chunk of columns is built as its own sparse matrix by the serial blocked build, and the chunks are concatenated.
function _terms_matrix(::Type{T}, terms, coeff, space, ::EagerSparseRepr, chunking::ThreadedChunking; kwargs...) where T
    precomps = _precomputed_terms(terms, space)
    mats = tmap(_state_chunk_data(space, chunking); scheduler=chunking.scheduler) do cols
        _blocked_sparse_matrix(T, precomps, coeff, space, cols; kwargs...)
    end
    return _narrowed(hcat(mats...))
end

# Each task writes the columns of its chunk of the shared matrix.
function _terms_matrix(::Type{T}, terms, coeff, space, repr::EagerDenseRepr, chunking::ThreadedChunking; kwargs...) where T
    precomps = _precomputed_terms(terms, space)
    m = matrix_accumulator(T, length(precomps), (dim(space), dim(space)), repr)
    tforeach(_state_chunk_data(space, chunking); scheduler=chunking.scheduler) do cols
        for (term, precomp) in precomps
            chunked_operator_indices_and_amplitudes_generic!(m, term, space, precomp, cols, 0; kwargs...)
        end
    end
    iszero(coeff) || add_identity!!(m, coeff, space)
    return m
end

# On a ProductSpace, the Kronecker-product matrices of the terms are built in parallel and summed.
function __matrix_representation(op::NCAdd, bases, space::ProductSpace, repr::Union{EagerSparseRepr,EagerDenseRepr}, chunking::ThreadedChunking; tree_split=10, kwargs...)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("The chunking scheduler must be an OhMyThreads `Scheduler`, got a `$(typeof(scheduler))`."))
    matrices = tmap(collect(NCterms(op)); scheduler) do term
        _matrix_representation(term, bases, space, repr, NoChunking(); kwargs...)
    end
    return _add_identity_term(_sum_tree_threaded(matrices, tree_split, scheduler), op.coeff, bases, space, repr)
end

function _sum_tree_threaded(matrices, k, scheduler)
    n = length(matrices)
    n == 0 && throw(ArgumentError("Cannot sum an empty collection of matrices."))
    n == 1 && return only(matrices)
    n <= k && return map(+, matrices...)

    chunk_size = cld(n, k)
    chunks = collect(Iterators.partition(matrices, chunk_size))

    partials = tmap(chunks; scheduler) do chunk
        _sum_tree_threaded(collect(chunk), k, scheduler)
    end
    return map(+, partials...)
end

end
