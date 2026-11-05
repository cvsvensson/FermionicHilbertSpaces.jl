module OhMyThreadsExt
using SparseArrays: SparseArrays, nonzeros, nzrange, rowvals
using OhMyThreads: OhMyThreads, Scheduler, chunks, tforeach, tmap
using FermionicHilbertSpaces: FermionicHilbertSpaces
using LinearAlgebra: LinearAlgebra, rmul!
using NonCommutativeProducts: NCAdd, NCMul
import SciMLOperators: cache_operator

import OhMyThreads: Schedulers.chunking_args, Consecutive
import FermionicHilbertSpaces: EagerDenseRepr, EagerSparseRepr, TermChunking, StateChunking, ProductOperator, _terms_matrix, _precomputed_terms, _blocked_sparse_matrix, _narrowed, matrix_accumulator, add_identity!!, NoChunking, chunked_operator_indices_and_amplitudes_generic!, LazyOperator, lazy_mul!, _apply_single_term!, _apply_local_operators, basisstate, state_index, _skip_outside, dim, _lazy_output_prototype, cache_operator!, _lazy_terms

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
    cache_operator!(L, y)
    _apply_term_chunks!(y, x, L, α, L.cache)
    if !iszero(op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(op.coeff) : op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

# The buffers come in as the argument `chnl`, not from the untyped field `L.cache`, so the loops are type stable.
function _apply_term_chunks!(y, x, L, α, chnl::Channel)
    lk = ReentrantLock()
    tforeach(_term_chunk_data(L.precomp, L.chunking); scheduler = L.chunking.scheduler) do local_terms
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for (term, coeff, precomp) in local_terms
            _apply_single_term!(local_mat, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection, NoChunking())
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
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
    _apply_terms!(y, x, L.space, _lazy_terms(L), α, L.conjugate, L.projection, L.chunking, L.cache)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:ProductOperator,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    cache_operator!(L, y)
    _apply_terms!(y, x, L.space, _lazy_terms(L), α, L.conjugate, L.projection, L.chunking, L.cache)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    cache_operator!(L, y)
    _apply_terms!(y, x, L.space, _lazy_terms(L), α, L.conjugate, L.projection, L.chunking, L.cache)
    if !iszero(L.op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(L.op.coeff) : L.op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

# Each chunk of input states applies all terms into its own buffer, which is added to `y` once.
# The output states of a chunk are scattered over all of `y`, so the buffers are full length.
function _apply_terms!(y, x, space, terms, α, conjugate, projection, chunking::StateChunking, chnl::Channel)
    lk = ReentrantLock()
    tforeach(_state_chunk_data(space, chunking); scheduler = chunking.scheduler) do cols
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for (term, coeff, precomp) in terms
            _apply_chunk!(local_mat, x, space, term, precomp, coeff * α, conjugate, projection, cols)
        end
        lock(lk) do
            y .+= local_mat
        end
        put!(chnl, local_mat)
    end
    return y
end

function _apply_chunk!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, projection, cols)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for n in cols
        xn = x[n] * coeff
        iszero(xn) && continue
        state = basisstate(n, space)
        state, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(state, space)
            if !_skip_outside(outind, projection)
                amp = (conjugate ? conj(_amp) : _amp)
                y[outind] += amp * xn
            end
        end
    end
    return y
end

function _apply_chunk!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, _coeff, conjugate, projection, cols)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    lo, hi = first(cols), last(cols)
    for (n, xn) in zip(SparseArrays.nonzeroinds(x), nonzeros(x))
        (n < lo || n > hi) && continue
        state = basisstate(n, space)
        state, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(state, space)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !_skip_outside(outind, projection)
                y[outind] += amp * xn
            end
        end
    end
    return y
end

function _apply_chunk!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, projection, cols)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for n in cols
        state = basisstate(n, space)
        newstate, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(newstate, space)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !_skip_outside(outind, projection)
                @views y[outind, :] .+= amp .* x[n, :]
            end
        end
    end
    return y
end

function _apply_chunk!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, _coeff, conjugate, projection, cols)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    rows = rowvals(x)
    vals = nonzeros(x)
    lo, hi = first(cols), last(cols)
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
                    y[outind, col] += amp * xn
                end
            end
        end
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

end
