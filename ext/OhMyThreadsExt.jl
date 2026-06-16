module OhMyThreadsExt
using SparseArrays
using OhMyThreads
using FermionicHilbertSpaces
using LinearAlgebra
import SciMLOperators: cache_operator

import OhMyThreads: Schedulers.chunking_args, Consecutive
import FermionicHilbertSpaces: NCAdd, NCterms, EagerDenseRepr, EagerSparseRepr, TermChunking, StateChunking, ProductOperator, _matrix_representation, _matrix_representation_single_space, __matrix_representation, finalize!, mat_eltype, matrix_accumulator, operator_indices_and_amplitudes!, push_inds_amps!, NoChunking, chunked_operator_indices_and_amplitudes!, partition_product, LazyOperator, lazy_mul!, _apply_single_term!, _apply_local_operators, precomputation_before_operator_application, basisstate, state_index, dim, _lazy_output_prototype, NonCommutativeProducts.NCMul

function _reduce_add!(y, partials)
    isempty(partials) && return y
    # Build a lazy nested broadcast tree: p1 .+ p2 .+ p3 .+ ...
    bc = reduce((acc, p) -> Base.broadcasted(+, acc, p), partials)
    # Materialize in one fused pass: y[i] += p1[i] + p2[i] + ...
    broadcast!(+, y, y, bc)
    return y
end

function _state_chunk_data(space, chunking::StateChunking)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    n = get_nchunks(scheduler)
    basis_chunks = collect(chunks(1:dim(space); n, split=Consecutive()))
    return basis_chunks
end

function _term_chunk_data(op::NCAdd, chunking::TermChunking)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    n = get_nchunks(scheduler)
    term_chunks = collect(chunks(collect(NCterms(op)); n, split=Consecutive()))
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

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:TermChunking}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    op = L.op
    term_chunks = _term_chunk_data(op, L.chunking)
    cache_operator!(L, y)
    chnl = L.cache
    scheduler = L.chunking.scheduler
    lk = ReentrantLock()
    tforeach(term_chunks; scheduler) do local_terms
        local_mat = take!(chnl)
        fill!(local_mat, zero(eltype(local_mat)))
        for term in local_terms
            precomp = precomputation_before_operator_application(term, L.space)
            _apply_single_term!(local_mat, x, L.space, term, precomp, α, L.conjugate, L.projection, NoChunking())
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

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:StateChunking}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    cache_operator!(L, y)
    for term in NCterms(L.op)
        precomp = precomputation_before_operator_application(term, L.space)
        _apply_single_term!(y, x, L.space, term, precomp, α, L.conjugate, L.projection, L.chunking, L.cache)
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
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
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
                if !projection || !iszero(outind)
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
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
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
                if !projection || !iszero(outind)
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
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
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
                if !projection || !iszero(outind)
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
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
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
                    if !projection || !iszero(outind)
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


function __matrix_representation(op::NCAdd, bases, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, chunking::TermChunking; tree_split=10, kwargs...)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    terms = collect(NCterms(op))
    mats = tmap(terms; scheduler) do term
        _matrix_representation(term, bases, space, repr, NoChunking(); kwargs...)
    end
    mat = _sum_tree_threaded(mats, tree_split, scheduler)
    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, bases, space, repr, NoChunking(); kwargs...)
    end
    return mat
end


function __matrix_representation(op::NCAdd, bases, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, chunking::StateChunking; kwargs...)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    n = get_nchunks(scheduler)
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())
    T = mat_eltype(op)
    ncterms = if length(bases) > 1
        map(NCterms(op)) do term
            partition_product(term, bases, factors(space))
        end
    else
        collect(NCterms(op))
    end
    Nterms = length(ncterms)
    local_mats = tmap(basis_chunks; scheduler) do local_cols
        cols = collect(local_cols)
        accum = matrix_accumulator(T, Nterms, (dim(space), length(cols)), repr)
        col_offset = first(cols) - 1
        for term in ncterms
            chunked_operator_indices_and_amplitudes!(accum, term, space, cols, col_offset; kwargs...)
        end
        finalize!(accum, dim(space), length(cols))
    end
    mat = hcat(local_mats...)
    if !iszero(op.coeff)
        mat .+= op.coeff * _matrix_representation(missing, bases, space, repr, NoChunking(); kwargs...)
    end
    return mat
end

function _sum_tree_threaded(matrices, k, scheduler)
    n = length(matrices)
    n == 0 && throw(ArgumentError("Empty collection"))
    n == 1 && return only(matrices)
    n <= k && return map(+, matrices...)

    chunk_size = cld(n, k)
    chunks = collect(Iterators.partition(matrices, chunk_size))

    partials = tmap(chunks; scheduler) do chunk
        _sum_tree_threaded(collect(chunk), k, scheduler)
    end
    return map(+, partials...)
end

function _matrix_representation_single_space(op::NCAdd, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, chunking::TermChunking; tree_split=10, kwargs...)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    n = get_nchunks(scheduler)
    chunked_terms = chunks(collect(NCterms(op)); n)
    mats = tmap(chunked_terms; scheduler) do terms
        accumulator = matrix_accumulator(op, space, repr)
        for term in terms
            operator_indices_and_amplitudes!(accumulator, term, space; kwargs...)
        end
        return finalize!(accumulator, space)
    end
    mat = _sum_tree_threaded(mats, tree_split, scheduler)

    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, nothing, space, repr, NoChunking(); kwargs...)
    end
    return mat
end

function _matrix_representation_single_space(op::NCAdd, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, chunking::StateChunking; kwargs...)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    n = get_nchunks(scheduler)
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())
    T = mat_eltype(op)
    ncterms = collect(NCterms(op))
    nterms = length(ncterms)
    local_mats = tmap(basis_chunks; scheduler) do local_cols
        cols = collect(local_cols)
        accum = matrix_accumulator(T, nterms, (dim(space), length(cols)), repr)
        col_offset = first(cols) - 1
        for term in ncterms
            chunked_operator_indices_and_amplitudes!(accum, term, space, cols, col_offset; kwargs...)
        end
        finalize!(accum, dim(space), length(cols))
    end
    mat = hcat(local_mats...)
    if !iszero(op.coeff)
        mat .+= op.coeff * _matrix_representation(missing, nothing, space, repr, NoChunking(); kwargs...)
    end
    return mat
end

end
