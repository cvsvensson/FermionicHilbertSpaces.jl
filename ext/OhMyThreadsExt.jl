module OhMyThreadsExt
using SparseArrays
using OhMyThreads
using FermionicHilbertSpaces
using LinearAlgebra

import OhMyThreads: Schedulers.chunking_args, Consecutive#, chunks, Scheduler, tmap, Consecutive
import FermionicHilbertSpaces: NCAdd, NCterms, EagerDenseRepr, EagerSparseRepr, TermChunking, StateChunking, _matrix_representation, _matrix_representation_single_space, __matrix_representation, finalize!, mat_eltype, matrix_accumulator, operator_indices_and_amplitudes!, push_inds_amps!, NoChunking, chunked_operator_indices_and_amplitudes!, partition_product, LazyOperator, lazy_mul!, _apply_single_term!, _apply_local_operators, _precomputation_before_operator_application, basisstate, state_index, dim

function _reduce_add!(y, partials)
    for part in partials
        y .+= part
    end
    return y
end
function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCAdd,<:Any,<:Any,<:TermChunking}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    op = L.op
    scheduler = L.chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    terms = collect(op.dict)

    partials = tmap(terms; scheduler) do termcoeff
        term, coeff = termcoeff
        local_mat = zero(y)
        precomp = _precomputation_before_operator_application(term, L.space)
        _apply_single_term!(local_mat, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection, NoChunking())
        local_mat
    end
    _reduce_add!(y, partials)

    if !iszero(op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(op.coeff) : op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

function _apply_single_term!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())

    partials = tmap(basis_chunks; scheduler) do local_cols
        local_mat = zero(y) #TODO: preallocate cache and reuse
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
        local_mat
    end
    _reduce_add!(y, partials)
    return y
end

function _apply_single_term!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())
    inds, vals = findnz(x)

    partials = tmap(basis_chunks; scheduler) do local_cols
        local_mat = zero(y)
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
        local_mat
    end
    _reduce_add!(y, partials)
    return y
end

function _apply_single_term!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())

    partials = tmap(basis_chunks; scheduler) do local_cols
        local_mat = zero(y)
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
        local_mat
    end
    _reduce_add!(y, partials)
    return y
end

function _apply_single_term!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, _coeff, conjugate, projection, chunking::StateChunking)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    scheduler = chunking.scheduler
    scheduler isa Scheduler || throw(ArgumentError("Unsupported chunking scheduler of type $(typeof(scheduler)). Expected an OhMyThreads Scheduler."))
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    basis_chunks = chunks(1:dim(space); n, split=Consecutive())
    rows = rowvals(x)
    vals = nonzeros(x)

    partials = tmap(basis_chunks; scheduler) do local_cols
        local_mat = zero(y)
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
        local_mat
    end
    _reduce_add!(y, partials)
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
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
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
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
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
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
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
