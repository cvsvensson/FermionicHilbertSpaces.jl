module OhMyThreadsExt
# using FermionicHilbertSpaces
using SparseArrays
import OhMyThreads: Schedulers.chunking_args, chunks, Scheduler, tmapreduce, tmap, StaticScheduler, DynamicScheduler, tforeach, Consecutive
import FermionicHilbertSpaces: AbstractHilbertSpace, ChunkedSparseRepr, NCAdd, NCterms, EagerDenseRepr, EagerSparseRepr, _apply_local_operators, _matrix_representation, _matrix_representation_single_space, _matrix_representation_threaded, _precomputation_before_operator_application, _term_matrix_representation, basisstate, dim, finalize!, mat_eltype, matrix_accumulator, operator_indices_and_amplitudes!, state_index, push_inds_amps!


function _matrix_representation_threaded(op::NCAdd, bases, space, repr, scheduler::Scheduler; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space_threaded(op, space, repr, scheduler; kwargs...)
    end
    return __matrix_representation_threaded(op, bases, space, repr, scheduler; kwargs...)
end
function __matrix_representation_threaded(op::NCAdd, bases, space, repr::Union{EagerSparseRepr, EagerDenseRepr}, scheduler::Scheduler; tree_split=10, kwargs...)
    terms = collect(NCterms(op))
    mats = tmap(terms; scheduler) do term
        _matrix_representation(term, bases, space, repr; kwargs...)
    end
    mat = _sum_tree_threaded(mats, tree_split, scheduler)
    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, bases, space, repr; kwargs...)
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

function _matrix_representation_single_space_threaded(op::NCAdd, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, scheduler::Scheduler; tree_split=10, kwargs...)
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    chunked_terms = chunks(collect(NCterms(op)); n)
    T = mat_eltype(op)
    mats = tmap(chunked_terms; scheduler) do terms
        accumulator = matrix_accumulator(T, length(terms), space, repr)
        for term in terms
            operator_indices_and_amplitudes!(accumulator, term, space; kwargs...)
        end
        return finalize!(accumulator, space)
    end
    mat = _sum_tree_threaded(mats, tree_split, scheduler)

    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, nothing, space, repr; kwargs...)
    end
    return mat
end

function _matrix_representation_single_space_threaded(op::NCAdd, space, repr::ChunkedSparseRepr, scheduler; projection=false, kwargs...)
    _n = chunking_args(scheduler).n
    n = isnothing(_n) ? 1 : _n
    basis_chunks = chunks(1:dim(space); n, split = Consecutive())
    terms = collect(NCterms(op))
    precomps = map(terms) do term
        _precomputation_before_operator_application(term, space)
    end
    local_mats = tmap(basis_chunks; scheduler) do local_cols
        accum = matrix_accumulator(op, space, EagerSparseRepr())
        for (term, precomp) in zip(terms, precomps)
            for inind in local_cols
                state = basisstate(inind, space)
                newstate, amp = _apply_local_operators(term, state, space, precomp)
                if !iszero(amp)
                    outind = state_index(newstate, space)
                    if iszero(outind)
                        projection || throw(ArgumentError("Operator maps outside of the provided space. Set projection=true to ignore those states."))
                    else
                        push_inds_amps!(accum, outind, inind - first(local_cols) + 1, amp) 
                    end
                end
            end
        end
        finalize!(accum, dim(space), length(local_cols))
    end
    mat = hcat(local_mats...)

    if !iszero(op.coeff)
        mat .+= op.coeff * _matrix_representation(missing, nothing, space, EagerSparseRepr())
    end
    return mat
end


function __matrix_representation_threaded(op::NCAdd, bases, space, repr::ChunkedSparseRepr, scheduler::Scheduler; tree_split=10, kwargs...)
    matrices = [_matrix_representation(term, bases, space, repr; kwargs...) for term in NCterms(op)]
    _sum_tree_threaded(matrices, tree_split, scheduler) + op.coeff * _matrix_representation(missing, bases, space, EagerSparseRepr(); kwargs...)
end


function _term_matrix_representation(op, H::AbstractHilbertSpace, repr::ChunkedSparseRepr; kwargs...)
    _term_matrix_representation(op, H, EagerSparseRepr(); kwargs...)
end

end
