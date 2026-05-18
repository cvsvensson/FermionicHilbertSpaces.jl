module OhMyThreadsExt

import OhMyThreads: Schedulers.chunking_args, chunks, Scheduler, tmapreduce, tmap
import FermionicHilbertSpaces: NCAdd, NCterms, EagerDenseRepr, EagerSparseRepr, _matrix_representation, _matrix_representation_single_space, _matrix_representation_threaded, mat_eltype, matrix_accumulator, operator_indices_and_amplitudes!, finalize!


function _matrix_representation_threaded(op::NCAdd, bases, space, repr, scheduler::Scheduler; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space_threaded(op, space, repr, scheduler; kwargs...)
    end
    return __matrix_representation_threaded(op, bases, space, repr, scheduler; kwargs...)
end
function __matrix_representation_threaded(op::NCAdd, bases, space, repr, scheduler::Scheduler; tree_split=10, kwargs...)
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

end
