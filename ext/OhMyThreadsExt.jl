module OhMyThreadsExt

import OhMyThreads: Schedulers.chunking_args, chunks, Scheduler, tmapreduce
import FermionicHilbertSpaces: NCAdd, NCterms, EagerDenseRepr, EagerSparseRepr, _matrix_representation, _matrix_representation_single_space, _matrix_representation_threaded, mat_eltype, matrix_accumulator, operator_indices_and_amplitudes!, finalize!


function _matrix_representation_threaded(op::NCAdd, bases, space, repr, scheduler::Scheduler; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space_threaded(op, space, repr, scheduler; kwargs...)
    end
    return __matrix_representation_threaded(op, bases, space, repr, scheduler; kwargs...)
end
function __matrix_representation_threaded(op::NCAdd, bases, space, repr, scheduler::Scheduler; kwargs...)
    terms = collect(NCterms(op))
    mat = tmapreduce(+, terms; scheduler) do term
        _matrix_representation(term, bases, space, repr; kwargs...)
    end
    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, bases, space, repr; kwargs...)
    end
    return mat
end

function _matrix_representation_single_space_threaded(op::NCAdd, space, repr::Union{EagerSparseRepr,EagerDenseRepr}, scheduler::Scheduler; kwargs...)
    n = chunking_args(scheduler).n
    chunked_terms = chunks(collect(NCterms(op)); n)
    T = mat_eltype(op)
    mat = tmapreduce(+, chunked_terms; scheduler) do terms
        accumulator = matrix_accumulator(T, length(terms), space, repr)
        for term in terms
            operator_indices_and_amplitudes!(accumulator, term, space; kwargs...)
        end
        finalize!(accumulator, space)
    end
    if !iszero(op.coeff)
        mat += op.coeff * _matrix_representation(missing, nothing, space, repr; kwargs...)
    end
    return mat
end

end
