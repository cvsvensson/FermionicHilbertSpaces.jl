module PartitionedArraysExt

import PartitionedArrays: own_to_global, psparse, tuple_of_arrays, uniform_partition
import FermionicHilbertSpaces: PartitionedSparseRepr, _terms_matrix, _collect_ijv, _precomputed_terms, dim

function _resolve_ranks(repr::PartitionedSparseRepr)
    repr.nparts > 0 || throw(ArgumentError("`nparts` must be positive, got $(repr.nparts)."))
    ranks_source = LinearIndices((repr.nparts,))
    return repr.backend(ranks_source)
end

function _resolve_partitions(repr::PartitionedSparseRepr, n::Int)
    has_row = !isnothing(repr.row_partition)
    has_col = !isnothing(repr.col_partition)
    if has_row != has_col
        throw(ArgumentError("`row_partition` and `col_partition` must be both provided or both omitted."))
    end
    if has_row
        return repr.row_partition, repr.col_partition
    end

    ranks = _resolve_ranks(repr)
    row_partition = uniform_partition(ranks, n)
    col_partition = uniform_partition(ranks, n)
    return row_partition, col_partition
end

function _terms_matrix(::Type{T}, terms, coeff, space, repr::PartitionedSparseRepr, chunking; kwargs...) where {T}
    row_partition, col_partition = _resolve_partitions(repr, dim(space))
    precomps = _precomputed_terms(terms, space)
    I, J, V = tuple_of_arrays(map(cols -> _collect_ijv(T, precomps, coeff, space, own_to_global(cols), 0; kwargs...), col_partition))
    fetch(psparse(I, J, V, row_partition, col_partition))
end

end
