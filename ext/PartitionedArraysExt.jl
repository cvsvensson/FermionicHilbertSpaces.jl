module PartitionedArraysExt

import PartitionedArrays: own_to_global, psparse, uniform_partition
using NonCommutativeProducts: NCAdd, NCMul, NCterms
import FermionicHilbertSpaces: PartitionedSparseRepr, _matrix_representation_single_space, _term_matrix_representation, _collect_ijv, _precomputed_terms, dim, mat_eltype

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

function _coo_from_chunked_terms(terms, space, col_partition; projection, identity_coeff=0)
    T = isempty(terms) ? typeof(identity_coeff) : mat_eltype(first(terms))
    precomps = _precomputed_terms(terms, space)
    coo = map(col_partition) do local_cols
        _collect_ijv(T, precomps, identity_coeff, space, own_to_global(local_cols), 0; projection)
    end

    return map(t -> t[1], coo), map(t -> t[2], coo), map(t -> t[3], coo)
end

function _term_matrix_representation(op::NCMul, H, repr::PartitionedSparseRepr, chunking; projection, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    I, J, V = _coo_from_chunked_terms((op,), H, col_partition; projection)
    fetch(psparse(I, J, V, row_partition, col_partition))
end

function _matrix_representation_single_space(op::NCAdd, H, repr::PartitionedSparseRepr, chunking; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    terms = collect(NCterms(op))
    I, J, V = _coo_from_chunked_terms(terms, H, col_partition; projection, identity_coeff=op.coeff)
    fetch(psparse(I, J, V, row_partition, col_partition))
end

end
