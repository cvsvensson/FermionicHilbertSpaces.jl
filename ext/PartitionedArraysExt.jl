module PartitionedArraysExt

import PartitionedArrays: DebugArray, own_to_global, psparse, uniform_partition
import FermionicHilbertSpaces: NonCommutativeProducts.NCAdd, NonCommutativeProducts.NCMul, ProductOperator, PartitionedSparseRepr, _factorized_term_matrix_representation, _matrix_representation_single_space, _sum_matrices, _term_matrix_representation, dim, mat_eltype, operator_indices_and_amplitudes!, sparse_matrix_accumulator

function _resolve_ranks(repr::PartitionedSparseRepr)
    repr.nparts > 0 || throw(ArgumentError("nparts must be positive, got $(repr.nparts)."))
    ranks_source = LinearIndices((repr.nparts,))
    return repr.backend(ranks_source)
end

function _resolve_partitions(repr::PartitionedSparseRepr, n::Int)
    has_row = !isnothing(repr.row_partition)
    has_col = !isnothing(repr.col_partition)
    if has_row != has_col
        throw(ArgumentError("row_partition and col_partition must be both provided or both omitted."))
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
    coo = map(col_partition) do local_cols
        cols = own_to_global(local_cols)
        accum = sparse_matrix_accumulator(T, max(length(cols), 16))
        for term in terms
            operator_indices_and_amplitudes!(accum, term, space, cols; projection)
        end
        if !iszero(identity_coeff)
            for inind in cols
                push!(accum[1], inind)
                push!(accum[2], inind)
                push!(accum[3], identity_coeff)
            end
        end
        accum
    end

    return map(t -> t[1], coo), map(t -> t[2], coo), map(t -> t[3], coo)
end

_assemble_psparse(I, J, V, row_partition, col_partition) = fetch(psparse(I, J, V, row_partition, col_partition))

function _term_matrix_representation(op::NCMul, H, repr::PartitionedSparseRepr, chunking; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    I, J, V = _coo_from_chunked_terms((op,), H, col_partition; projection=projection)
    _assemble_psparse(I, J, V, row_partition, col_partition)
end

# function _factorized_term_matrix_representation(ops::ProductOperator, H, repr::PartitionedSparseRepr, chunking; projection=false, kwargs...)
#     row_partition, col_partition = _resolve_partitions(repr, dim(H))
#     I, J, V = _coo_from_chunked_terms((ops,), H, col_partition; projection=projection)
#     _assemble_psparse(I, J, V, row_partition, col_partition)
# end

function _matrix_representation_single_space(op::NCAdd, H, repr::PartitionedSparseRepr, chunking; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    terms = Tuple(coeff * term for (term, coeff) in op.dict)
    I, J, V = _coo_from_chunked_terms(terms, H, col_partition; projection=projection, identity_coeff=op.coeff)
    _assemble_psparse(I, J, V, row_partition, col_partition)
end

_sum_matrices(matrices, ::PartitionedSparseRepr; tree_split=10) = map(+, matrices...)

end
