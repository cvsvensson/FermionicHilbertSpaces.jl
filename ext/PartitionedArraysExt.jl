module PartitionedArraysExt

import PartitionedArrays: DebugArray, own_to_global, psparse, uniform_partition
import FermionicHilbertSpaces: NonCommutativeProducts.NCAdd, NonCommutativeProducts.NCMul, ProductOperator, PartitionedSparseRepr, _apply_local_operators, _factorized_term_matrix_representation, _matrix_representation_single_space, _precomputation_before_operator_application, _sum_matrices, _term_matrix_representation, basisstate, dim, mat_eltype, state_index

function _resolve_ranks(repr::PartitionedSparseRepr)
    repr.nparts > 0 || throw(ArgumentError("nparts must be positive, got $(repr.nparts)."))
    ranks_source = LinearIndices((repr.nparts,))

    if isnothing(repr.backend)
        return DebugArray(ranks_source)
    elseif repr.backend isa AbstractArray
        return repr.backend
    elseif repr.backend isa Function
        return repr.backend(ranks_source)
    elseif repr.backend isa DataType
        return repr.backend(ranks_source)
    end

    throw(ArgumentError("Invalid backend $(repr.backend). Provide either a distributed ranks array, a backend function like distribute_with_mpi, or a constructor like DebugArray."))
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

function _coo_from_single_term(op, space, col_partition; projection)
    precomp = _precomputation_before_operator_application(op, space)
    T = mat_eltype(op)

    coo = map(col_partition) do local_cols
        own_cols = own_to_global(local_cols)
        outinds = Int[]
        ininds = Int[]
        vals = T[]
        sizehint!(vals, length(own_cols))
        sizehint!(outinds, length(own_cols))
        sizehint!(ininds, length(own_cols))
        for inind in own_cols
            state = basisstate(inind, space)
            newstate, amp = _apply_local_operators(op, state, space, precomp)
            if !iszero(amp)
                outind = state_index(newstate, space)
                if iszero(outind)
                    projection || throw(ArgumentError("Operator maps outside of the provided space. Set projection=true to ignore those states."))
                else
                    push!(outinds, outind)
                    push!(ininds, inind)
                    push!(vals, amp)
                end
            end
        end
        (outinds, ininds, vals)
    end

    I = map(t -> t[1], coo)
    J = map(t -> t[2], coo)
    V = map(t -> t[3], coo)
    return I, J, V
end

function _coo_from_add(op::NCAdd, space, col_partition; projection)
    T = mat_eltype(op)

    coo = map(col_partition) do local_cols
        own_cols = own_to_global(local_cols)
        outinds = Int[]
        ininds = Int[]
        vals = T[]
        for inind in own_cols
            state = basisstate(inind, space)
            for (term, coeff) in op.dict
                precomp = _precomputation_before_operator_application(term, space)
                newstate, amp = _apply_local_operators(coeff * term, state, space, precomp)
                if !iszero(amp)
                    outind = state_index(newstate, space)
                    if iszero(outind)
                        projection || throw(ArgumentError("Operator maps outside of the provided space. Set projection=true to ignore those states."))
                    else
                        push!(outinds, outind)
                        push!(ininds, inind)
                        push!(vals, amp)
                    end
                end
            end
            if !iszero(op.coeff)
                push!(outinds, inind)
                push!(ininds, inind)
                push!(vals, op.coeff)
            end
        end
        (outinds, ininds, vals)
    end

    I = map(t -> t[1], coo)
    J = map(t -> t[2], coo)
    V = map(t -> t[3], coo)
    return I, J, V
end

_assemble_psparse(I, J, V, row_partition, col_partition) = fetch(psparse(I, J, V, row_partition, col_partition))

function _term_matrix_representation(op::NCMul, H, repr::PartitionedSparseRepr; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    I, J, V = _coo_from_single_term(op, H, col_partition; projection=projection)
    _assemble_psparse(I, J, V, row_partition, col_partition)
end

function _factorized_term_matrix_representation(ops::ProductOperator, H, repr::PartitionedSparseRepr; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    I, J, V = _coo_from_single_term(ops, H, col_partition; projection=projection)
    _assemble_psparse(I, J, V, row_partition, col_partition)
end

function _matrix_representation_single_space(op::NCAdd, H, repr::PartitionedSparseRepr; projection=false, kwargs...)
    row_partition, col_partition = _resolve_partitions(repr, dim(H))
    I, J, V = _coo_from_add(op, H, col_partition; projection=projection)
    _assemble_psparse(I, J, V, row_partition, col_partition)
end

_sum_matrices(matrices, ::PartitionedSparseRepr; tree_split=10) = map(+, matrices...)

end
