const SPARSE_BLOCK_SIZE = 4096

_terms_matrix(::Type{T}, terms, coeff, space, ::EagerSparseRepr, ::NoChunking; kwargs...) where T = _narrowed(_blocked_sparse_matrix(T, _precomputed_terms(terms, space), coeff, space; kwargs...))

# `m` with its element type narrowed to that of its stored entries when `eltype(m)` is abstract
_narrowed(m) = isconcretetype(eltype(m)) ? m : SparseMatrixCSC(size(m)..., SparseArrays.getcolptr(m), rowvals(m), identity.(nonzeros(m)))

_precomputed_terms(terms, space) = [(term, precomputation_before_operator_application(term, space)) for term in terms]

# `len` entries from `done` units of work, extrapolated with a margin to `todo` more units
_extrapolated_length(len, done, todo) = len + ceil(Int, len / done * todo * 3 / 2)

# After k of N terms: once the next term, estimated as the average of the first k, and `extra` entries would exceed
# `capacity`, reserves room for the remaining terms and the `extra` entries. Returns the new capacity.
function _reserve_remaining!(accumulator, capacity, k, N, extra)
    len = length(first(accumulator))
    len + cld(len, k) + extra <= capacity && return capacity
    capacity = _extrapolated_length(len, k, N - k) + extra
    foreach(v -> sizehint!(v, capacity; shrink=false), accumulator)
    return capacity
end

# Replaces the entries of `block` by those of the columns `incols` of the matrix, stored at the column indices
# `incols .- offset`. `capacity` is the number of entries reserved in `block`; returns the new one.
function _collect_block!(block, capacity, precomps, coeff, space, incols, offset; kwargs...)
    foreach(empty!, block)
    extra = iszero(coeff) ? 0 : length(incols)
    for (k, (term, precomp)) in enumerate(precomps)
        chunked_operator_indices_and_amplitudes_generic!(block, term, space, precomp, incols, offset; kwargs...)
        capacity = _reserve_remaining!(block, capacity, k, length(precomps), extra)
    end
    iszero(coeff) || add_identity!!(block, coeff, space, incols, offset)
    return capacity
end

# Builds the columns `incols` of the matrix one block of columns at a time: the entries of a block are collected
# term by term, then sorted into columns, with duplicate entries summed in the order of the terms.
# A nonempty range of columns that fits in one block is built by `sparse!` from that block.
# `precomps` holds pairs `(term, precomp)` from `_precomputed_terms`.
function _blocked_sparse_matrix(::Type{T}, precomps, coeff, space, incols=1:dim(space); kwargs...) where T
    N = dim(space)
    M = length(incols)
    offset = first(incols) - 1
    B = SPARSE_BLOCK_SIZE
    blockcapacity = _sparse_length_guess(length(precomps), min(B, M))
    block = (rows, cols, amps) = sparse_matrix_accumulator(T, (blockcapacity, blockcapacity))
    if 0 < M <= B
        _collect_block!(block, blockcapacity, precomps, coeff, space, incols, offset; kwargs...)
        return _trimmed!(SparseArrays.sparse!(rows, cols, amps, N, M), length(rows))
    end
    colptr = Vector{Int}(undef, M + 1)
    colptr[1] = 1
    rowval = Int[]
    nzval = T[]
    capacity = 0
    sortedrows = Int[]
    sortedamps = T[]
    mergebuffers = (zeros(Int, N), Int[], T[])
    colstarts = Vector{Int}(undef, B + 1)
    for c0 in 0:B:M-1
        nb = min(B, M - c0)
        blockcapacity = _collect_block!(block, blockcapacity, precomps, coeff, space, offset+c0+1:offset+c0+nb, offset + c0; kwargs...)
        fill!(colstarts, 0)
        for j in cols
            colstarts[j+1] += 1
        end
        colstarts[1] = 1
        for j in 1:nb
            colstarts[j+1] += colstarts[j]
        end
        resize!(sortedrows, length(rows))
        resize!(sortedamps, length(rows))
        for (r, j, a) in zip(rows, cols, amps)
            q = colstarts[j]
            sortedrows[q] = r
            sortedamps[q] = a
            colstarts[j] = q + 1
        end
        # Now column j ends at colstarts[j]-1. The merged columns are written to the first n entries.
        n = 0
        start = 1
        for j in 1:nb
            stop = colstarts[j] - 1
            n = _sort_merge_column!(sortedrows, sortedamps, start, stop, n, mergebuffers)
            colptr[c0+j+1] = length(rowval) + n + 1
            start = stop + 1
        end
        total = length(rowval) + n
        if total > capacity
            capacity = _extrapolated_length(total, c0 + nb, M - c0 - nb)
            sizehint!(rowval, capacity)
            sizehint!(nzval, capacity)
        end
        append!(rowval, view(sortedrows, 1:n))
        append!(nzval, view(sortedamps, 1:n))
    end
    foreach(v -> sizehint!(v, length(v)), (rowval, nzval))
    return SparseMatrixCSC(N, M, colptr, rowval, nzval)
end

# Writes the entries start:stop of `rows` and `amps` after entry n, sorted by row and with the entries of equal rows
# summed in their order. Returns the new number of written entries.
# Equal rows are merged before sorting, with `slots[r]` the position of row r in `rowbuf` and `ampbuf`.
# `slots` has one entry per row of the matrix and is all zeros between calls.
function _sort_merge_column!(rows, amps, start, stop, n, (slots, rowbuf, ampbuf))
    resize!(rowbuf, stop - start + 1)
    resize!(ampbuf, stop - start + 1)
    m = 0
    for k in start:stop
        r = rows[k]
        if iszero(slots[r])
            m += 1
            slots[r] = m
            rowbuf[m] = r
            ampbuf[m] = amps[k]
        else
            ampbuf[slots[r]] += amps[k]
        end
    end
    sort!(view(rowbuf, 1:m))
    for r in view(rowbuf, 1:m)
        n += 1
        rows[n] = r
        amps[n] = ampbuf[slots[r]]
        slots[r] = 0
    end
    return n
end


@testitem "Sparse matrix of a sum of many terms" begin
    using SparseArrays, LinearAlgebra
    @fermions f
    N = 14
    H = hilbert_space(f, 1:N, NumberConservation(6:7))
    # Diagonal terms and the identity add up in the same entries
    terms = [[(1 + 0.5im) * f[i]' * f[j] for i in 1:N for j in i+1:N]; [f[i]' * f[i] for i in 1:N]]
    op = sum(terms) + 1.5
    M = matrix_representation(op, H)
    entries = map(findnz, [matrix_representation(term, H) for term in terms])
    @test M == sparse((reduce(vcat, getindex.(entries, k)) for k in 1:3)..., dim(H), dim(H)) + 1.5I

    # f[1]' and f[1] map some states out of the space
    @test_throws ArgumentError matrix_representation(f[1]' + f[1] + 1, H)
    @test matrix_representation(f[1]' + f[1] + 1, H; projection=true) ==
          matrix_representation(f[1]', H; projection=true) + matrix_representation(f[1], H; projection=true) + I

    # A space without states
    @test size(matrix_representation(f[1]' * f[2] + 1, constrain_space(hilbert_space(f, 1:2), NumberConservation(5)))) == (0, 0)

    # A constrained space with several groups, and an identity coefficient that widens the element type
    @boson b
    Hprod = tensor_product(hilbert_space(f, 1:10), hilbert_space(b, 7))
    Hc = tensor_product(hilbert_space(f, 1:10), hilbert_space(b, 7); constraint=NumberConservation(1:15))
    inds = [state_index(state, Hprod) for state in basisstates(Hc)]
    op = sum(f[i]' * f[i+1] + hc for i in 1:9) + sum(f[i]' * b + hc for i in 1:10) + 2 * b' * b + f[1]' + hc + 0.5im
    M = matrix_representation(op, Hc; projection=true)
    @test M isa SparseMatrixCSC{ComplexF64}
    @test M == matrix_representation(op, Hprod)[inds, inds]
end
