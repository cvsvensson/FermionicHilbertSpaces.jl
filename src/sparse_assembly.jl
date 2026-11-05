const SPARSE_BLOCK_SIZE = 4096

function _terms_matrix(::Type{T}, terms, coeff, space, ::EagerSparseRepr, ::NoChunking; kwargs...) where {T}
    m = _blocked_sparse_matrix(T, _precomputed_terms(terms, space), coeff, space; kwargs...)
    # The chunks of a chunked build are copied by `hcat`, so only the whole matrix releases the room left by the blocks
    dim(space) > SPARSE_BLOCK_SIZE && foreach(v -> sizehint!(v, length(v)), (rowvals(m), nonzeros(m)))
    return _narrowed(m)
end

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
    capacity = max(capacity, _extrapolated_length(len, k, N - k) + extra)
    foreach(v -> sizehint!(v, capacity; shrink=false), accumulator)
    return capacity
end

# The number of entries that the terms push for the columns `incols`, or `nothing` if a term can't tell in advance
_nentries(term, incols) = nothing
function _known_length(precomps, incols)
    total = 0
    for (term, _) in precomps
        n = _nentries(term, incols)
        isnothing(n) && return nothing
        total += n
    end
    return total
end

_total_nentries(term, N) = nothing
# An upper bound on the stored entries of the `N` columns of a matrix, `typemax(Int)` if a term can't tell in advance
function _max_stored_entries(precomps, coeff, N)
    total = iszero(coeff) ? 0 : N
    for (term, _) in precomps
        n = _total_nentries(term, N)
        isnothing(n) && return typemax(Int)
        total += n
    end
    return total
end

# Replaces the entries of `block` by those of the columns `incols` of the matrix, stored at the column indices
# `incols .- offset`. `capacity` is the number of entries reserved in `block`; returns the new one.
function _collect_block!(block, capacity, precomps, coeff, space, incols, offset; kwargs...)
    foreach(empty!, block)
    extra = iszero(coeff) ? 0 : length(incols)
    known = _known_length(precomps, incols)
    capacity = max(capacity, isnothing(known) ? _sparse_length_guess(length(precomps), length(incols)) : known + extra)
    foreach(v -> sizehint!(v, capacity; shrink = false), block)
    for (k, (term, precomp)) in enumerate(precomps)
        chunked_operator_indices_and_amplitudes_generic!(block, term, space, precomp, incols, offset; kwargs...)
        isnothing(known) && (capacity = _reserve_remaining!(block, capacity, k, length(precomps), extra))
    end
    iszero(coeff) || add_identity!!(block, coeff, space, incols, offset)
    return capacity
end

# The entries `(rows, cols, amps)` of the columns `incols` of the matrix, with duplicate entries not summed
function _collect_ijv(::Type{T}, precomps, coeff, space, incols, offset; kwargs...) where {T}
    ijv = sparse_matrix_accumulator(T, (0, 0))
    _collect_block!(ijv, 0, precomps, coeff, space, incols, offset; kwargs...)
    return ijv
end

# Builds the columns `incols` of the matrix one block of columns at a time: the entries of a block are collected
# term by term, then sorted into columns, with duplicate entries summed in the order of the terms.
# A nonempty range of columns that fits in one block is built by `sparse!` from that block. A matrix of more than
# one block can have room for more entries in its vectors.
# `precomps` holds pairs `(term, precomp)` from `_precomputed_terms`.
function _blocked_sparse_matrix(::Type{T}, precomps, coeff, space, incols=1:dim(space); kwargs...) where T
    N = dim(space)
    M = length(incols)
    offset = first(incols) - 1
    # The blocks shrink with the share of the columns of the space in `incols`, so that the chunks of a chunked build
    # together buffer about as many entries as the serial build
    B = max(1, cld(SPARSE_BLOCK_SIZE * M, max(N, 1)))
    if 0 < M <= B
        ijv = _collect_ijv(T, precomps, coeff, space, incols, offset; kwargs...)
        return _trimmed!(SparseArrays.sparse!(ijv..., N, M), length(first(ijv)))
    end
    blockcapacity = 0
    block = (rows, cols, amps) = sparse_matrix_accumulator(T, (0, 0))
    colptr = Vector{Int}(undef, M + 1)
    colptr[1] = 1
    rowval = Int[]
    nzval = T[]
    capacity = 0
    maxentries = M > 0 ? _max_stored_entries(precomps, coeff, N) : 0
    sortedrows = Int[]
    sortedamps = T[]
    mergebuffers = (zeros(Int32, N), Int[], T[])
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
            capacity = min(_extrapolated_length(total, c0 + nb, M - c0 - nb), maxentries)
            rowval = _with_capacity(rowval, capacity)
            nzval = _with_capacity(nzval, capacity)
        end
        append!(rowval, view(sortedrows, 1:n))
        append!(nzval, view(sortedamps, 1:n))
    end
    return SparseMatrixCSC(N, M, colptr, rowval, nzval)
end

# A copy of `v` with room for exactly `capacity` entries. `sizehint!` would round a growing capacity up, and the
# final trim would then copy the vectors once more.
_with_capacity(v, capacity) = resize!(copyto!(similar(v, capacity), v), length(v))

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
    M = representation(op, H)
    entries = map(findnz, [representation(term, H) for term in terms])
    @test M == sparse((reduce(vcat, getindex.(entries, k)) for k in 1:3)..., dim(H), dim(H)) + 1.5I

    # f[1]' and f[1] map some states out of the space
    @test_throws ArgumentError representation(f[1]' + f[1] + 1, H)
    @test representation(f[1]' + f[1] + 1, H; projection = true) ==
        representation(f[1]', H; projection = true) + representation(f[1], H; projection = true) + I

    # A space without states
    @test size(representation(f[1]' * f[2] + 1, constrain_space(hilbert_space(f, 1:2), NumberConservation(5)))) == (0, 0)
    @spin s 1//2
    Hempty = tensor_product(GenericHilbertSpace(:a, [:x, :y]), GenericHilbertSpace(s, Int[]))
    @test size(representation(s[:z] + s[:x] + 1, Hempty)) == (0, 0)

    # A constrained space with several groups, and an identity coefficient that widens the element type
    @boson b
    Hprod = tensor_product(hilbert_space(f, 1:10), hilbert_space(b, 7))
    Hc = tensor_product(hilbert_space(f, 1:10), hilbert_space(b, 7); constraint=NumberConservation(1:15))
    inds = [state_index(state, Hprod) for state in basisstates(Hc)]
    op = sum(f[i]' * f[i+1] + hc for i in 1:9) + sum(f[i]' * b + hc for i in 1:10) + 2 * b' * b + f[1]' + hc + 0.5im
    M = representation(op, Hc; projection = true)
    @test M isa SparseMatrixCSC{ComplexF64}
    @test M == representation(op, Hprod)[inds, inds]
end
