const SPARSE_BLOCK_SIZE = 4096

_terms_matrix(::Type{T}, terms, coeff, space, ::EagerSparseRepr; kwargs...) where T = _blocked_sparse_matrix(T, terms, coeff, space; kwargs...)

# `len` entries from `done` units of work, extrapolated with a margin to `todo` more units
_extrapolated_length(len, done, todo) = len + ceil(Int, len / done * todo * 3 / 2)

# Builds the matrix one block of columns at a time: the entries of a block are collected term by term,
# then sorted into columns, with duplicate entries summed in the order of the terms.
function _blocked_sparse_matrix(::Type{T}, terms, coeff, space; kwargs...) where T
    N = dim(space)
    B = SPARSE_BLOCK_SIZE
    precomps = [(term, precomputation_before_operator_application(term, space)) for term in terms]
    colptr = Vector{Int}(undef, N + 1)
    colptr[1] = 1
    rowval = Int[]
    nzval = T[]
    capacity = 0
    guess = _sparse_length_guess(length(precomps), min(B, N))
    block = (rows, cols, amps) = sparse_matrix_accumulator(T, (guess, guess))
    sortedrows = Int[]
    sortedamps = T[]
    mergebuffers = (zeros(Int, N), Int[], T[])
    colstarts = Vector{Int}(undef, min(B, N) + 1)
    for c0 in 0:B:N-1
        nb = min(B, N - c0)
        foreach(empty!, block)
        for (term, precomp) in precomps
            chunked_operator_indices_and_amplitudes_generic!(block, term, space, precomp, c0+1:c0+nb, c0; kwargs...)
        end
        iszero(coeff) || add_identity!!(block, coeff, space, c0+1:c0+nb, c0)
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
            capacity = _extrapolated_length(total, c0 + nb, N - c0 - nb)
            sizehint!(rowval, capacity)
            sizehint!(nzval, capacity)
        end
        append!(rowval, view(sortedrows, 1:n))
        append!(nzval, view(sortedamps, 1:n))
    end
    foreach(v -> sizehint!(v, length(v)), (rowval, nzval))
    isconcretetype(T) || (nzval = identity.(nzval))
    return SparseMatrixCSC(N, N, colptr, rowval, nzval)
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
    import FermionicHilbertSpaces.SciMLOperators: concretize
    @fermions f
    N = 14
    H = hilbert_space(f, 1:N, NumberConservation(6:7))
    # Diagonal terms and the identity add up in the same entries
    terms = [[(1 + 0.5im) * f[i]' * f[j] for i in 1:N for j in i+1:N]; [f[i]' * f[i] for i in 1:N]]
    op = sum(terms) + 1.5
    M = matrix_representation(op, H)
    entries = map(findnz, [matrix_representation(term, H) for term in terms])
    @test M == sparse((reduce(vcat, getindex.(entries, k)) for k in 1:3)..., dim(H), dim(H)) + 1.5I
    @test concretize(matrix_representation(op, H, :lazy)) == M

    # f[1]' and f[1] map some states out of the space
    @test_throws ArgumentError matrix_representation(f[1]' + f[1] + 1, H)
    @test matrix_representation(f[1]' + f[1] + 1, H; projection=true) ==
          matrix_representation(f[1]', H; projection=true) + matrix_representation(f[1], H; projection=true) + I

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
