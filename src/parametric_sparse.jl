"""
    ParametricSparse(op, space; projection=false, T)

The sparse matrix of the sum `op` on `space`, together with the contribution of each monomial
of `op` to each stored entry, so that the coefficients can be changed without rebuilding the
matrix. The matrix is the field `matrix`, an ordinary `SparseMatrixCSC{T}`.
`update_coefficients!` rewrites its entries in place for an operator with new
coefficients. The element type `T` defaults to that of `representation(op, space)`; pass
`T=ComplexF64` to allow complex coefficients in later updates of a real operator.

# Examples
```julia
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(2))
ham(t, U) = sum(t * f[n]' * f[n+1] + hc for n in 1:3) + U * f[1]' * f[1] * f[2]' * f[2]
A = ParametricSparse(ham(1.0, 2.0), H)
update_coefficients!(A, ham(0.5, 3.0))
A.matrix ≈ representation(ham(0.5, 3.0), H)
```
"""
struct ParametricSparse{T,K,A}
    matrix::SparseMatrixCSC{T,Int}
    monomials::Dict{K,Int}
    identity_index::Int
    contribution_ptr::Vector{Int}
    contribution_monomial::Vector{Int}
    contribution_amp::Vector{A}
end

function ParametricSparse(op::NCAdd, space::AbstractHilbertSpace; projection=false, T=nothing)
    d = dim(space)
    monomials = Dict(m => i for (i, m) in enumerate(keys(op.dict)))
    entries = [(i, findnz(representation(m, space; projection))...) for (m, i) in monomials]
    if !iszero(op.coeff)
        push!(entries, (length(monomials) + 1, collect(1:d), collect(1:d), ones(Bool, d)))
    end
    rows = reduce(vcat, [e[2] for e in entries])
    cols = reduce(vcat, [e[3] for e in entries])
    amps = reduce(vcat, [e[4] for e in entries])
    inds = reduce(vcat, [fill(e[1], length(e[2])) for e in entries])
    Tm = something(T, promote_type(eltype(amps), mapreduce(typeof, promote_type, values(op.dict)), typeof(op.coeff)))
    matrix, ptr, perm = _union_pattern(Tm, rows, cols, d)
    identity_index = iszero(op.coeff) ? 0 : length(monomials) + 1
    P = ParametricSparse(matrix, monomials, identity_index, ptr, inds[perm], amps[perm])
    return update_coefficients!(P, op)
end

# A matrix with the union pattern of the entries `(rows[k], cols[k])`, the permutation `perm`
# that sorts the entries in the order of the stored values, and the ranges
# `ptr[n]:ptr[n+1]-1` of `perm` that are summed into the stored value `n`.
function _union_pattern(T, rows, cols, d)
    # A counting sort by column, then a sort by row within each short column, is much faster
    # than one sort of all entries.
    colstart = zeros(Int, d + 1)
    colstart[1] = 1
    foreach(c -> colstart[c+1] += 1, cols)
    cumsum!(colstart, colstart)
    next = colstart[1:d]
    perm = similar(rows)
    for (k, c) in enumerate(cols)
        perm[next[c]] = k
        next[c] += 1
    end
    for c in 1:d
        sort!(view(perm, colstart[c]:colstart[c+1]-1); by=k -> rows[k])
    end
    colptr = zeros(Int, d + 1)
    rowval = Int[]
    ptr = Int[]
    for (j, k) in enumerate(perm)
        if j == 1 || rows[k] != rows[perm[j-1]] || cols[k] != cols[perm[j-1]]
            push!(rowval, rows[k])
            push!(ptr, j)
            colptr[cols[k]+1] += 1
        end
    end
    push!(ptr, length(perm) + 1)
    colptr[1] = 1
    cumsum!(colptr, colptr)
    matrix = SparseMatrixCSC(d, d, colptr, rowval, Vector{T}(undef, length(rowval)))
    return matrix, ptr, perm
end

"""
    update_coefficients!(A::ParametricSparse, op)

Set the entries of `A.matrix` to those of `representation(op, space)`, where `op` has
monomials of the operator that `A` was built from, with any coefficients. Monomials of that
operator that are missing in `op` get the coefficient zero. Throws an `ArgumentError` if `op`
has a monomial, or a nonzero scalar term, that the operator `A` was built from does not have.
Returns `A`.
"""
function update_coefficients!(A::ParametricSparse{T}, op::NCAdd) where T
    coeffs = zeros(T, length(A.monomials) + !iszero(A.identity_index))
    for (m, c) in op.dict
        i = get(A.monomials, m, 0)
        iszero(i) && throw(ArgumentError("The monomial $m is not a monomial of the operator the `ParametricSparse` was built from."))
        coeffs[i] = c
    end
    if !iszero(op.coeff)
        iszero(A.identity_index) && throw(ArgumentError("The operator the `ParametricSparse` was built from has no scalar term."))
        coeffs[A.identity_index] = op.coeff
    end
    nzval = nonzeros(A.matrix)
    ptr, monomial, amp = A.contribution_ptr, A.contribution_monomial, A.contribution_amp
    for n in eachindex(nzval)
        v = zero(T)
        for k in ptr[n]:ptr[n+1]-1
            v += coeffs[monomial[k]] * amp[k]
        end
        nzval[n] = v
    end
    return A
end

@testitem "ParametricSparse" begin
    using SparseArrays
    @fermions f
    H = hilbert_space(f, 1:4, NumberConservation(2))
    ham(t, U, μ) = sum(t * f[n]' * f[n+1] + hc for n in 1:3) + U * f[1]' * f[1] * f[2]' * f[2] + μ * f[3]' * f[3] + 0.2
    A = ParametricSparse(ham(1.0, 2.0, 0.5), H; T=ComplexF64)
    @test A.matrix ≈ representation(ham(1.0, 2.0, 0.5), H)
    update_coefficients!(A, ham(0.3im, -1.0, 0.7))
    @test A.matrix ≈ representation(ham(0.3im, -1.0, 0.7), H)
    @test_throws ArgumentError update_coefficients!(A, ham(1.0, 2.0, 0.5) + f[4]' * f[4])
end
