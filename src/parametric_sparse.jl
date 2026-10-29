"""
    ParametricSparseMatrix(op, space; projection=false, T)

The sparse matrix of the operator `op` on `space`, together with the contribution of each
monomial of `op` to each stored entry, so that the coefficients can be changed without
rebuilding the matrix. The matrix is the field `matrix`, an ordinary `SparseMatrixCSC{T}`.
`FermionicHilbertSpaces.update_coefficients!` rewrites its entries in place for an operator
with new coefficients. The element type `T` defaults to that of `representation(op, space)`;
pass `T=ComplexF64` to allow complex coefficients in later updates of a real operator.

# Examples
```julia
import FermionicHilbertSpaces: update_coefficients!
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(2))
ham(t, U) = sum(t * f[n]' * f[n+1] + hc for n in 1:3) + U * f[1]' * f[1] * f[2]' * f[2]
A = ParametricSparseMatrix(ham(1.0, 2.0), H)
update_coefficients!(A, ham(0.5, 3.0))
A.matrix ≈ representation(ham(0.5, 3.0), H)
```
"""
struct ParametricSparseMatrix{T,K,A}
    matrix::SparseMatrixCSC{T,Int}
    monomials::Dict{K,Int}
    identity_index::Int
    contribution_ptr::Vector{Int}
    contribution_monomial::Vector{Int32}
    contribution_amp::Vector{A}
end

function ParametricSparseMatrix(_op, space::AbstractHilbertSpace; projection=false, T=mat_eltype(_op))
    op = _as_ncadd(_op)
    bases = group_ids(space)
    all(in(bases), symbolic_groups(op)) || throw(ArgumentError("The symbolic bases of the operator must match the atomic groups of the provided space, got operator groups $(symbolic_groups(op)) and space groups $bases."))
    d = dim(space)
    monomials = Dict(m => i for (i, m) in enumerate(keys(op.dict)))
    rows, cols, amps, ids = Int[], Int[], mat_eltype(op)[], Int32[]
    for (m, i) in monomials
        if length(bases) > 1 && space isa ProductSpace
            # The terms of an unconstrained product space are only represented through kron.
            r, c, v = findnz(representation(m, space; projection))
            append!(rows, r); append!(cols, c); append!(amps, v)
        else
            term = length(bases) > 1 ? partition_product(m, bases, groups(space)) : m
            precomp = precomputation_before_operator_application(term, space)
            chunked_operator_indices_and_amplitudes_generic!((rows, cols, amps), term, space, precomp, 1:d, 0; projection)
        end
        append!(ids, Iterators.repeated(Int32(i), length(rows) - length(ids)))
    end
    identity_index = iszero(op.coeff) ? 0 : length(monomials) + 1
    if !iszero(identity_index)
        append!(rows, 1:d); append!(cols, 1:d); append!(amps, Iterators.repeated(1, d))
        append!(ids, Iterators.repeated(Int32(identity_index), d))
    end
    matrix, ptr, perm = _union_pattern(T, rows, cols, d)
    A = ParametricSparseMatrix(matrix, monomials, identity_index, ptr, ids[perm], amps[perm])
    return update_coefficients!(A, op)
end
_as_ncadd(op::NCAdd) = op
_as_ncadd(op) = NCAdd(op)

# A matrix with the union pattern of the entries `(rows[k], cols[k])`, the permutation `perm`
# that sorts the entries in the order of the stored values, and the ranges
# `ptr[n]:ptr[n+1]-1` of `perm` that are summed into the stored value `n`.
function _union_pattern(T, rows, cols, d)
    # Stable counting sorts by row and then by column are much faster than a comparison sort.
    perm = _counting_sort(cols, d, _counting_sort(rows, d, eachindex(rows)))
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

# `order` stably sorted by `keys[k]`, where the keys lie in `1:d`.
function _counting_sort(keys, d, order)
    next = zeros(Int, d + 1)
    next[1] = 1
    foreach(k -> next[keys[k]+1] += 1, order)
    cumsum!(next, next)
    sorted = Vector{Int}(undef, length(order))
    for k in order
        sorted[next[keys[k]]] = k
        next[keys[k]] += 1
    end
    return sorted
end

"""
    FermionicHilbertSpaces.update_coefficients!(A::ParametricSparseMatrix, op)

Set the entries of `A.matrix` to those of `representation(op, space)`, where `op` has
monomials of the operator that `A` was built from, with any coefficients. Monomials of that
operator that are missing in `op` get the coefficient zero. Throws an `ArgumentError` if `op`
has a monomial, or a nonzero scalar term, that the operator `A` was built from does not have.
Returns `A`.

This is a method of `SciMLOperators.update_coefficients!`, which is not exported. Call it
qualified, or import it from `SciMLOperators`.
"""
update_coefficients!(A::ParametricSparseMatrix, op) = update_coefficients!(A, _as_ncadd(op))
function update_coefficients!(A::ParametricSparseMatrix{T}, op::NCAdd) where T
    coeffs = zeros(T, length(A.monomials) + !iszero(A.identity_index))
    for (m, c) in op.dict
        i = get(A.monomials, m, 0)
        iszero(i) && throw(ArgumentError("The monomial $m is not a monomial of the operator the `ParametricSparseMatrix` was built from."))
        coeffs[i] = c
    end
    if !iszero(op.coeff)
        iszero(A.identity_index) && throw(ArgumentError("The operator the `ParametricSparseMatrix` was built from has no scalar term."))
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

@testitem "ParametricSparseMatrix" begin
    using SparseArrays
    import FermionicHilbertSpaces: update_coefficients!
    @fermions f
    H = hilbert_space(f, 1:4, NumberConservation(2))
    ham(t, U, μ) = sum(t * f[n]' * f[n+1] + hc for n in 1:3) + U * f[1]' * f[1] * f[2]' * f[2] + μ * f[3]' * f[3] + 0.2
    A = ParametricSparseMatrix(ham(1.0, 2.0, 0.5), H; T=ComplexF64)
    @test A.matrix ≈ representation(ham(1.0, 2.0, 0.5), H)
    update_coefficients!(A, ham(0.3im, -1.0, 0.7))
    @test A.matrix ≈ representation(ham(0.3im, -1.0, 0.7), H)
    @test update_coefficients!(A, ham(1.0, 2.0, 0.0)).matrix ≈ representation(ham(1.0, 2.0, 0.0), H)
    @test_throws ArgumentError update_coefficients!(A, ham(1.0, 2.0, 0.5) + f[4]' * f[4])
end
