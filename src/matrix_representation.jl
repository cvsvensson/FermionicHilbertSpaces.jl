abstract type AbstractFermionSym <: AbstractSym end

"""
    EagerDenseRepr{T}()
    EagerDenseRepr()

Internal format, selected by `Matrix{T}` in [`representation`](@ref), that builds a dense `Matrix` with element type `T`.
Without `T`, the element type follows from the operator, see [`mat_eltype`](@ref).
"""
struct EagerDenseRepr{T} end
"""
    EagerSparseRepr{T}()
    EagerSparseRepr()

Internal format, selected by `SparseMatrixCSC{T}` in [`representation`](@ref), that builds a `SparseMatrixCSC` with element type `T`.
This is the default format. Without `T`, the element type follows from the operator, see [`mat_eltype`](@ref).
"""
struct EagerSparseRepr{T} end
EagerSparseRepr() = EagerSparseRepr{Missing}()
EagerDenseRepr() = EagerDenseRepr{Missing}()
"""
    PartitionedSparseRepr(; backend, row_partition=nothing, col_partition=nothing, nparts=4)

Format argument of [`representation`](@ref) that builds a distributed sparse matrix with
PartitionedArrays.jl. `backend` is applied to the part indices, for example `DebugArray`
or `distribute_with_mpi`. Without explicit partitions, rows and columns are split uniformly
into `nparts` parts. Requires `using PartitionedArrays`.
"""
struct PartitionedSparseRepr{B,R,C}
    backend::B
    row_partition::R
    col_partition::C
    nparts::Int
end

PartitionedSparseRepr(; backend=nothing, row_partition=nothing, col_partition=nothing, nparts::Int=4) =
    PartitionedSparseRepr{typeof(backend),typeof(row_partition),typeof(col_partition)}(backend, row_partition, col_partition, nparts)

"""
    LazyRepr(input=missing)

Internal format, selected by `:lazy` in [`representation`](@ref), that gives a matrix-free `SciMLOperators`
operator wrapping a [`LazyOperator`](@ref). `input` is the kind of
vector the operator is prepared for: `:dense`, `:sparse`, or an array whose type and size
are used. With `missing`, a dense vector is used for small spaces and a sparse one otherwise.
"""
struct LazyRepr{T}
    input::T
end
LazyRepr() = LazyRepr(missing)

abstract type AbstractChunkingStrategy end
"""
    NoChunking()

Chunking strategy for the `chunking` keyword of [`representation`](@ref) that builds the
matrix, or applies a lazy operator, serially. This is the default.
"""
struct NoChunking <: AbstractChunkingStrategy end
"""
    TermChunking(scheduler)

Chunking strategy for the `chunking` keyword of [`representation`](@ref) that runs in
parallel with an OhMyThreads.jl `scheduler`. Lazy operators split the work by terms of the
operator. Requires `using OhMyThreads`.
"""
struct TermChunking{S} <: AbstractChunkingStrategy
    scheduler::S
end
"""
    StateChunking(scheduler)

Chunking strategy for the `chunking` keyword of [`representation`](@ref) that runs in
parallel with an OhMyThreads.jl `scheduler`. Lazy operators split the work by basis states.
Requires `using OhMyThreads`.
"""
struct StateChunking{S} <: AbstractChunkingStrategy
    scheduler::S
end

# A factor type that is not concrete, as in a product of different operator types, says nothing about the
# matrix elements, so the factors are promoted one by one.
function mat_eltype(x::NCAdd{C, NCMul{C2, S, F}}) where {C, C2, S, F}
    isconcretetype(S) && return _term_eltype(C, S)
    return _factors_eltype(C, Iterators.flatten(term.factors for term in keys(x.dict)))
end
function mat_eltype(x::NCMul{C, S}) where {C, S}
    isconcretetype(S) && return _term_eltype(C, S)
    return _factors_eltype(C, x.factors)
end
_factors_eltype(C, factors) = mapreduce(f -> _term_eltype(C, typeof(f)), promote_type, factors; init = C)
mat_eltype(::S) where {S} = mat_eltype(S)
"""
    mat_eltype(::Type{O})

Element type of the matrix elements of a symbolic operator of type `O`. Matrix
representations promote it with the coefficients of the expression, after giving its floats
the precision of float coefficients. The fallback is `Float64`, so an operator type with
complex matrix elements must add a method, for example `mat_eltype(::Type{<:MyOp}) = ComplexF64`.
"""
mat_eltype(::Type{S}) where {S} = Float64

_term_eltype(::Type{C}, ::Type{S}) where {C, S} = promote_type(C, _with_precision(mat_eltype(S), _precision(C)))
# The float type of `T`, and `nothing` for a type such as `Int`, which doesn't request a precision
_precision(::Type{P}) where {P <: AbstractFloat} = P
_precision(::Type{Complex{P}}) where {P <: AbstractFloat} = P
_precision(::Type) = nothing
_with_precision(::Type{T}, ::Nothing) where {T} = T
_with_precision(::Type{<:AbstractFloat}, ::Type{P}) where {P} = P
_with_precision(::Type{<:Complex{<:AbstractFloat}}, ::Type{P}) where {P} = Complex{P}
_with_precision(::Type{T}, ::Type) where {T} = T
_amp_at_precision(amp, ::Type{P}) where {P} = convert(_with_precision(typeof(amp), P), amp)

# Gives the coefficients the precision of the element type of the format, so that the amplitudes are computed in it
_at_format_precision(op, repr) = _at_precision(op, _precision(_repr_eltype(Nothing, repr)))
_at_precision(op, ::Nothing) = op
_at_precision(op, ::Type) = op
_at_precision(op::AbstractSym, ::Type{P}) where {P} = NCMul(one(P), [op])
_at_precision(op::NCMul, ::Type{P}) where {P} = NCMul(_coeff_at_precision(op.coeff, P), op.factors)
_at_precision(op::NCAdd, ::Type{P}) where {P} =
    NCAdd(_coeff_at_precision(op.coeff, P), Dict(k => _coeff_at_precision(v, P) for (k, v) in pairs(op.dict)))
_coeff_at_precision(c::Union{Integer, Rational, AbstractFloat}, ::Type{P}) where {P} = P(c)
_coeff_at_precision(c::Complex{<:Union{Integer, Rational, AbstractFloat}}, ::Type{P}) where {P} = Complex{P}(c)
_coeff_at_precision(c, ::Type) = c


function operator_indices_and_amplitudes!(accumulator, op, space; kwargs...)
    precomp = precomputation_before_operator_application(op, _input_space(space))
    return operator_indices_and_amplitudes_generic!(accumulator, op, space, precomp; kwargs...)
end

# Whether `outind` lies outside the space and `projection` allows skipping it. Throws if it lies outside and `projection` is false.
function _skip_outside(outind, projection)
    iszero(outind) || return false
    projection || throw(ArgumentError("The operator maps basis states of this space to states outside it. Pass `projection=true` to drop those matrix elements (correct when the operator should act within this subspace), or use a space that also contains the output states (e.g. one including the neighbouring sectors)."))
    return true
end
# On a transposed group, `newstate` holds the transposed action: an output outside the space is an element
# ⟨state|op|t⟩ with t outside the space, which is not part of the matrix. Only the other groups can leave the space.
_skip_outside(outind, projection, newstate, state, space) =
    _skip_outside(outind, projection || iszero(outind) && !iszero(state_index(_keep_transposed_parts(newstate, state, space), space)))

function operator_indices_and_amplitudes_generic!(accumulator, op, space, precomp; projection)
    Hin, Hout = _input_space(space), _output_space(space)
    for (inind, state) in enumerate(basisstates(Hin))
        newstate, amp = _apply_local_operators(op, state, Hin, precomp)
        if !iszero(amp)
            outind = state_index(newstate, Hout)
            if !_skip_outside(outind, projection, newstate, state, Hout)
                push_inds_amps!(accumulator, outind, inind, amp)
            end
        end
    end
    return accumulator
end
@inline function chunked_operator_indices_and_amplitudes_generic!(accumulator, op, space::AbstractHilbertSpace, precomp, input_inds, col_offset; projection)
    for inind in input_inds
        state = basisstate(inind, space)
        newstate, amp = _apply_local_operators(op, state, space, precomp)
        if !iszero(amp)
            outind = state_index(newstate, space)
            if !_skip_outside(outind, projection, newstate, state, space)
                push_inds_amps!(accumulator, outind, inind - col_offset, amp)
            end
        end
    end
    return accumulator
end
function push_inds_amps!((outinds, ininds, amps), outind, inind, amp)
    push!(outinds, outind)
    push!(ininds, inind)
    push!(amps, amp)
    return nothing
end
function push_inds_amps!(m::AbstractMatrix, outind, inind, amp)
    m[outind, inind] += amp
end


function _apply_local_operators(op, state, space, precomp)
    apply_local_operators(op, state, space, precomp; transpose=false)
end

"""
    apply_local_operators(op::NCMul, state, space, precomps; transpose) -> (newstate, amplitude)

Apply the product `op`, including its coefficient, to the basis state `state` of `space`.
`precomps` holds one value per factor, from [`_precomputation_before_operator_application`](@ref).
With `transpose=true`, apply the transpose of `op` instead. The generic method applies the factors one by one with
[`apply_local_operator`](@ref). A method for products of one operator type can be faster,
as for fermions, where the whole product acts on a Fock state with bit operations.
"""
function apply_local_operators(op::NCMul{C, S}, state, space, precomps; transpose) where {C, S}
    # Accumulate in the element type of the matrix so that `amp` keeps one type through the loop. It is taken from the
    # factor type so that it costs nothing per state; with factors of different types, `amp` is promoted by their amplitudes.
    T = _term_eltype(C, S)
    P = something(_precision(T), Float64)
    amp = convert(T, op.coeff)
    if !transpose
        for (factor, precomp) in Iterators.reverse(zip(op.factors, precomps))
            state, _amp = apply_local_operator(factor, state, space, precomp, P)
            iszero(_amp) && return state, zero(T)
            amp *= _amp_at_precision(_amp, P)
        end
    else
        for (factor, precomp) in zip(op.factors, precomps)
            state, _amp = apply_local_operator(adjoint(factor), state, space, precomp, P)
            iszero(_amp) && return state, zero(T)
            amp *= conj(_amp_at_precision(_amp, P))
        end
    end
    return state, amp
end

struct TupleConcretizer end
struct NoConcretizer end
___concretize(factors, ::TupleConcretizer) = Tuple(factors)
___concretize(factors, ::NoConcretizer) = factors
__concretize(x::NCMul, concr) = NCMul(x.coeff, ___concretize(x.factors, concr))
__concretize(x::NCAdd, concr) = sum(__concretize(term, concr) for term in NCterms(x)) + x.coeff
__concretize(x::ProductOperator, concr) = ProductOperator(map(op -> __concretize(op, concr), x.ops), x.spaces, x.inds)

"""
    partition_product(op::NCMul, bases, spaces)

Split `op` into a `ProductOperator` with one `NCMul` per basis it acts on.
"""
function partition_product(op::NCMul, bases, spaces)
    parts = NonCommutativeProducts.split_by_group(op)
    group_inds = map(((_, part),) -> _basis_index(symbolic_group(first(part.factors)), bases), parts)
    any(iszero, group_inds) && throw(ArgumentError("Not all factors of the operator belong to a basis of the provided space."))
    inds = sort!(unique(group_inds))
    ops = ntuple(k -> _part_on_basis(parts, group_inds, inds[k]), length(inds))
    return ProductOperator(ops, spaces[inds], inds)
end
# A product that isn't sorted can have several runs of factors on one basis. Operators on different bases commute, so
# the runs can be joined.
function _part_on_basis(parts, group_inds, ind)
    runs = [last(part) for (part, i) in zip(parts, group_inds) if i == ind]
    length(runs) == 1 && return only(runs)
    return NCMul(prod(run -> run.coeff, runs), reduce(vcat, [run.factors for run in runs]))
end
_basis_index(basis, bases) = something(findfirst(==(basis), bases), 0)

# Other formats have no identity matrices for the groups without factors
function _matrix_representation(op::NCMul, bases, space::ProductSpace, repr::Union{EagerDenseRepr, EagerSparseRepr}, chunking; kwargs...)
    spaces = groups(space)
    length(spaces) == 1 && return _term_matrix_representation(op, only(spaces), repr, chunking; kwargs...)
    prodop = partition_product(op, bases, spaces)
    matrices = map(enumerate(spaces)) do (n, local_space)
        m = findfirst(==(n), prodop.inds)
        local_op = !isnothing(m) ? prodop.ops[m] : missing
        _term_matrix_representation(local_op, local_space, repr, chunking; kwargs...)
    end
    return _kron_groups(matrices, prodop.inds, repr)
end
# Kronecker product of the group matrices, with the first group as the fastest-varying index.
# Only the groups in `active` have a matrix other than the identity.
_kron_groups(matrices, active, repr) = _kron_groups_generic(matrices, repr)
function _kron_groups_generic(matrices, repr)
    mergedmatrices = _merge_diags(matrices, repr)
    return foldl(kron, Iterators.reverse(mergedmatrices))
end
function _kron_groups(matrices, active, repr::EagerSparseRepr)
    if isempty(active) || !all(n -> matrices[n] isa SparseMatrixCSC, active)
        return _kron_groups_generic(matrices, repr)
    end
    first_active, last_active = extrema(active)
    # Bool identities between the active groups keep the element type and the values of the other factors
    inner = map(first_active:last_active) do n
        n in active ? matrices[n] : sparse(I(size(matrices[n], 1)))
    end
    core = foldl(kron, Iterators.reverse(inner))
    # The element type also includes that of the identity matrices of the inactive groups
    n_identity = findfirst(!in(active), eachindex(matrices))
    T = isnothing(n_identity) ? eltype(core) : promote_type(eltype(core), eltype(matrices[n_identity]))
    r = prod(n -> size(matrices[n], 1), 1:first_active-1; init=1)
    l = prod(n -> size(matrices[n], 1), last_active+1:length(matrices); init=1)
    l == r == 1 && return convert(SparseMatrixCSC{T,Int}, core)
    return _kron_identities(core, l, r, T)
end
# kron(I(l), A, I(r)) for a SparseMatrixCSC `A`, with element type `T`
function _kron_identities(A, l, r, ::Type{T}) where {T}
    m, n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    nzA = nnz(A)
    colptr = Vector{Int}(undef, l * n * r + 1)
    rowval = Vector{Int}(undef, l * nzA * r)
    nzval = Vector{T}(undef, l * nzA * r)
    colptr[1] = 1
    k = 1
    col = 1
    for b in 0:l-1, j in 1:n, q in 1:r
        for p in nzrange(A, j)
            rowval[k] = (b * m + rows[p] - 1) * r + q
            nzval[k] = vals[p]
            k += 1
        end
        col += 1
        colptr[col] = k
    end
    return SparseMatrixCSC(l * m * r, l * n * r, colptr, rowval, nzval)
end
# Bool keeps the element type of the factors it is multiplied with
function _term_matrix_representation(::Missing, space::AbstractHilbertSpace, repr::Union{EagerDenseRepr,EagerSparseRepr}, chunking; kwargs...)
    return Eye{_repr_eltype(Bool, repr)}(dim(space))
end
function _merge_diags(matrices, repr)
    # go through list of matrices and merge consecutive Diagonals with kron
    newmats = Any[]
    n = 1
    while n <= length(matrices)
        if matrices[n] isa Diagonal
            m = 1
            while m + n <= length(matrices) && matrices[n+m] isa Diagonal
                m += 1
            end
            m > 1 ? push!(newmats, kron(matrices[n:n+m-1]...)) : push!(newmats, matrices[n])
            n += m
        else
            push!(newmats, matrices[n])
            n += 1
        end
    end
    return newmats
end

function _matrix_representation(op::NCMul, bases, space, repr, chunking; kwargs...)
    if isempty(op.factors)
        return _convert_repr_eltype(op.coeff * I(dim(space)), repr)
    else
        newop = length(bases) > 1 ? partition_product(op, bases, groups(space)) : op
        return _term_matrix_representation(newop, space, repr, chunking; kwargs...)
    end
end
function _matrix_representation(op::NCAdd, bases, space, repr, chunking; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space(op, space, repr, chunking; kwargs...)
    end
    __matrix_representation(op, bases, space, repr, chunking; kwargs...)
end
function __matrix_representation(op::NCAdd, bases, space::ProductSpace, repr::Union{EagerDenseRepr, EagerSparseRepr}, chunking; tree_split = 10, kwargs...)
    return _sum_terms_matrix(_product_terms(op, bases, space, repr; tree_split, kwargs...), op.coeff, space, repr, chunking; kwargs...)
end
# The terms of `op` on the groups of `space`, as `KronTerm`s.
# Terms acting on the same range of groups are summed on those groups, so each sum is embedded with identities once
# instead of once per term. Terms on a single group join those on it and a neighbour, if any, since their sum is still
# small. Joining larger ranges can make the summed matrices much larger than the terms. Terms acting on all groups are
# not summed, as that would build full-size matrices, but those that differ only on their first group are one term
# with the sum of their factors on that group. The diagonals of the sums are added into one term, since the entries
# that several sums share are mostly on the diagonal.
function _product_terms(op::NCAdd, bases, space::ProductSpace, repr::Union{EagerDenseRepr, EagerSparseRepr}; tree_split = 10, kwargs...)
    spaces = groups(space)
    by_span = Dict{UnitRange{Int}, Vector{Any}}()
    for term in NCterms(op)
        prodop = partition_product(term, bases, spaces)
        push!(get!(() -> [], by_span, minimum(prodop.inds):maximum(prodop.inds)), prodop)
    end
    for n in eachindex(spaces), span in (n:(n + 1), (n - 1):n)
        if span != eachindex(spaces) && haskey(by_span, n:n) && haskey(by_span, span)
            append!(by_span[span], pop!(by_span, n:n))
        end
    end
    products = OrderedDict{Any, Vector{Any}}()
    for p in get(by_span, eachindex(spaces), [])
        push!(get!(() -> [], products, (p.inds, map(op -> op.factors, Base.tail(p.ops)))), p)
    end
    sums = [(convert(SparseMatrixCSC, _sum_matrices([_span_matrix(p, span, spaces, repr; kwargs...) for p in prodops], repr; tree_split)), _stride(spaces, first(span))) for (span, prodops) in by_span if span != eachindex(spaces)]
    return [_split_diagonals(sums, dim(space)); [_kron_term(ps, spaces, repr; kwargs...) for ps in values(products)]]
end
function _span_matrix(prodop, span, spaces, repr; kwargs...)
    matrices = map(span) do n
        m = findfirst(==(n), prodop.inds)
        _term_matrix_representation(isnothing(m) ? missing : prodop.ops[m], spaces[n], repr, NoChunking(); kwargs...)
    end
    return _kron_groups(matrices, prodop.inds .- (first(span) - 1), repr)
end
_stride(spaces, n) = prod(m -> dim(spaces[m]), 1:(n - 1); init = 1)
# The sum of the products `prodops`, which differ only in their factor on the first group
function _kron_term(prodops, spaces, repr; kwargs...)
    matrix(op, n) = _term_matrix_representation(op, spaces[n], repr, NoChunking(); kwargs...)
    p = first(prodops)
    inds = Tuple(p.inds)
    A1 = sum(q -> prod(op -> op.coeff, Base.tail(q.ops); init = true) * matrix(first(q.ops), first(inds)), prodops)
    rest = map((op, n) -> matrix(NCMul(one(op.coeff), op.factors), n), Base.tail(p.ops), Base.tail(inds))
    factors = map((A, n) -> (convert(SparseMatrixCSC, A), _stride(spaces, n)), (A1, rest...), inds)
    T = mapreduce(eltype ∘ first, promote_type, factors)
    return KronTerm(map(((A, r),) -> (convert(SparseMatrixCSC{T, Int}, A), r), factors))
end
# The terms of the off-diagonal parts of the pairs `(A, r)` as in `KronTerm`, and one term of the sum of their diagonals
function _split_diagonals(sums, N)
    isempty(sums) && return []
    diagonal = zeros(mapreduce(eltype ∘ first, promote_type, sums), N)
    for (A, r) in sums
        reshape(diagonal, r, size(A, 1), :) .+= transpose(Vector(diag(A)))
    end
    return [[KronTerm(((A - Diagonal(diag(A)), r),)) for (A, r) in sums]; KronTerm(((sparse(Diagonal(diagonal)), 1),))]
end
# The term kron(…, A_2, I, A_1, I) of a sum, with identities on the groups that no factor acts on. Each pair `(A, r)` in
# the tuple `factors` is a square matrix on one group or a range of groups, and the product `r` of the dimensions of the
# groups before them.
struct KronTerm{F <: Tuple}
    factors::F
end
mat_eltype(term::KronTerm) = eltype(first(first(term.factors)))
precomputation_before_operator_application(::KronTerm, ::AbstractHilbertSpace) = nothing
function chunked_operator_indices_and_amplitudes_generic!(accumulator, term::KronTerm, space::AbstractHilbertSpace, precomp, input_inds, col_offset; kwargs...)
    return _foldl_columns(accumulator, term.factors, input_inds) do accumulator, j, digits
        _push_kron_entries!(accumulator, term.factors, digits, j - 1, true, j - col_offset)
    end
end
_nentries(term::KronTerm, incols) =
    _foldl_columns((n, _, digits) -> n + prod(map(((A, _), c) -> length(nzrange(A, c + 1)), term.factors, digits)), 0, term.factors, incols)
_total_nentries(term::KronTerm, N) = prod(map(((A, _),) -> nnz(A), term.factors)) * div(N, prod(map(((A, _),) -> size(A, 2), term.factors)))
# `acc = f(acc, j, digits)` for the columns `j` in `incols`, where column `j` of the term draws on column
# `digits[k] + 1` of factor k. The digits are counted up instead of divided out of `j`, as divisions dominate the loop.
function _foldl_columns(f, acc, factors, incols)
    counters = map(((A, r),) -> (rem(first(incols) - 1, r), rem(div(first(incols) - 1, r), size(A, 2))), factors)
    for j in incols
        acc = f(acc, j, map(last, counters))
        counters = map(factors, counters) do (A, r), (q, c)
            q + 1 < r && return (q + 1, c)
            return (0, c + 1 == size(A, 2) ? 0 : c + 1)
        end
    end
    return acc
end
_push_kron_entries!(accumulator, ::Tuple{}, ::Tuple{}, row, amp, col) = (push_inds_amps!(accumulator, row + 1, col, amp); accumulator)
function _push_kron_entries!(accumulator, factors, digits, row, amp, col)
    (A, r), c = first(factors), first(digits)
    rows, vals = rowvals(A), nonzeros(A)
    for p in nzrange(A, c + 1)
        _push_kron_entries!(accumulator, Base.tail(factors), Base.tail(digits), row + (rows[p] - 1 - c) * r, amp * vals[p], col)
    end
    return accumulator
end
operator_indices_and_amplitudes_generic!(accumulator, term::KronTerm, space, precomp; kwargs...) =
    chunked_operator_indices_and_amplitudes_generic!(accumulator, term, space, precomp, 1:dim(space), 0; kwargs...)
# On spaces with a basis-state loop (for example a constrained space over a ProductSpace) and for formats with no method of its own,
# `_terms_matrix` builds the sum of all terms.
function __matrix_representation(op::NCAdd, bases, space, repr, chunking; tree_split = 10, kwargs...)
    prodops = _product_terms(op, bases, space, repr)
    T = _repr_eltype(promote_type(mapreduce(mat_eltype, promote_type, prodops), typeof(op.coeff)), repr)
    return _terms_matrix(T, prodops, op.coeff, space, repr, chunking; kwargs...)
end
_product_terms(op::NCAdd, bases, space, repr; _...) = [partition_product(term, bases, groups(_input_space(space))) for term in NCterms(op)]
# The terms are summed in the element type of their matrices, and the identity term is added in that element type
# promoted with the type of `coeff`.
function _sum_terms_matrix(terms, coeff, space, repr, chunking; kwargs...)
    Tterms = _repr_eltype(mapreduce(mat_eltype, promote_type, terms), repr)
    T = _repr_eltype(promote_type(Tterms, typeof(coeff)), repr)
    Tterms == T && return _terms_matrix(Tterms, terms, coeff, space, repr, chunking; kwargs...)
    summed = _terms_matrix(Tterms, terms, zero(coeff), space, repr, chunking; kwargs...)
    iszero(coeff) && return convert(AbstractMatrix{T}, summed)
    return summed + _terms_matrix(T, (), coeff, space, repr, NoChunking(); kwargs...)
end

function _matrix_representation(op, bases, space, repr, chunking; kwargs...) #Assume op is a single symbolic operator
    _matrix_representation(NCMul(1, [op]), bases, space, repr, chunking; kwargs...)
end


matrix_accumulator(op::ProductOperator, space, repr) = matrix_accumulator(_repr_eltype(mat_eltype(op), repr), 1, _matrix_size(space), repr)
matrix_accumulator(op, space, repr) = matrix_accumulator(_repr_eltype(mat_eltype(op), repr), length(NCterms(op)), _matrix_size(space), repr)
_matrix_size(space) = (dim(_output_space(space)), dim(_input_space(space)))

# The element type given in the format (e.g. `EagerSparseRepr{ComplexF64}()`) takes precedence over the operator's
_repr_eltype(default, ::Union{EagerSparseRepr{T},EagerDenseRepr{T}}) where T = T === Missing ? default : T
_repr_eltype(default, _) = default

function matrix_accumulator(::Type{T}, N::Int, (d1, d2)::Tuple{Int,Int}, ::EagerSparseRepr) where T
    return sparse_matrix_accumulator(T, _sparse_length_guess.(N, (d1, d2)))
end
# Initial number of entries reserved for N terms on d states, mildly increasing with the number of terms
_sparse_length_guess(N, d) = ndigits(N + 1; base=2) * d

function sparse_matrix_accumulator(::Type{T}, (N, M)) where T
    outinds = Int[]
    ininds = Int[]
    amps = T[]
    sizehint!(outinds, N)
    sizehint!(ininds, M)
    sizehint!(amps, N)
    return (outinds, ininds, amps)
end
# Adds `coeff` at the diagonal entries of the columns `cols`, stored at the column indices `cols .- col_offset`
function add_identity!!((outinds, ininds, amps), coeff, space, cols=1:dim(space), col_offset=0)
    append!(outinds, cols)
    append!(ininds, cols .- col_offset)
    append!(amps, Fill(coeff, length(cols)))
end
finalize!(accumulator, space) = finalize!(accumulator, dim(_output_space(space)), dim(_input_space(space)))
function finalize!((outinds, ininds, amps), N, M)
    L = length(outinds)
    # `sparse!` throws a BoundsError for zero rows
    iszero(N) && return spzeros(eltype(amps), N, M)
    isconcretetype(eltype(amps)) && return _trimmed!(SparseArrays.sparse!(outinds, ininds, amps, N, M), L)
    return _trimmed!(SparseArrays.sparse!(outinds, ininds, identity.(amps), N, M), L)
end
# `sparse!` stores the matrix in the vectors of its L input entries. Releases their unused room unless L is small.
function _trimmed!(S, L)
    L > 2^12 && foreach(v -> sizehint!(v, length(v)), (SparseArrays.getcolptr(S), rowvals(S), nonzeros(S)))
    return S
end

function matrix_accumulator(::Type{T}, N::Int, (d1, d2), ::EagerDenseRepr) where T
    zeros(T, d1, d2)
end
function add_identity!!(m::Matrix{T}, coeff, space::AbstractHilbertSpace; _...) where T
    m .+= coeff * I(size(m, 1))
end
function add_identity!!(accumulator, coeff, ax::Axes; projection)
    Hout = _output_space(ax)
    for (inind, state) in enumerate(basisstates(_input_space(ax)))
        outind = state_index(state, Hout)
        _skip_outside(outind, projection) || push_inds_amps!(accumulator, outind, inind, coeff)
    end
    return accumulator
end
function finalize!(m::Matrix, space)
    return m
end


function _matrix_representation_single_space(op::NCAdd, space, repr, chunking; kwargs...)
    _terms_matrix(_repr_eltype(mat_eltype(op), repr), NCterms(op), op.coeff, space, repr, chunking; kwargs...)
end

# The matrix of `sum(terms) + coeff * I` with element type `T`. A new format defines a method of this and of
# `_canonicalize_repr`, and of `_convert_repr_eltype` for operators that are a multiple of the identity.
_terms_matrix(::Type{T}, terms, coeff, space, repr::EagerDenseRepr, ::NoChunking; kwargs...) where {T} = _accumulated_matrix(T, terms, coeff, space, repr; kwargs...)
_terms_matrix(::Type{T}, terms, coeff, space::Axes, repr::EagerSparseRepr, ::NoChunking; kwargs...) where {T} = _accumulated_matrix(T, terms, coeff, space, repr; kwargs...)

function _accumulated_matrix(::Type{T}, terms, coeff, space, repr; kwargs...) where T
    accumulator = matrix_accumulator(T, length(terms), _matrix_size(space), repr)
    for term in terms
        operator_indices_and_amplitudes!(accumulator, term, space; kwargs...)
    end
    iszero(coeff) || add_identity!!(accumulator, coeff, space; kwargs...)
    return finalize!(accumulator, space)
end

_term_matrix_representation(op, H, repr, chunking; kwargs...) = _terms_matrix(_repr_eltype(mat_eltype(op), repr), (op,), 0, H, repr, chunking; kwargs...)
function _term_matrix_representation(op, H, repr::Union{EagerSparseRepr, EagerDenseRepr}, chunking; kwargs...)
    _accumulator = matrix_accumulator(op, H, repr)
    accumulator = operator_indices_and_amplitudes!(_accumulator, op, H; kwargs...)
    finalize!(accumulator, H)
end

# function apply_local_operator(op::NCMul{C,F}, state::AbstractFockState, space::AbstractHilbertSpace, (pos, daggers)) where {C,F<:AbstractFermionSym}
#     new_state, amp = togglefermions(Iterators.reverse(pos), Iterators.reverse(daggers), state)
#     return new_state, amp
# end


## Automatic basis inference from operator
"""
    symbolic_groups(op)

Extract all unique symbolic bases from an operator expression.
Returns a set of unique bases found in the operator.
"""
function symbolic_groups(op)
    combine(x, y) = nothing
    reducer = (combine, combine)
    accum = Set()
    f(op) = push!(accum, symbolic_group(op))
    NonCommutativeProducts.ncmapreduce(f, reducer, op)
    return accum
end

"""
    representation(op_or_state, space::AbstractHilbertSpace, [repr]; kwargs...)
    representation(op, Axes(Hout, Hin), [repr]; kwargs...)
    representation(state_expression; kwargs...)

Return a concrete representation of `op_or_state` in Hilbert space `space`. This is the main entry point, which calls
[`matrix_representation`](@ref) for operators and [`vector_representation`](@ref) for states.

- For symbolic operators and numbers, returns a sparse matrix by default. The optional `repr` selects the format:
  an array type such as `Matrix` or `SparseMatrixCSC`, with an optional element type as in `Matrix{ComplexF64}`,
  or a number type such as `Float64` for a sparse result of that element type. `:lazy` gives a matrix-free
  `SciMLOperators.FunctionOperator` that wraps a `LazyOperator`, and `:sparse` also works.
  A number `c` gives `c` times the identity. With a GPU package loaded, a GPU array type such as `CuArray` or
  `cuSPARSE.CuSparseMatrixCSR{Float32}` builds the matrix on the GPU.
- For `AbstractBasisState` and strings such as `"10"`, returns a one-hot sparse column vector.
- For `SymbolicState` kets and bras, and sums of them, returns the corresponding column or row vector. Their array types
  are `Vector` and `SparseVector`. `Array` gives a dense result for both states and operators.
- For ketbras, and operators that contain ketbras, returns a matrix.

Coefficients are converted to a format's element type, so a complex coefficient with element type `Float64` throws an `InexactError`.
Matrix elements are computed in the precision of a float element type. Without an element type, `Float32` coefficients give `Float32` matrix elements.

Expressions that combine kets or bras with operators or with states of other spaces have no representation and throw an
`ArgumentError`. Without the `space` argument, only expressions of symbolic states are accepted, and the space is taken from the states.
The single-particle space `hilbert_space(f, labels, FreeFermions.SingleParticle())` and the Bogoliubov-de Gennes space `hilbert_space(f, labels, FreeFermions.BdG())`
represent quadratic fermionic operators only, and a number is represented by the zero matrix.

With `Axes(Hout, Hin)`, `op` is represented as a map from `Hin` to `Hout`.

```jldoctest
julia> @fermions f;

julia> representation(f[1]' * f[1], hilbert_space(f, 1:1)) == [0 0; 0 1]
true
```
"""
representation

"""
    matrix_representation(op, space::AbstractHilbertSpace, [repr]; kwargs...)
    matrix_representation(op, Axes(Hout, Hin), [repr]; kwargs...)

Return the matrix representation of symbolic operator `op` in Hilbert space `space`. This is the operator-specific form of [`representation`](@ref).

With [`Axes`](@ref)`(Hout, Hin)`, return the `dim(Hout) × dim(Hin)` matrix of `op` from `Hin` to `Hout`. The two
spaces must have the same groups, for example two sectors of the same modes. Two different spaces support only eager
formats without chunking.

The optional `repr` selects the format, as in [`representation`](@ref).

Keyword arguments include `projection` (default `false`) which if true will ignore the error thrown when the operator maps outside the space (outside `Hout` with `Axes`). Projection drops matrix elements to states outside the space, so an operator that changes a conserved quantity gives zero on a single sector; use `Axes(Hout, Hin)` with the two sectors instead. Use `chunking` to control NCAdd construction strategy: `NoChunking()`, `TermChunking(scheduler)`, or `StateChunking(scheduler)`.

# Examples
```julia
@fermions f
H = hilbert_space(f, 1:2)
op = f[1]' * f[2] + hc
M = FermionicHilbertSpaces.matrix_representation(op, H)
size(M) == (dim(H), dim(H))
```
"""
function matrix_representation(op, space::AbstractHilbertSpace, repr=EagerSparseRepr(); projection=false, chunking=NoChunking(), kwargs...)
    repr === :dense && return Matrix(matrix_representation(op, space, EagerSparseRepr(); projection, chunking, kwargs...))
    # The barrier keeps inference from compiling every format a runtime Symbol could select
    repr isa Symbol && return Base.inferencebarrier(matrix_representation)(op, space, _canonicalize_repr(repr); projection, chunking, kwargs...)
    format = _matrix_format(repr)
    if trivial_operator(op)
        return _convert_repr_eltype(get_trivial_op_coeff(op) * I(dim(space)), repr)
    end
    return _with_format_eltype(_matrix_representation(_at_format_precision(op, format), _checked_group_ids(op, space), space, format, chunking; projection, kwargs...), format)
end
function matrix_representation(op, ax::Axes, repr=EagerSparseRepr(); projection=false, chunking=NoChunking(), kwargs...)
    Hout, Hin = _operator_axes(ax)
    Hout == Hin && return matrix_representation(op, Hin, repr; projection, chunking, kwargs...)
    repr === :dense && return Matrix(matrix_representation(op, ax, EagerSparseRepr(); projection, chunking, kwargs...))
    repr = _matrix_format(repr)
    repr isa Union{EagerSparseRepr,EagerDenseRepr} && chunking isa NoChunking || throw(ArgumentError("Lazy and chunked representations with `Axes(Hout, Hin)` need `Hout == Hin`."))
    if trivial_operator(op)
        coeff = get_trivial_op_coeff(op)
        return _accumulated_matrix(_repr_eltype(typeof(coeff), repr), (), coeff, ax, repr; projection)
    end
    return _with_format_eltype(_matrix_representation(_at_format_precision(op, repr), _checked_group_ids(op, Hin), ax, repr, chunking; projection, kwargs...), repr)
end
function _checked_group_ids(op, space)
    op_groups = symbolic_groups(op)
    space_groups = group_ids(space)
    all(in(space_groups), op_groups) || throw(ArgumentError("The symbolic bases of the operator must match the atomic groups of the provided space, got operator groups $op_groups and space groups $space_groups."))
    return space_groups
end

"""
    expectation(op, state, H)

Return the expectation value of the operator `op` in `state` on the space `H`. A vector `ψ` gives `ψ' * M * ψ` and a
density matrix `ρ` gives `tr(ρ * M)`, where `M = representation(op, H)`. The state is assumed to be normalized.

On `hilbert_space(f, labels, FreeFermions.SingleParticle())`, `state` is the correlation matrix `G[i, j] = ⟨f_i' f_j⟩`, and on
`hilbert_space(f, labels, FreeFermions.BdG())` it is the Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩`, with `Ψ = (f; f')` in
the order of the BdG space. On the Majorana space `hilbert_space(γ, labels, FreeFermions.SingleParticle())` it is
`G[i, j] = ⟨γ_i γ_j⟩ / 2`. The state is then the Gaussian state with that correlation matrix, and `op` of any degree
is evaluated with Wick's theorem. Terms that do not conserve the particle number (single-particle space) or the parity
(BdG and Majorana spaces) throw an error, unless `projection=true`, which drops them.

# Examples
```julia
@fermions f
H = hilbert_space(f, 1:2)
ψ = representation("10", H)
expectation(f[1]' * f[1] + 2, ψ, H) == 3
```
"""
expectation(op, ψ::AbstractVector, H::AbstractHilbertSpace) = dot(ψ, representation(op, H), ψ)
expectation(op, ρ::AbstractMatrix, H::AbstractHilbertSpace) = dot(ρ', representation(op, H))

@testitem "Expectation values" begin
    using LinearAlgebra, SparseArrays
    @fermions f
    H = hilbert_space(f, 1:4, NumberConservation(2))
    op = f[1]' * f[2] * f[3]' * f[4] + 1im * f[2]' * f[3] + hc + 2
    M = representation(op, H, Matrix)
    ψ = normalize(rand(ComplexF64, dim(H)))
    @test expectation(op, ψ, H) ≈ ψ' * M * ψ
    ρ = rand(ComplexF64, dim(H), dim(H))
    @test expectation(op, ρ, H) ≈ tr(ρ * M)
    ρs = sprand(ComplexF64, dim(H), dim(H), 0.3)
    @test expectation(op, ρs, H) ≈ tr(ρs * M)
end

@testitem "Expectation values with symbolic coefficients" begin
    using Symbolics
    @variables t
    @fermions f
    H = hilbert_space(f, 1:2)
    ψ = representation("10", H)
    @test isequal(Symbolics.simplify(expectation(t * f[1]' * f[1] + f[1]' * f[2] + 2, ψ, H)), t + 2)
end
_canonicalize_repr(repr::Union{EagerSparseRepr, EagerDenseRepr, LazyRepr, PartitionedSparseRepr}) = repr
_canonicalize_repr(::Union{Type{Array}, Type{Matrix}, Type{Vector}}) = EagerDenseRepr()
_canonicalize_repr(::Union{Type{Array{T}}, Type{Matrix{T}}, Type{Vector{T}}}) where {T} = EagerDenseRepr{T}()
_canonicalize_repr(::Union{Type{SparseArrays.SparseMatrixCSC}, Type{SparseArrays.SparseVector}}) = EagerSparseRepr()
_canonicalize_repr(::Union{Type{SparseArrays.SparseMatrixCSC{T}}, Type{SparseArrays.SparseMatrixCSC{T, Int}}, Type{SparseArrays.SparseVector{T}}, Type{SparseArrays.SparseVector{T, Int}}}) where {T} = EagerSparseRepr{T}()
_canonicalize_repr(::Type{T}) where T<:Number = EagerSparseRepr{T}()
_canonicalize_repr(repr) = throw(ArgumentError("Unknown representation format `$(repr)`. Use `:sparse`, `:lazy`, a number type, or `Array`, `Matrix`, `Vector`, `SparseMatrixCSC` or `SparseVector`, optionally with an element type."))
_matrix_format(repr) = _canonicalize_repr(repr)
_matrix_format(repr::Type{<:Union{Vector, SparseArrays.SparseVector}}) = throw(ArgumentError("`$(repr)` is a vector format, but this representation is a matrix. Use `Matrix`, `SparseMatrixCSC` or `Array`."))
_vector_format(repr) = _canonicalize_repr(repr)
_vector_format(repr::Type{<:Union{Matrix, SparseArrays.SparseMatrixCSC}}) = throw(ArgumentError("`$(repr)` is a matrix format, but kets and bras give vectors. Use `Vector`, `SparseVector` or `Array`."))
function _canonicalize_repr(s::Symbol)
    s == :sparse && return EagerSparseRepr()
    s == :dense && return EagerDenseRepr()
    s == :lazy && return LazyRepr()
    throw(ArgumentError("The representation type must be `:sparse`, `:dense` or `:lazy`, got `:$s`."))
end


group_ids(space::AbstractHilbertSpace) = map(group_id, groups(space))

# The default sparse format keeps the structured identity; the other formats give the matrix type they name
_convert_repr_eltype(M, ::EagerSparseRepr{Missing}) = M
_convert_repr_eltype(M, ::EagerDenseRepr{Missing}) = Matrix(M)
_convert_repr_eltype(M, ::EagerSparseRepr{T}) where T = SparseArrays.sparse(T.(M))
_convert_repr_eltype(M, ::EagerDenseRepr{T}) where T = Matrix{T}(M)
_convert_repr_eltype(M, ::Union{LazyRepr, PartitionedSparseRepr}) = M
_convert_repr_eltype(M, repr::Type) = _convert_repr_eltype(M, _canonicalize_repr(repr))
_convert_repr_eltype(M, ::Type{A}) where {A <: SparseArrays.AbstractSparseArray} = SparseArrays.sparse(_convert_repr_eltype(M, _canonicalize_repr(A)))

# A concrete element type fixes the matrix type, also where inference cannot follow the assembly
_with_format_eltype(M, ::Any) = M
_with_format_eltype(M, ::EagerDenseRepr{T}) where {T <: Number} = isconcretetype(T) ? convert(Matrix{T}, M)::Matrix{T} : M
_with_format_eltype(M, ::EagerSparseRepr{T}) where {T <: Number} = isconcretetype(T) ? convert(SparseArrays.SparseMatrixCSC{T, Int}, M)::SparseArrays.SparseMatrixCSC{T, Int} : M

trivial_operator(op::Union{UniformScaling,Number}) = true
trivial_operator(op::NCMul) = length(op.factors) == 0
trivial_operator(op::NCAdd) = length(op.dict) == 0
trivial_operator(op) = false
get_trivial_op_coeff(op::NCMul) = op.coeff
get_trivial_op_coeff(op::NCAdd) = op.coeff
get_trivial_op_coeff(op::UniformScaling) = op.λ
get_trivial_op_coeff(op) = op

@testitem "Multi-space matrix representation" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: SpinSpace, SpinState

    # Test 1: Fermion ⊗ Spin system
    @fermions f
    @spin s

    Nf = 2
    Hf = hilbert_space(f, 1:Nf)
    Hs = SpinSpace(s, 1 // 2)
    H = tensor_product(Hf, Hs)

    # Test simple product operator: f[1]' * f[1] ⊗ S_z
    op = f[1]' * f[1] * s[:z]
    M = representation(op, H)
    @test M == kron(representation(s[:z], Hs), representation(f[1]' * f[1], Hf))

    # Verify dimensions: (2^2) × 2 = 8×8
    @test size(M) == (8, 8)

    # Check that the matrix is sparse
    @test M isa SparseMatrixCSC

    # Test 2: Sum of mixed operators
    ops = [f[1]' * f[1] * s[:z], f[2]' * f[2] * s[:+], 2im]
    op_mixed = sum(ops)
    M_mixed = representation(op_mixed, H)
    @test size(M_mixed) == (8, 8)
    @test M_mixed == sum(representation(op, H) for op in ops)


    # Test 3: Verify the result is hermitian for hermitian operators
    op_herm = f[1]' * f[2] * s[:z] + hc
    M_herm = representation(op_herm, H)
    @test M_herm ≈ M_herm'  # Should be hermitian
    @test M_herm == representation(op_herm', H)
end

@testitem "Three-space product" begin
    using SparseArrays, LinearAlgebra

    @fermions f
    @fermions g  # Different fermion species, commuting with f
    @spin s

    Hf = hilbert_space(f[1])
    Hg = hilbert_space(g[1])
    Hs = hilbert_space(s, 1 // 2)
    H = tensor_product(Hf, Hg, Hs)
    # Operator acting on all three spaces
    op = f[1]' * f[1] * g[1]' * g[1] * s[:z]
    M = representation(op, H)
    @test M == kron(reverse([representation(f[1]' * f[1], Hf), representation(g[1]' * g[1], Hg), representation(s[:z], Hs)])...)

    # Should have dimension 2 × 2 × 2 = 8
    @test size(M) == (8, 8)
    @test M isa SparseMatrixCSC

    @test_throws ArgumentError representation(op, Hs)
    @test_throws ArgumentError representation(op, Hf)
    @test_throws ArgumentError representation(op, Hg)
end


@testitem "Constrained product space agrees with the unconstrained one" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: EagerDenseRepr
    @fermions f
    @spin s 1 // 2
    @bosons b
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(s), hilbert_space(b, 1:2, 3))
    states = collect(basisstates(H))[1:2:end]
    inds = [state_index(state, H) for state in states]
    Hc = constrain_space(H, states)
    ops = [f[1]' * f[2] * s[:+] + hc + 2 * b[1]' * b[2] + hc + 1,
        0.5 * f[1]' * f[1] * b[1]' * b[1] + s[:z] * b[2]' * b[2],
        1im * f[1]' * s[:-] * b[2] + hc + 2.5,
        f[1]' * f[2] + f[2]' * f[1],
        sum(0.3n * (b[1]' + b[1]) * f[n]' * f[n] for n in 1:2) + 1im * s[:x] * f[1]' * f[2] * b[2],
    ]
    for op in ops, repr in (:sparse, :dense, EagerDenseRepr())
        M = representation(op, H, repr)
        Mc = representation(op, Hc, repr; projection=true)
        @test typeof(Mc) == typeof(M)
        @test Mc ≈ M[inds, inds]
    end
    # Integer terms and an integer identity term keep their element type.
    op_int = 2 * f[1]' * f[2] * b[1]' + hc
    @test typeof(representation(op_int, Hc; projection=true)) == typeof(representation(op_int, H))
    @test typeof(representation(op_int + 1, Hc; projection=true)) == typeof(representation(op_int + 1, H))
    @test representation(op_int + 1, Hc; projection=true) ≈ representation(op_int + 1, H)[inds, inds]
end

@testitem "Requested element type on a constrained product space" begin
    using SparseArrays
    using FermionicHilbertSpaces: EagerSparseRepr, EagerDenseRepr
    @fermions f
    @boson b
    H = tensor_product(hilbert_space(f, 1:3), hilbert_space(b, 3); constraint=NumberConservation(1:2))
    op = f[1]' * f[2] + b' * f[1] + 1
    for (Repr, T) in ((EagerSparseRepr, ComplexF32), (EagerDenseRepr, Float32))
        M = representation(op, H, Repr{T}())
        @test eltype(M) == T
        @test M ≈ representation(op, H)
    end
end

@testitem "Requested element type on a product space" begin
    using FermionicHilbertSpaces: EagerSparseRepr, EagerDenseRepr
    @fermions f
    @spin s 1 // 2
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(s))
    reprs = ((Float32, Float32), (Matrix{ComplexF32}, ComplexF32), (EagerSparseRepr{Float32}(), Float32), (EagerDenseRepr{ComplexF32}(), ComplexF32))
    for op in (f[1]' * f[2], s[:z] + 1.0, 2.0 * f[1]' * f[2] * s[:z]), (repr, T) in reprs
        M = representation(op, H, repr)
        @test eltype(M) == T
        @test M == representation(op, H, Matrix)
    end
    @test eltype(representation(f[1]' * f[2] + 1, H)) == Int
end

@testitem "Matrix elements are computed in the requested precision" begin
    @fermions f
    @boson b
    @spin s 1 // 2
    @spin t
    Hb = hilbert_space(b, 3)
    Hs = hilbert_space(s)
    Hmixed = tensor_product(hilbert_space(f, 1:1), Hb)

    @test representation(b', Hb, Matrix{BigFloat})[3, 2] == sqrt(big(2))
    @test representation(0.5 * b', Hb, Matrix{BigFloat})[3, 2] == sqrt(big(2)) / 2
    @test representation(b' + 1, Hb, Matrix{BigFloat})[3, 2] == sqrt(big(2))
    @test representation(f[1]' * b', Hmixed, Matrix{BigFloat})[6, 3] == sqrt(big(2))
    @test representation(t[:+], hilbert_space(t, 3 // 2), Matrix{BigFloat})[2, 1] == sqrt(big(3))

    for op in (b' * b + b, 0.5 * b' + 1), T in (Float32, ComplexF32)
        M = @inferred representation(op, Hb, Matrix{T})
        @test M isa Matrix{T}
        @test M ≈ representation(op, Hb, Matrix)
    end
    @test representation(1 // 3 * s[:z], Hs, Matrix{Float32}) ≈ representation(1 // 3 * s[:z], Hs, Matrix)
    @test eltype(representation(0.5f0 * b', Hb)) == Float32
    @test eltype(representation(0.5f0 * s[:z] + 1.0f0im * s[:+], Hs)) == ComplexF32
    @test eltype(representation(0.5 * b', Hb)) == Float64
    @test eltype(representation(b', Hb)) == Float64
end

@testitem "Amplitudes of custom operators in the requested precision" begin
    import FermionicHilbertSpaces as FHS
    struct RootOp <: FHS.AbstractSym end
    struct PreciseRootOp <: FHS.AbstractSym end
    Base.adjoint(op::Union{RootOp, PreciseRootOp}) = op
    FHS.apply_local_operator(::RootOp, state, space, precomp) = (state, sqrt(2.0))
    FHS.apply_local_operator(::PreciseRootOp, state, space, precomp, ::Type{T}) where {T} = (state, sqrt(T(2)))

    @boson b
    H = hilbert_space(b, 3)
    amplitude(op, ::Type{T}, H, transpose) where {T} =
        last(FHS.apply_local_operators(FHS.NCMul(one(T), [op]), FHS.basisstate(1, H), H, (nothing,); transpose))

    @test (@inferred amplitude(RootOp(), Float32, H, false)) === Float32(sqrt(2.0))
    @test (@inferred amplitude(RootOp(), Float32, H, true)) === Float32(sqrt(2.0))
    @test amplitude(PreciseRootOp(), BigFloat, H, false) == sqrt(big(2))

    struct BigRootOp <: FHS.AbstractSym end
    Base.adjoint(op::BigRootOp) = op
    FHS.symbolic_group(::BigRootOp) = FHS.symbolic_group(b)
    FHS.mat_eltype(::Type{BigRootOp}) = BigFloat
    FHS.apply_local_operator(::BigRootOp, state, space, precomp) = (state, sqrt(big(2)))
    @test representation(BigRootOp(), H)[1, 1] == sqrt(big(2))
end

@testitem "Products of operator types with different matrix element types" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: AbstractSym, AbstractBasisState, symbolic_group, apply_local_operator, mat_eltype
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc
    struct Shift <: AbstractSym
        k::Int
    end
    struct Phase <: AbstractSym end
    @nc Shift Phase
    Base.adjoint(s::Shift) = Shift(-s.k)
    Base.adjoint(p::Phase) = p
    symbolic_group(::Union{Shift, Phase}) = 1
    struct Site <: AbstractBasisState
        n::Int
    end
    apply_local_operator(s::Shift, state::Site, space, precomp) = (Site(mod(state.n + s.k, 3)), 1)
    apply_local_operator(::Phase, state::Site, space, precomp) = (state, im^state.n)
    mat_eltype(::Type{Shift}) = Int
    mat_eltype(::Type{Phase}) = ComplexF64

    H = GenericHilbertSpace(1, Site.(0:2))
    S = representation(Shift(1), H, Matrix)
    @test representation(Phase() * Shift(1), H, Matrix) ≈ Diagonal(im .^ (0:2)) * S
    @test representation(Phase() * Shift(1) + Shift(1), H) ≈ (Diagonal(im .^ (0:2)) + I) * S
end

@testitem "Operators between two spaces with Axes" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: Axes
    @fermions f
    H = hilbert_space(f, 1:3)
    Hin = hilbert_space(f, 1:3, NumberConservation([1, 2]))
    Hout = hilbert_space(f, 1:3, NumberConservation([0, 1]))
    sliced(op) = representation(op, H)[indices(Hout, H), indices(Hin, H)]

    op = f[1]' * f[2] + hc + 2
    @test representation(op, Axes(H, H)) == representation(op, H)

    @test_throws ArgumentError representation(f[2] + 3, Axes(Hout, Hin))
    @test representation(f[2] + 3, Axes(Hout, Hin); projection=true) == sliced(f[2] + 3)
    @test representation(2I, Axes(Hout, Hin); projection=true) == sliced(2)

    @test_throws ArgumentError representation(f[1], Axes(hilbert_space(f, 1:2), Hin))
    @boson b
    @test_throws ArgumentError representation(b', Axes(hilbert_space(b, 5), hilbert_space(b, 3)))

    @majoranas γ
    Hγb = tensor_product(hilbert_space(γ, 1:2), hilbert_space(b, 3))
    states = collect(basisstates(Hγb))
    Kout, Kin = constrain_space(Hγb, states[1:4]), constrain_space(Hγb, states[3:6])
    op = γ[1] * γ[2] + b' * b
    @test representation(op, Axes(Kout, Kin); projection=true) == representation(op, Hγb)[indices(Kout, Hγb), indices(Kin, Hγb)]

    ket = Ket(basisstate(1, Hout), Hout)
    bra = Ket(basisstate(2, Hin), Hin)'
    @test representation(ket * bra, Axes(Hout, Hin)) == representation(ket, Hout) * representation(bra, Hin)

    @test representation(ket, Axes(Hout)) == representation(ket, Hout)
    @test representation("010", Axes(H)) == representation("010", H)
    @test_throws ArgumentError representation(f[1], Axes(H))
    @test_throws ArgumentError representation(ket, Axes(Hout, Hout))
end

@testitem "Product-space term matrices padded by identities" begin
    using SparseArrays, LinearAlgebra
    @fermions f
    @spin s 1 // 2
    @bosons b
    Hf = hilbert_space(f, 1:2)
    Hs = hilbert_space(s)
    Hb = hilbert_space(b, 1:2, 3)
    H = tensor_product(Hf, Hs, Hb)
    Mf = representation(f[1]' * f[2], Hf)
    Mb = representation(2 * b[1]' * b[2], Hb)
    Ms = representation(s[:z], Hs)
    @test representation(2 * f[1]' * f[2] * b[1]' * b[2], H) == kron(Mb, I(dim(Hs)), Mf)
    @test representation(f[1]' * f[2], H) == kron(I(dim(Hb) * dim(Hs)), Mf)
    @test representation(2 * b[1]' * b[2], H) == kron(Mb, I(dim(Hs) * dim(Hf)))
    @test representation(s[:z], H) == kron(I(dim(Hb)), Ms, I(dim(Hf)))
    op = f[1]' * f[2] * s[:z] + 2 * b[1]' * b[2] + hc
    M = representation(op, H)
    @test M isa SparseMatrixCSC{Float64}
    @test M ≈ kron(I(dim(Hb)), Ms, Mf) + kron(Mb, I(dim(Hs) * dim(Hf))) + adjoint(kron(I(dim(Hb)), Ms, Mf) + kron(Mb, I(dim(Hs) * dim(Hf))))
    @test representation(2 * f[1]' * f[2] * b[1]' + hc, H) isa SparseMatrixCSC{Float64}
end

@testitem "Dense/sparse/lazy matrix representation agree" begin
    using SparseArrays, LinearAlgebra
    import FermionicHilbertSpaces.SciMLOperators: concretize
    @fermions f
    @spin s 1 // 2

    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im * f[2]' * f[1] + 2
    Md = representation(op, Hf, Matrix)
    Ms = representation(op, Hf)
    Ml = concretize(representation(op, Hf, :lazy))
    @test Md isa Matrix
    @test Md == Matrix(Ms)
    @test Md == Ms
    @test Md == Ml
    Md_single = Md

    Hs = hilbert_space(s)
    Hprod = tensor_product(Hf, Hs)
    op_prod = f[1]' * f[2] * s[:z] + 2 * f[2]' * f[1] * s[:x] + 1im
    Md = representation(op_prod, Hprod, Matrix)
    Ms = representation(op_prod, Hprod)
    Ml = concretize(representation(op_prod, Hprod, :lazy))
    @test Md isa Matrix
    @test Md == Matrix(Ms)
    @test Md == Ms
    @test Md == Ml

    trivial = representation(2, Hf, Matrix)
    @test trivial isa Matrix
    @test trivial == Matrix(2I, dim(Hf), dim(Hf))

    direct_dense = representation(op, Hf, FermionicHilbertSpaces.EagerDenseRepr())
    @test direct_dense isa Matrix
    @test direct_dense == Md_single
end

@testitem "Array types as representation formats" begin
    using SparseArrays, LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + hc
    k = Ket(basisstate(2, H), H)
    ket = representation("10", H, Vector)
    formats = (
        (Matrix, Vector, Matrix{Int}, Vector{Int}), (Array, Array, Matrix{Int}, Vector{Int}),
        (Matrix{ComplexF64}, Vector{ComplexF64}, Matrix{ComplexF64}, Vector{ComplexF64}),
        (SparseMatrixCSC, SparseVector, SparseMatrixCSC{Int, Int}, SparseVector{Int, Int}),
        (SparseMatrixCSC{Float64}, SparseVector{Float64}, SparseMatrixCSC{Float64, Int}, SparseVector{Float64, Int}),
        (Float32, Float32, SparseMatrixCSC{Float32, Int}, SparseVector{Float32, Int}),
    )
    for (mformat, vformat, MT, VT) in formats
        @test (@inferred representation(op, H, mformat)) isa MT
        @test representation(op, H, mformat) == representation(op, H, Matrix)
        @test representation(2, H, mformat) isa MT
        @test representation(2, H, mformat) == 2 * representation(1, H, Matrix)
        @test representation(k * k', H, mformat) == representation(k * k', H, Matrix)
        @test (@inferred representation(k, H, vformat)) isa VT
        @test representation("10", H, vformat) == ket
    end
    for format in ("dense", nothing, Any, Diagonal, Hermitian, Array{Float64, 3}, SparseMatrixCSC{Float64, Int32})
        @test_throws ArgumentError representation(op, H, format)
        @test_throws ArgumentError representation(2, H, format)
        @test_throws ArgumentError representation("10", H, format)
    end
    for format in (Vector, SparseVector{Float64})
        @test_throws ArgumentError representation(op, H, format)
        @test_throws ArgumentError representation(2, H, format)
        @test_throws ArgumentError representation(k * k', H, format)
    end
    for format in (Matrix, SparseMatrixCSC{Float64})
        @test_throws ArgumentError representation(k, H, format)
        @test_throws ArgumentError representation("10", H, format)
    end
    for format in (Vector{Float32}, SparseVector{Float32})
        @test representation(1.5 * k, H, format) isa format
        @test representation(1.5 * (k + Ket(basisstate(1, H), H)), H, format) isa format
        @test representation(1.5 * k - 1.5 * k, H, format) isa format
    end
    @test_throws InexactError representation(1im * k, H, Vector{Float64})
    Hbdg = hilbert_space(f, 1:2, FermionicHilbertSpaces.FreeFermions.BdG())
    @test representation(2, Hbdg, Matrix{ComplexF64}) isa Matrix{ComplexF64}
end

@testitem "Format without a method for the identity" begin
    import FermionicHilbertSpaces: _canonicalize_repr
    struct BareFormat end
    _canonicalize_repr(repr::BareFormat) = repr
    @fermions f
    @test_throws MethodError representation(2, hilbert_space(f, 1:2), BareFormat())
end

@testitem "Format defined by _terms_matrix" begin
    import FermionicHilbertSpaces: _canonicalize_repr, _terms_matrix, EagerSparseRepr
    struct TermsFormat end
    _canonicalize_repr(repr::TermsFormat) = repr
    _terms_matrix(::Type{T}, terms, coeff, space, ::TermsFormat, chunking; kwargs...) where {T} =
        _terms_matrix(T, terms, coeff, space, EagerSparseRepr(), chunking; kwargs...)
    @fermions f
    @boson b
    Hf = hilbert_space(f, 1:2)
    H = tensor_product(Hf, hilbert_space(b, 3))
    op = f[1]' * f[2] + hc + 0.5 * f[1]' * f[1] * (b + b') + 2
    @test representation(op, H, TermsFormat()) == representation(op, H)
    opf = f[1]' * f[2] + hc + 2
    @test representation(opf, Hf, TermsFormat()) == representation(opf, Hf)
    @test representation(f[1], Hf, TermsFormat()) == representation(f[1], Hf)
    @test representation(f[1], H, TermsFormat()) == representation(f[1], H)
end

@testitem "Scheduler extension correctness" begin
    using SparseArrays, LinearAlgebra
    using OhMyThreads
    import FermionicHilbertSpaces: TermChunking, StateChunking, EagerDenseRepr, EagerSparseRepr

    @fermions f
    H = hilbert_space(f, 1:4)
    op = f[1]' * f[2] + 1im * f[2]' * f[1] + f[3]' * f[4] + f[4]' * f[3] + 2

    M_serial = representation(op, H)
    scheduler = StaticScheduler(; nchunks=4)
    M_terms = representation(op, H; chunking=TermChunking(scheduler))
    M_states = representation(op, H; chunking=StateChunking(scheduler))
    @test M_terms == M_serial
    @test M_states == M_serial

    M_terms_dense = representation(op, H, EagerDenseRepr(); chunking=TermChunking(scheduler))
    M_states_dense = representation(op, H, EagerDenseRepr(); chunking=StateChunking(scheduler))
    @test M_terms_dense == M_serial
    @test M_states_dense == M_serial

    # Element type of an abstract representation type
    op_int = f[4]' * f[4] + f[4]' * f[3] * f[3]' * f[4]
    @test typeof(representation(op_int, H, EagerSparseRepr{Number}(); chunking=StateChunking(scheduler))) == typeof(representation(op_int, H, EagerSparseRepr{Number}()))

    # Space of dimension zero
    H0 = constrain_space(hilbert_space(f, 1:2), NumberConservation(5))
    @test representation(f[1]' * f[2] + 1, H0; chunking=StateChunking(scheduler)) == representation(f[1]' * f[2] + 1, H0)

    # Larger space
    Hl = hilbert_space(f, 1:13)
    opl = sum(f[k]' * f[k+1] + hc for k in 1:12) + sum(k * f[k]' * f[k] * f[k+1]' * f[k+1] for k in 1:12) + 1.5
    @test representation(opl, Hl; chunking=StateChunking(StaticScheduler(; nchunks=3))) == representation(opl, Hl)
end

@testitem "Chunking on product and constrained spaces" begin
    using SparseArrays, LinearAlgebra
    using OhMyThreads
    import FermionicHilbertSpaces: TermChunking, StateChunking, EagerDenseRepr

    scheduler = StaticScheduler(; nchunks=4)

    # Product space
    @fermions f
    @spin s 1 // 2
    Hf = hilbert_space(f, 1:3)
    Hs = hilbert_space(s)
    Hprod = tensor_product(Hf, Hs)
    op_prod = f[1]' * f[2] * s[:z] + 2 * f[2]' * f[3] * s[:x] + hc + 1im

    M_serial_prod = representation(op_prod, Hprod)
    M_terms_prod = representation(op_prod, Hprod; chunking=TermChunking(scheduler))
    M_states_prod = representation(op_prod, Hprod; chunking=StateChunking(scheduler))
    @test M_terms_prod == M_serial_prod
    @test M_states_prod == M_serial_prod

    M_terms_prod_dense = representation(op_prod, Hprod, EagerDenseRepr(); chunking=TermChunking(scheduler))
    M_states_prod_dense = representation(op_prod, Hprod, EagerDenseRepr(); chunking=StateChunking(scheduler))
    @test M_terms_prod_dense == M_serial_prod
    @test M_states_prod_dense == M_serial_prod

    # Terms on single groups, the middle one padded by identities on both sides
    @boson b
    H3 = tensor_product(Hf, Hs, hilbert_space(b, 3))
    op3 = f[1]' * f[2] + hc + 2 * s[:z] + b' * b + 1im
    @test representation(op3, H3; chunking = StateChunking(scheduler)) == representation(op3, H3)
    @test representation(op3, H3, EagerDenseRepr(); chunking = TermChunking(scheduler)) == representation(op3, H3)

    # Constrained single space
    Hc = constrain_space(Hf, NumberConservation(1:2))
    op_cons = f[1]' * f[2] + f[2]' * f[3] + hc + 2

    M_serial_cons = representation(op_cons, Hc)
    M_terms_cons = representation(op_cons, Hc; chunking=TermChunking(scheduler))
    M_states_cons = representation(op_cons, Hc; chunking=StateChunking(scheduler))
    @test M_terms_cons == M_serial_cons
    @test M_states_cons == M_serial_cons

    M_terms_cons_dense = representation(op_cons, Hc, EagerDenseRepr(); chunking=TermChunking(scheduler))
    M_states_cons_dense = representation(op_cons, Hc, EagerDenseRepr(); chunking=StateChunking(scheduler))
    @test M_terms_cons_dense == M_serial_cons
    @test M_states_cons_dense == M_serial_cons

    # Constrained product space
    @fermions g
    Hg = hilbert_space(g, 1:2)
    Hf1 = constrain_space(hilbert_space(f, 1:2), NumberConservation(1))
    Hg1 = constrain_space(Hg, NumberConservation(1))
    Hcons_prod = tensor_product(Hf1, Hg1)
    op_cons_prod = f[1]' * f[2] + g[1]' * g[2] + hc + 1

    M_serial_cons_prod = representation(op_cons_prod, Hcons_prod)
    M_terms_cons_prod = representation(op_cons_prod, Hcons_prod; chunking=TermChunking(scheduler))
    M_states_cons_prod = representation(op_cons_prod, Hcons_prod; chunking=StateChunking(scheduler))
    @test M_terms_cons_prod == M_serial_cons_prod
    @test M_states_cons_prod == M_serial_cons_prod

    M_terms_cons_prod_dense = representation(op_cons_prod, Hcons_prod, EagerDenseRepr(); chunking=TermChunking(scheduler))
    M_states_cons_prod_dense = representation(op_cons_prod, Hcons_prod, EagerDenseRepr(); chunking=StateChunking(scheduler))
    @test M_terms_cons_prod_dense == M_serial_cons_prod
    @test M_states_cons_prod_dense == M_serial_cons_prod
end

@testitem "Distributed sparse matrix representation" begin
    import FermionicHilbertSpaces: PartitionedSparseRepr
    using LinearAlgebra
    using PartitionedArrays
    @fermions f
    H = hilbert_space(f, 1:3)
    op = f[1]' * f[2] + (1 + 2im) * f[3]' * f[1] + 2.0

    M_ref = representation(op, H)
    rep = PartitionedSparseRepr(; backend=DebugArray)
    M_dist = representation(op, H, rep)

    v = pvector(M_dist.col_partition) do inds
        rand(length(inds))
    end
    @test collect(M_dist * v) ≈ M_ref * collect(v)

    @boson b
    Hp = tensor_product(H, hilbert_space(b, 2))
    for opp in (op * b' + hc + 1, f[1])
        Mp = representation(opp, Hp, rep)
        w = pvector(inds -> rand(ComplexF64, length(inds)), Mp.col_partition)
        @test collect(Mp * w) ≈ representation(opp, Hp) * collect(w)
    end
end

## Symmetries
isnumberconserving(x::AbstractFermionSym) = false
isnumberconserving(x::NCMul) = iszero(sum(s -> 2s.creation - 1, x.factors))
isnumberconserving(x::NCAdd) = all(isnumberconserving, NCterms(x))

isparityconserving(x::AbstractFermionSym) = false
isparityconserving(x::NCMul) = iseven(length(x.factors))
isparityconserving(x::NCAdd) = all(isparityconserving, NCterms(x))

isquadratic(::AbstractFermionSym) = false
isquadratic(x::NCMul) = length(x.factors) == 2
isquadratic(x::NCAdd) = all(isquadratic, NCterms(x))

@testitem "Fermion symmetry property checks" begin
    import FermionicHilbertSpaces: isnumberconserving, isparityconserving, isquadratic
    @fermions f
    # isnumberconserving
    @test !isnumberconserving(f[1])
    @test isnumberconserving(f[1]'f[2])
    @test !isnumberconserving(f[1]f[2])
    @test !isnumberconserving(f[1]'f[2] + f[3])
    @test isnumberconserving(f[1]'f[2] + f[3]f[3]' + 1)
    # isparityconserving
    @test !isparityconserving(f[1])
    @test isparityconserving(f[1]f[2])
    @test !isparityconserving(f[1]f[2] * f[3])
    @test !isparityconserving(f[1]f[2] + f[3])
    @test isparityconserving(f[1]f[2] + f[3]f[3]' + 1)
    # isquadratic
    @test !isquadratic(f[1])
    @test isquadratic(f[1]f[2])
    @test !isquadratic(f[1]f[2] * f[3])
    @test isquadratic(f[1]f[2] + f[3] * f[3]' + 1)
end

@testitem "Single-particle and BdG operator checks" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    @majoranas γ
    @spin s 1 // 2
    Hsp = hilbert_space(f, 1:3, SingleParticle())
    Hbdg = hilbert_space(f, 1:3, BdG())
    op = 1.0 * f[1]' * f[2] + hc + f[3]' * f[3] + 2
    @test representation(op, Hsp) == representation(op - 2, Hsp)
    @test representation(op, Hbdg) == representation(op - 2, Hbdg)
    @test representation(f[1]' * f[2]' + hc, Hbdg) isa AbstractMatrix
    for H in (Hsp, Hbdg)
        @test_throws ArgumentError representation(f[1]' * f[2] * f[3]' * f[3], H)
        @test_throws ArgumentError representation(f[1]', H)
        @test_throws ArgumentError representation(f[1]' * f[2] + f[3]', H)
        @test_throws ArgumentError representation(1im * γ[1] * γ[2], H)
        @test_throws ArgumentError representation(f[1]' * s[:z], H)
    end
    @test_throws ArgumentError representation(f[1]' * f[2]' + hc, Hsp)
end

@testitem "Consistency between + and add!!" begin
    import FermionicHilbertSpaces.NonCommutativeProducts.add!!
    @fermions f
    a = 1.0 * f[2] * f[1] + 1 + f[1]
    for b in [1.0, 1, f[1], 1.0 * f[1], f[2] * f[1], a]
        a2 = copy(a)
        a3 = add!!(a2, b) # Should mutate
        @test a + b == a3
        @test a2 == a3
        anew = add!!(a, 1im * b) #Should not mutate
        @test a2 !== anew
    end
    @test a == 1.0 * f[2] * f[1] + 1 + f[1]
end

_sum_matrices(matrices, ::Union{EagerSparseRepr,EagerDenseRepr}; tree_split=10) = _sum_tree(matrices, tree_split)
function _sum_tree(matrices, k)
    n = length(matrices)
    n == 0 && throw(ArgumentError("Cannot sum an empty collection of matrices."))
    n == 1 && return only(matrices)
    n <= k && return map(+, matrices...)

    # Divide into k roughly equal parts
    chunk_size = cld(n, k)
    chunks = Iterators.partition(matrices, chunk_size)

    partials = [_sum_tree(collect(chunk), k) for chunk in chunks]
    return map(+, partials...)
end
