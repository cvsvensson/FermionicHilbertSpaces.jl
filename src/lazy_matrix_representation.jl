"""
    LazyOperator{O,S,T,C,P}

A matrix-free (lazy) representation of a symbolic operator on a Hilbert space.
Acts directly on vectors and matrices without constructing a sparse matrix.

`representation(op, space, :lazy)` returns a `SciMLOperators.FunctionOperator`
that wraps a `LazyOperator`. The bare `LazyOperator` is available as
`LazyOperator(op, space, chunking)`.
"""
mutable struct LazyOperator{O,S,T,C,P}
    const op::O
    const space::S
    const chunking::C
    const precomp::P
    const projection::Bool
    const conjugate::Bool
    const ishermitian::Bool
    cache::Any
end

function _show_lazy_operator_expression(io::IO, L::LazyOperator)
    show(IOContext(io, :compact => true), L.op)
    L.conjugate && print(io, "^*")
end

function Base.show(io::IO, L::LazyOperator)
    _show_lazy_operator_expression(io, L)
    print(io, " acting on ")
    show(IOContext(io, :compact => true), L.space)
end

function LazyOperator(_op, space::S, chunking::C=NoChunking(); projection=false, conjugate=false, ishermitian=_ishermitian(_op), T=mat_eltype(_op), concretizer=NoConcretizer()
) where {S,C<:AbstractChunkingStrategy}
    op = __concretize(_op, concretizer)
    precomp = lazy_precomputation(op, space; projection)
    O = typeof(op)
    P = typeof(precomp)
    LazyOperator{O,S,T,C,P}(op, space, chunking, precomp, projection, conjugate, ishermitian, nothing)
end
function LazyOperator(op::O, space::S, chunking::C, precomp::P; projection=false, conjugate=false, ishermitian=_ishermitian(op), T=mat_eltype(op)) where {O,S,C<:AbstractChunkingStrategy,P}
    LazyOperator{O,S,T,C,P}(op, space, chunking, precomp, projection, conjugate, ishermitian, nothing)
end

# Data that `lazy_mul!` needs to apply `op` on `space`, computed once when the `LazyOperator`
# is constructed. For an `NCAdd` it is a vector of `(term, precomp)` tuples for the terms of
# `op` without its scalar, which are the same as those of the eager representation.
lazy_precomputation(op, space; _...) = precomputation_before_operator_application(op, space)
function lazy_precomputation(op::NCAdd, space; projection)
    bases = group_ids(space)
    terms = length(bases) == 1 ? NCterms(op) : _product_terms(op, bases, space, EagerSparseRepr(); projection)
    return _precomputed_terms(terms, space)
end

function _lazy_output_prototype(L::LazyOperator, input::AbstractVector)
    size(L, 2) == length(input) || throw(DimensionMismatch("The input must have length $(size(L, 2)), got $(length(input))"))
    T = promote_type(eltype(L), eltype(input))
    return similar(input, T, size(L, 1))
end

function _lazy_output_prototype(L::LazyOperator, input::AbstractMatrix)
    size(input, 1) == size(L, 2) || throw(DimensionMismatch("The input must have $(size(L, 2)) rows, got size $(size(input))"))
    T = promote_type(eltype(L), eltype(input))
    return similar(input, T, (size(L, 1), size(input, 2)))
end


# The default cache, `similar(input, T, dims)`, fails for lengths that do not fit in an `Int`.
function scimloperator(L::LazyOperator, input=Vector{eltype(L)}(undef, dim(L.space)), output=input; kwargs...)
    SciMLOperators.FunctionOperator(L, input, output; ishermitian=ishermitian(L), op_adjoint=adjoint(L), isconstant=true, T=eltype(L), islinear=true, batch=true, cache=(similar(input), similar(output)), kwargs...)
end

(L::LazyOperator)(w, v, u, p, t, α, β) = mul!(w, L, v, α, β)
(L::LazyOperator)(w, v, u, p, t) = mul!(w, L, v)
(L::LazyOperator)(v, u, p, t) = L * v

_ishermitian(x::NCMul) = iszero(x - hc)
_ishermitian(x::NCAdd) = iszero(x - hc)
function _ishermitian(x::ProductOperator)
    _ishermitian(prod(x.ops))
end

Base.size(L::LazyOperator) = (d = dim(L.space); (d, d))
Base.size(L::LazyOperator, i::Int) = size(L)[i]
Base.eltype(::LazyOperator{O,S,T,C,P}) where {O,S,T,C,P} = T
Base.conj(L::LazyOperator) = LazyOperator(L.op, L.space, L.chunking, L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
Base.adjoint(L::LazyOperator) = conj(transpose(L))
function Base.transpose(L::LazyOperator)
    return LazyOperator(_transposed(L.op), TransposedSpace(L.space), L.chunking, _transposed_precomp(L); projection = L.projection, conjugate = L.conjugate, ishermitian = L.ishermitian)
end
# On the transposed space, operators act transposed with the same precomputation, except for the
# spaces of a `ProductOperator` and the matrices of a `KronTerm`.
_transposed(op) = op
_transposed(op::ProductOperator{C}) where {C} = ProductOperator{C}(op.ops, map(TransposedSpace, op.spaces), op.inds)
_transposed(term::KronTerm) = KronTerm(map(((A, r),) -> (_transposed_matrix(A), r), term.factors))
# Symmetric factors, such as the diagonal term of a product space, are shared instead of copied.
_transposed_matrix(A) = issymmetric(A) ? A : copy(transpose(A))
_transposed_precomp(L::LazyOperator) = L.precomp
_transposed_precomp(L::LazyOperator{<:NCAdd}) = [(_transposed(term), precomp) for (term, precomp) in L.precomp]
const LazyFunctionOperator = SciMLOperators.FunctionOperator{<:Any, <:Any, <:Any, <:Any, <:LazyOperator}
# SciMLOperators' fallbacks for these build the full matrix, so they are rewrapped lazily.
for f in (:transpose, :conj)
    @eval function Base.$f(L::LazyFunctionOperator)
        isnothing(L.cache) && return scimloperator($f(L.op))
        return scimloperator($f(L.op), first(L.cache))
    end
end
SparseArrays.sparse(L::LazyFunctionOperator) = sparse(L.op)
LinearAlgebra.ishermitian(L::LazyOperator) = L.ishermitian

function _eager_matrix_representation(L::LazyOperator{<:Union{NCMul,ProductOperator}})
    return _term_matrix_representation(L.op, L.space, EagerSparseRepr(), L.chunking; projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:NCAdd})
    return _terms_matrix(eltype(L), map(first, L.precomp), L.op.coeff, L.space, EagerSparseRepr(), L.chunking; projection = L.projection)
end

function Base.convert(::Type{AbstractMatrix}, L::LazyOperator)
    sparse(L)
end

function SparseArrays.sparse(L::LazyOperator)
    M = _eager_matrix_representation(L)
    L.conjugate && (M = conj(M))
    return M
end

function LinearAlgebra.mul!(y::AbstractVector, L::LazyOperator, x::AbstractVector, α=true, β=false)
    return lazy_mul!(y, L, x, α, β)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, L::LazyOperator, X::AbstractMatrix, α=true, β=false)
    size(X, 1) == size(L, 2) || throw(DimensionMismatch("The input must have $(size(L, 2)) rows, got size $(size(X))"))
    expected_size = (size(L, 1), size(X, 2))
    size(Y) == expected_size || throw(DimensionMismatch("The output must have size $expected_size, got $(size(Y))"))
    lazy_mul!(Y, L, X, α, β)
    return Y
end

# `conjugate` applies to the amplitudes of the terms. The `α` of `mul!` is never conjugated.
_conj_if(c, conjugate) = conjugate ? conj(c) : c

# Adds `coeff * amp * x[inind]` to `y[outind]`, or to row `outind` of `y` for matrices, for the entries of a term.
struct MulAccumulator{Y, X, C}
    y::Y
    x::X
    coeff::C
    conjugate::Bool
end
function push_inds_amps!(acc::MulAccumulator{<:AbstractVector}, outind, inind, amp)
    acc.y[outind] += _conj_if(amp, acc.conjugate) * acc.coeff * acc.x[inind]
    return nothing
end
function push_inds_amps!(acc::MulAccumulator{<:AbstractMatrix}, outind, inind, amp)
    @views acc.y[outind, :] .+= (_conj_if(amp, acc.conjugate) * acc.coeff) .* acc.x[inind, :]
    return nothing
end
function push_inds_amps!(acc::MulAccumulator{<:Function, <:Number}, outind, _, amp)
    acc.y(outind, _conj_if(amp, acc.conjugate) * acc.coeff * acc.x)
    return nothing
end

# For a dense input, the entries of a `KronTerm` are visited factor by factor over the stored entries, with the
# identities between the factors as ranges of indices, instead of column by column. This skips empty columns.
function chunked_operator_indices_and_amplitudes_generic!(acc::MulAccumulator{<:AbstractVecOrMat}, term::KronTerm, space::AbstractHilbertSpace, precomp, input_inds, col_offset; kwargs...)
    return _kron_mul!(acc, term.factors, dim(space), 0, 0, true, input_inds)
end
# The entries of `amp * kron(…, A_2, I, A_1, I)` on `n` states in the columns `cols ⊆ col+1:col+n`, offset by `row` and `col`.
function _kron_mul!(acc, factors, n, row, col, amp, cols)
    A, r = last(factors)
    rows, vals = rowvals(A), nonzeros(A)
    for b in 0:(r * size(A, 2)):(n - 1), c in axes(A, 2)
        j0 = col + b + (c - 1) * r
        js = max(j0 + 1, first(cols)):min(j0 + r, last(cols))
        isempty(js) && continue
        if factors isa Tuple{Any} && r == 1 && acc.y isa AbstractVector
            # The runs have length one, so the loop over the stored entries is innermost.
            xj = _conj_if(amp, acc.conjugate) * acc.coeff * acc.x[j0 + 1]
            for p in nzrange(A, c)
                acc.y[row + b + rows[p]] += _conj_if(vals[p], acc.conjugate) * xj
            end
        else
            for p in nzrange(A, c)
                _kron_mul!(acc, Base.front(factors), r, row + b + (rows[p] - 1) * r, j0, amp * vals[p], js)
            end
        end
    end
    return acc
end
function _kron_mul!(acc, ::Tuple{}, n, row, col, amp, cols)
    for j in cols
        push_inds_amps!(acc, j + row - col, j, amp)
    end
    return acc
end

# Adds `coeff * term * x[cols]` to `y`, where `x[cols]` is `x` with the entries outside the rows `cols` set to zero.
function _apply_single_term!(y, x, space, term, precomp, coeff, conjugate, projection, cols = axes(x, 1))
    acc = MulAccumulator(y, x, coeff, conjugate)
    return chunked_operator_indices_and_amplitudes_generic!(acc, term, space, precomp, cols, 0; projection)
end
function _apply_single_term!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, coeff, conjugate, projection, cols = axes(x, 1))
    return _foreach_term_entry((i, v) -> (y[i] += v), x, space, term, precomp, coeff, conjugate, projection, cols)
end
function _apply_single_term!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, coeff, conjugate, projection, cols = axes(x, 1))
    for col in axes(x, 2)
        _foreach_term_entry((i, v) -> (y[i, col] += v), view(x, :, col), space, term, precomp, coeff, conjugate, projection, cols)
    end
end

# Calls `f(outind, value)` for each output entry of `coeff * term` applied to the stored entries of the sparse `x` in
# the rows `cols`. Each entry is applied as a range of one column, as a `KronTerm` takes ranges.
# `f::F` makes Julia specialize on `f`, which it otherwise skips as `f` is only passed on.
function _foreach_term_entry(f::F, x, space, term, precomp, coeff, conjugate, projection, cols = axes(x, 1)) where {F}
    for (n, xn) in zip(SparseArrays.nonzeroinds(x), nonzeros(x))
        (iszero(xn) || !(n in cols)) && continue
        chunked_operator_indices_and_amplitudes_generic!(MulAccumulator(f, xn, coeff, conjugate), term, space, precomp, n:n, 0; projection)
    end
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator, x::AbstractVecOrMat, α, β) where {T}
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    for (term, precomp) in _lazy_terms(L)
        _apply_single_term!(y, x, L.space, term, precomp, α, L.conjugate, L.projection)
    end
    _add_scalar!(y, L, x, α)
    return y
end
function _add_scalar!(y, L, x, α)
    L.op isa NCAdd && !iszero(L.op.coeff) && !iszero(α) && (y .+= (α * _conj_if(L.op.coeff, L.conjugate)) .* x)
    return y
end

function Base.:*(L::LazyOperator, x::AbstractVecOrMat)
    T = promote_type(eltype(L), eltype(x))
    y = similar(x, T)
    return mul!(y, L, x)
end

# `sparsevec` sums the collected values with equal index.
function Base.:*(L::LazyOperator, x::SparseArrays.AbstractSparseVector)
    size(L, 2) == length(x) || throw(DimensionMismatch("The input must have length $(size(L, 2)), got $(length(x))"))
    T = promote_type(eltype(L), eltype(x))
    outinds = typeof(size(L, 1))[]
    amps = T[]
    push_entry = (i, v) -> (push!(outinds, i); push!(amps, v))
    for (term, precomp) in _lazy_terms(L)
        _foreach_term_entry(push_entry, x, L.space, term, precomp, true, L.conjugate, L.projection)
    end
    if L.op isa NCAdd && !iszero(L.op.coeff)
        scalar_coeff = _conj_if(L.op.coeff, L.conjugate)
        append!(outinds, SparseArrays.nonzeroinds(x))
        append!(amps, scalar_coeff .* nonzeros(x))
    end
    return SparseArrays.sparsevec(outinds, amps, size(L, 1))
end
_lazy_terms(L::LazyOperator{<:NCAdd}) = L.precomp
_lazy_terms(L::LazyOperator) = ((L.op, L.precomp),)
_sparse_input(H) = (d = dim(H); spzeros(mat_eltype(H), typeof(d), d))
get_input(::LazyRepr{Missing}, H) = dim(H) < 1e4 ? Vector{mat_eltype(H)}(undef, dim(H)) : _sparse_input(H)
function get_input(rep::LazyRepr{<:AbstractArray}, H)
    dim(H) == size(rep.input, 1) || throw(DimensionMismatch("The input must have $(dim(H)) rows, got size $(size(rep.input))"))
    return rep.input
end
function get_input(rep::LazyRepr{Symbol}, H)
    rep.input == :dense && return Vector{mat_eltype(H)}(undef, dim(H))
    rep.input == :sparse && return _sparse_input(H)
end

function _term_matrix_representation(op::Union{NCMul,ProductOperator}, H::AbstractHilbertSpace, rep::LazyRepr, chunking; kwargs...)
    scimloperator(LazyOperator(op, H, chunking; kwargs...), get_input(rep, H))
end

function _matrix_representation(op::NCAdd, bases, space, rep::LazyRepr, chunking; kwargs...)
    return scimloperator(LazyOperator(op, space, chunking; kwargs...), get_input(rep, space))
end

@testitem "lazy matrix_representation" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    @fermions f
    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im
    L = representation(op, Hf, :lazy)
    M = representation(op, Hf)
    v = randn(dim(Hf))
    w = randn(ComplexF64, dim(Hf))
    Vm = randn(dim(Hf), 3)
    Wm = randn(ComplexF64, dim(Hf), 3)
    α = 0.7
    β = -0.2
    @test SciMLOperators.isconstant(L)
    @test SciMLOperators.islinear(L)
    @test !SciMLOperators.isconvertible(L)
    @test SciMLOperators.has_mul(L)
    @test SciMLOperators.has_mul!(L)
    @test SciMLOperators.has_adjoint(L)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) isa SciMLOperators.FunctionOperator
    @test adjoint(L) isa SciMLOperators.FunctionOperator
    @test transpose(L) * v ≈ transpose(M) * v
    @test conj(L) isa SciMLOperators.FunctionOperator
    @test conj(L) * v ≈ conj(M) * v
    @test sparse(L) ≈ M
    @test sparse(L') ≈ M'
    @test L * Vm ≈ M * Vm
    t = 0
    u = p = nothing
    @test L(v, u, p, t) ≈ M * v
    @test L(Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t) ≈ M * v
    @test L(copy(Wm), Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t, α, β) ≈ α .* (M * v) .+ β .* w
    # a zero β overwrites the output, including non-finite entries
    @test L(fill(NaN + 0im, dim(Hf)), v, u, p, t, α, 0.0) ≈ α .* (M * v)
    @test L(fill(NaN + 0im, dim(Hf), 3), Vm, u, p, t, α, 0.0) ≈ α .* (M * Vm)
    @test L(copy(Wm), Vm, u, p, t, α, β) ≈ α .* (M * Vm) .+ β .* Wm
    vs = sprandn(dim(Hf), 0.5)
    ws = sprandn(ComplexF64, dim(Hf), 0.5)
    Xs = sprandn(dim(Hf), 3, 0.5)
    Ys = sprandn(ComplexF64, dim(Hf), 3, 0.5)
    @test L * vs ≈ M * vs
    @test L' * vs ≈ M' * vs
    @test transpose(L) * vs ≈ transpose(M) * vs
    @test L * Xs ≈ M * Xs
    @test L' * Xs ≈ M' * Xs
    @test transpose(L) * Xs ≈ transpose(M) * Xs
    @test L(copy(ws), vs, u, p, t, α, β) ≈ α .* (M * vs) .+ β .* ws
    @test L(copy(Ys), Xs, u, p, t, α, β) ≈ α .* (M * Xs) .+ β .* Ys
    @test (L * vs) isa SparseVector
    @test (L * Xs) isa SparseMatrixCSC
    @test (L + one(L)) * v ≈ (M + I) * v
    @test (2 * L) * v ≈ 2 * (M * v)
    @test concretize(L) == M
    @test !ishermitian(L)
    @test ishermitian(representation(op + hc, Hf, :lazy))

    import FermionicHilbertSpaces: LazyRepr
    for input in (:dense, :sparse, zeros(ComplexF64, dim(Hf)), spzeros(ComplexF64, dim(Hf)))
        @test representation(op, Hf, LazyRepr(input)) * v ≈ M * v
    end

    @test_throws DimensionMismatch representation(op, Hf, LazyRepr(zeros(dim(Hf) + 1)))

    @spin s 1 // 2
    H = tensor_product(Hf, hilbert_space(s))
    op = 5 * f[1]' * f[2] * s[:x] + 2 * s[:z] + f[1] + 1im

    Hcons = constrain_space(H, NumberConservation(2, [Hf]))
    L = representation(op, Hcons, :lazy; projection=true)
    M = representation(op, Hcons; projection=true)
    v = randn(dim(Hcons))
    Vm = randn(ComplexF64, dim(Hcons), 2)
    @test L * v ≈ M * v
    @test L * Vm ≈ M * Vm
    vcons_s = sprandn(ComplexF64, dim(Hcons), 0.5)
    Vcons_s = sprandn(ComplexF64, dim(Hcons), 2, 0.5)
    @test L * vcons_s ≈ M * vcons_s
    @test L * Vcons_s ≈ M * Vcons_s
    @test L' * Vcons_s ≈ M' * Vcons_s
    @test transpose(L) * Vcons_s ≈ transpose(M) * Vcons_s
    @test (L * vcons_s) isa SparseVector
    @test (L * Vcons_s) isa SparseMatrixCSC
    @test L' * vcons_s ≈ M' * vcons_s
    @test transpose(L) * vcons_s ≈ transpose(M) * vcons_s
    Lnoproj = representation(op, Hcons, :lazy)
    @test_throws ArgumentError Lnoproj * sparse(ones(dim(Hcons)))
    @test_throws ArgumentError Lnoproj * ones(dim(Hcons))
    @test_throws ArgumentError Lnoproj * ones(dim(Hcons), 2)
    @test Matrix(L) ≈ M
    @test Matrix(L') ≈ M'
    @test Matrix(transpose(L)) ≈ transpose(M)
    @test Matrix(transpose(L)') ≈ conj(M)
end

@testitem "lazy matrix_representation with dimensions wider than Int" begin
    using SparseArrays
    import FermionicHilbertSpaces: focknbr_from_site_indices, state_index
    @fermions f
    op = 0.3 * f[1]' * f[2] + 0.7im * f[2]' * f[3] + hc + 0.5 * f[1]' * f[1] + 0.2
    Hsmall = hilbert_space(f, 1:3)
    M = representation(op, Hsmall)
    for N in (63, 200)
        H = hilbert_space(f, 1:N)
        @test dim(H) == big(2)^N
        # the states of `Hsmall` with mode N occupied
        high = focknbr_from_site_indices((N,)).f
        inds = [state_index(FockNumber(high | (k - 1)), H) for k in 1:dim(Hsmall)]
        v = randn(ComplexF64, dim(Hsmall))
        x = sparsevec(inds, v, dim(H))
        L = representation(op, H, :lazy)
        for (Lop, Mop) in ((L, M), (L', M'), (transpose(L), transpose(M)))
            y = Lop * x
            @test y isa SparseVector && length(y) == dim(H) && nnz(y) == length(inds)
            @test [y[i] for i in inds] ≈ Mop * v
        end
    end
end

@testitem "lazy matrix_representation on product spaces" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: open_system, dissipator
    function lazy_matches(op, H)
        L = representation(op, H, :lazy)
        M = representation(op, H)
        v = randn(ComplexF64, dim(H))
        V = randn(ComplexF64, dim(H), 2)
        w = randn(ComplexF64, dim(H))
        xs = sprandn(ComplexF64, dim(H), 0.3)
        Lc = cache_operator(L, v)
        y = copy(w)
        mul!(y, Lc, v, 0.3, -0.7)
        return L * v ≈ M * v && L' * v ≈ M' * v && transpose(L) * v ≈ transpose(M) * v && conj(L) * v ≈ conj(M) * v &&
            L * V ≈ M * V && y ≈ 0.3 * (M * v) - 0.7 * w && concretize(L) ≈ M &&
            L * xs ≈ M * xs && L' * xs ≈ M' * xs && mul!(copy(w), L', xs, 0.3im, -0.7) ≈ 0.3im * (M' * xs) - 0.7 * w
    end

    @fermions c
    N = 3
    Hopen, _, _, left, right = open_system(c, 1:N)
    ham = sum((0.3n) * c[n]' * c[n] for n in 1:N) + sum(0.2im * c[n]' * c[n + 1] + hc for n in 1:(N - 1))
    liouvillian = -1im * (left(ham) - right(ham)) + sum(dissipator(sqrt(0.1n) * c[n]) for n in 1:N)
    @test lazy_matches(liouvillian + 0.5, Hopen)
    @test lazy_matches(left(ham) + 2, Hopen)

    @fermions f
    @boson b
    @spin s 1 // 2
    Hf = hilbert_space(f, 1:2)
    H = tensor_product(Hf, hilbert_space(b, 3), hilbert_space(s))
    op = f[1]' * f[2] + hc + 0.3 * b' * b + 0.2 * (b' + b) * f[1]' * f[1] + 0.7 * s[:z] +
         0.1 * s[:x] * (b + b') + 0.05 * f[2]' * f[2] * s[:z] * b' * b + f[1] * s[:x] + 0.4
    @test lazy_matches(op, H)

    # terms of one group with factors of different concrete types, and a complex coefficient
    H2 = tensor_product(hilbert_space(f, [1, :a]), hilbert_space(b, 3))
    @test lazy_matches(1im * f[1]' * f[:a] + hc + 0.5im * f[1]' * f[1] + b' * b, H2)
end

@testitem "lazy mul! on product spaces" begin
    using LinearAlgebra
    @spins S 1 // 2
    H = hilbert_space(S, 1:3)
    for op in (S[1][:z] * S[2][:z] + S[2][:z] * S[3][:z] + 0.5im * S[1][:x], S[1][:z] * S[3][:x])
        L = representation(op, H, :lazy)
        M = representation(op, H)
        v = randn(dim(H))
        w = randn(eltype(M), dim(H))
        @test mul!(similar(w), L, v) ≈ M * v
        @test mul!(copy(w), L, v, 0.3, -0.7) ≈ 0.3 * (M * v) - 0.7 * w
        @test mul!(similar(w), L', v) ≈ M' * v
    end
    op = S[1][:z] * S[2][:z] + 0.5im * S[1][:x]
    L, M = representation(op, H, :lazy), representation(op, H)
    v, w = randn(ComplexF64, dim(H)), randn(ComplexF64, dim(H))
    @test mul!(copy(w), L', v, 0.3im, -0.7) ≈ 0.3im * (M' * v) - 0.7 * w
end

@testitem "lazy matrix_representation chunking" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    using OhMyThreads
    import FermionicHilbertSpaces: TermChunking, StateChunking, NoChunking, cache_operator!

    scheduler = StaticScheduler(; nchunks=4)
    scheduler = SerialScheduler()

    @fermions f
    Hf = hilbert_space(f, 1:4)
    op = f[1]' * f[2] + 2im * f[3]' * f[4] + hc + 1.5

    M = representation(op, Hf)
    Ln = representation(op, Hf, :lazy; chunking=NoChunking())
    Lt = representation(op, Hf, :lazy; chunking=TermChunking(scheduler))
    Ls = representation(op, Hf, :lazy; chunking=StateChunking(scheduler))

    v = randn(ComplexF64, dim(Hf))
    V = randn(ComplexF64, dim(Hf), 3)
    vs = sprandn(ComplexF64, dim(Hf), 0.5)
    Vs = sprandn(ComplexF64, dim(Hf), 3, 0.5)

    cache_operator(Lt, v)
    cache_operator(Ls, V)

    @test Lt * v ≈ M * v
    @test Ls * v ≈ M * v
    @test (@inferred mul!(fill(NaN + 0im, dim(Hf)), Lt, v, 0.5, 0.0)) ≈ 0.5 * (M * v)
    @test (@inferred mul!(fill(NaN + 0im, dim(Hf)), Ls, v, 0.5, 0.0)) ≈ 0.5 * (M * v)
    for L in (Ln, Lt, Ls, representation(f[1]' * f[2], Hf, :lazy))
        @test mul!(zeros(ComplexF64, dim(Hf)), L', v, 0.5im, 0.0) ≈ 0.5im * (concretize(L)' * v)
    end
    @test Lt * V ≈ M * V
    @test Ls * V ≈ M * V
    @test Lt * vs ≈ M * vs
    @test Ls * vs ≈ M * vs
    @test Lt * Vs ≈ M * Vs
    @test Ls * Vs ≈ M * Vs
    @test concretize(Lt) ≈ M
    @test concretize(Ls) ≈ M

    @spin s 1 // 2
    Hs = hilbert_space(s)
    Hprod = tensor_product(Hf, Hs)
    op_prod = f[1]' * f[2] * s[:x] + f[3]' * f[4] * s[:z] + hc + 1im
    Mprod = representation(op_prod, Hprod)
    Ltprod = representation(op_prod, Hprod, :lazy; chunking=TermChunking(scheduler))
    Lsprod = representation(op_prod, Hprod, :lazy; chunking=StateChunking(scheduler))
    vprod = randn(ComplexF64, dim(Hprod))
    @test Ltprod * vprod ≈ Mprod * vprod
    @test Lsprod * vprod ≈ Mprod * vprod
    @test representation(op_prod, Hprod, :lazy; chunking = StateChunking(StaticScheduler(; nchunks = 3))) * vprod ≈ Mprod * vprod

    Hcons = constrain_space(Hprod, NumberConservation(2, [Hf]))
    vcons = randn(ComplexF64, dim(Hcons))
    Mcons = representation(op_prod, Hcons; projection=true)
    Mv = Mcons * vcons
    for concretizer in [FermionicHilbertSpaces.NoConcretizer(), FermionicHilbertSpaces.TupleConcretizer()]
        Lcons = representation(op_prod, Hcons, :lazy; projection=true, chunking=NoChunking(), concretizer)
        Ltcons = representation(op_prod, Hcons, :lazy; projection=true, chunking=TermChunking(scheduler), concretizer)
        Lscons = representation(op_prod, Hcons, :lazy; projection=true, chunking=StateChunking(scheduler), concretizer)
        @test Lcons * vcons ≈ Mv
        @test Ltcons * vcons ≈ Mv
        @test Lscons * vcons ≈ Mv
    end
end

