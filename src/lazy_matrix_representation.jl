"""
    LazyOperator{O,S,T,P}

A matrix-free (lazy) representation of a symbolic operator on a Hilbert space.
Acts directly on vectors and matrices via without constructing a sparse matrix.

Constructed via `matrix_representation(op, space, :lazy)`. `LazyOperator` conforms to
the `SciMLOperators.AbstractSciMLOperator` interface.
"""
struct LazyOperator{O,S,T,P}
    op::O
    space::S
    precomp::P
    projection::Bool
    conjugate::Bool
    ishermitian::Bool
    size::Tuple{Int,Int}
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

function LazyOperator(op::O, space::S, precomp::P=_precomputation_before_operator_application(op, space); projection=false, conjugate=false, ishermitian=_ishermitian(op), T=mat_eltype(op),
) where {O,S,P}
    LazyOperator{O,S,T,P}(op, space, precomp, projection, conjugate, ishermitian, (dim(space), dim(space)))
end
function scimloperator(L::LazyOperator, input=Vector{eltype(L)}(undef, dim(L.space)), output=input; kwargs...)
    SciMLOperators.FunctionOperator(L, input, output; ishermitian=ishermitian(L), op_adjoint=adjoint(L), isconstant=true, T=eltype(L), islinear=true, batch=true, kwargs...)
end

(L::LazyOperator)(w, v, u, p, t, α, β) = mul!(w, L, v, α, β)
(L::LazyOperator)(w, v, u, p, t) = mul!(w, L, v)
(L::LazyOperator)(v, u, p, t) = L * v

_ishermitian(x::NCMul) = iszero(x - hc)
_ishermitian(x::NCAdd) = iszero(x - hc)
function _ishermitian(x::ProductOperator)
    _ishermitian(prod(skipmissing(x.ops)))
end

Base.size(L::LazyOperator) = L.size
Base.size(L::LazyOperator, i::Int) = size(L)[i]
Base.eltype(::LazyOperator{O,S,T,P}) where {O,S,T,P} = T
Base.conj(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
function Base.adjoint(L::LazyOperator)
    LazyOperator(L.op, TransposedSpace(L.space), L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
end
function Base.transpose(L::LazyOperator)
    LazyOperator(L.op, TransposedSpace(L.space), L.precomp; projection=L.projection, conjugate=L.conjugate, ishermitian=L.ishermitian)
end
function Base.adjoint(L::LazyOperator{<:ProductOperator{C}}) where C
    newop = ProductOperator{C}(L.op.ops, map(TransposedSpace, L.op.spaces))
    LazyOperator(newop, TransposedSpace(L.space), L.precomp; projection=L.projection, conjugate=!L.conjugate, ishermitian=L.ishermitian)
end
function Base.transpose(L::LazyOperator{<:ProductOperator{C}}) where C
    newop = ProductOperator{C}(L.op.ops, map(TransposedSpace, L.op.spaces))
    LazyOperator(newop, TransposedSpace(L.space), L.precomp; projection=L.projection, conjugate=L.conjugate, ishermitian=L.ishermitian)
end
Base.transpose(L::SciMLOperators.FunctionOperator{<:Any,<:Any,<:Any,<:Any,<:LazyOperator}) = scimloperator(transpose(L.op))
LinearAlgebra.ishermitian(L::LazyOperator) = L.ishermitian

function _eager_matrix_representation(L::LazyOperator{<:NCMul})
    return _term_matrix_representation(L.op, L.space, EagerSparseRepr(); projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:ProductOperator})
    return _factorized_term_matrix_representation(L.op, L.space, EagerSparseRepr(); projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:NCAdd})
    return _matrix_representation_single_space(L.op, L.space, EagerSparseRepr(); projection=L.projection)
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
    size(X, 1) == size(L, 2) || throw(DimensionMismatch("input has size $(size(X)) but operator has size $(size(L))"))
    expected_size = (size(L, 1), size(X, 2))
    size(Y) == expected_size || throw(DimensionMismatch("output has size $(size(Y)) but expected $expected_size"))
    lazy_mul!(Y, L, X, α, β)
    return Y
end
function _apply_single_term!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for (n, state) in enumerate(basisstates(space))
        xn = x[n] * coeff
        iszero(xn) && continue
        state, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(state, space)
            if !projection || !iszero(outind)
                amp = (conjugate ? conj(_amp) : _amp)
                y[outind] += amp * xn
            end
        end
    end
end

function _apply_single_term!(y::AbstractVector, x::SparseArrays.AbstractSparseVector, space, term, precomp, _coeff, conjugate, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    inds, vals = findnz(x)
    for (n, xn) in zip(inds, vals)
        state = basisstate(n, space)
        state, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(state, space)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !projection || !iszero(outind)
                y[outind] += amp * xn
            end
        end
    end
end

function _apply_single_term!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for (n, state) in enumerate(basisstates(space))
        newstate, _amp = _apply_local_operators(term, state, space, precomp)
        if !iszero(_amp)
            outind = state_index(newstate, space)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !projection || !iszero(outind)
                @views y[outind, :] .+= amp .* x[n, :]
            end
        end
    end
end

function _apply_single_term!(y::AbstractMatrix, x::SparseArrays.SparseMatrixCSC, space, term, precomp, _coeff, conjugate, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    rows = rowvals(x)
    vals = nonzeros(x)
    for col in axes(x, 2)
        for ptr in nzrange(x, col)
            n = rows[ptr]
            xn = vals[ptr]
            state = basisstate(n, space)
            newstate, _amp = _apply_local_operators(term, state, space, precomp)
            if !iszero(_amp)
                outind = state_index(newstate, space)
                amp = (conjugate ? conj(_amp) : _amp) * coeff
                if !projection || !iszero(outind)
                    y[outind, col] += amp * xn
                end
            end
        end
    end
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:NCMul}, x::AbstractVecOrMat, α, β) where T
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection)
    return y
end

function lazy_mul!(y::AbstractVecOrMat{T}, L::LazyOperator{<:ProductOperator}, x::AbstractVecOrMat, α, β) where T
    # This is for productspaces, where ops is a list of operators applying to each factor space
    if iszero(β)
        fill!(y, zero(T))
    else
        rmul!(y, β)
    end
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.projection)
    return y
end

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCAdd}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    op = L.op
    for (term, coeff) in op.dict
        precomp = _precomputation_before_operator_application(term, L.space)
        _apply_single_term!(y, x, L.space, term, precomp, coeff * α, L.conjugate, L.projection)
    end
    if !iszero(op.coeff) && !iszero(α)
        scalar_coeff = α * (L.conjugate ? conj(op.coeff) : op.coeff)
        y .+= scalar_coeff .* x
    end
    return y
end

function Base.:*(L::LazyOperator, x::AbstractVecOrMat)
    T = promote_type(eltype(L), eltype(x))
    y = similar(x, T)
    return mul!(y, L, x)
end
get_input(::LazyRepr{Missing}, H) = Vector{mat_eltype(H)}(undef, dim(H))
function get_input(rep::LazyRepr{<:AbstractArray}, H)
    dim(H) == size(rep.input, 1) || throw(DimensionMismatch("input has size $(size(rep.input)) but expected $(dim(H))"))
    return rep.input
end
function get_input(rep::LazyRepr{Symbol}, H)
    rep.input == :dense && return Vector{mat_eltype(H)}(undef, dim(H))
    rep.input == :sparse && return spzeros(mat_eltype(H), dim(H))
end

function _term_matrix_representation(op::NCMul, H::AbstractHilbertSpace, rep::LazyRepr; kwargs...)
    scimloperator(LazyOperator(op, H; kwargs...), get_input(rep, H))
end
function _factorized_term_matrix_representation(ops::ProductOperator, H, rep::LazyRepr; kwargs...)
    scimloperator(LazyOperator(ops, H; kwargs...), get_input(rep, H))
end
function _matrix_representation_single_space(op::NCAdd, H, rep::LazyRepr; kwargs...)
    scimloperator(LazyOperator(op, H; kwargs...), get_input(rep, H))
end

@testitem "lazy matrix_representation" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    @fermions f
    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im
    L = matrix_representation(op, Hf, :lazy)
    M = matrix_representation(op, Hf)
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
    @test L * Vm ≈ M * Vm
    t = 0
    u = p = nothing
    @test L(v, u, p, t) ≈ M * v
    @test L(Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t) ≈ M * v
    @test L(copy(Wm), Vm, u, p, t) ≈ M * Vm
    @test L(copy(w), v, u, p, t, α, β) ≈ α .* (M * v) .+ β .* w
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
    @test ishermitian(matrix_representation(op + hc, Hf, :lazy))

    import FermionicHilbertSpaces: LazyRepr
    @test matrix_representation(op, Hf, :lazy).cache[1] isa Vector
    @test matrix_representation(op, Hf, LazyRepr(:dense)).cache[1] isa Vector
    @test matrix_representation(op, Hf, LazyRepr(:sparse)).cache[1] isa SparseVector
    dense_cache = zeros(ComplexF64, dim(Hf))
    sparse_cache = spzeros(ComplexF64, dim(Hf))
    symbol_cache = [:a for _ in 1:dim(Hf)]
    @test matrix_representation(op, Hf, LazyRepr(dense_cache)).cache[1] isa typeof(dense_cache)
    @test matrix_representation(op, Hf, LazyRepr(sparse_cache)).cache[1] isa typeof(sparse_cache)
    @test matrix_representation(op, Hf, LazyRepr(symbol_cache)).cache[1] isa typeof(symbol_cache)

    @test_throws DimensionMismatch matrix_representation(op, Hf, LazyRepr(zeros(dim(Hf) + 1)))

    @spin s 1 // 2
    Hs = hilbert_space(s)
    H = tensor_product(Hf, Hs)
    op = 5 * f[1]' * f[2] * s[:x] + 2 * s[:z] + f[1] + 1im
    L = matrix_representation(op, H, :lazy)
    M = matrix_representation(op, H)
    v = randn(dim(H))
    Vm = randn(dim(H), 2)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v
    @test L * Vm ≈ M * Vm

    Hcons = constrain_space(H, NumberConservation(2, [Hf]))
    L = matrix_representation(op, Hcons, :lazy; projection=true)
    M = matrix_representation(op, Hcons; projection=true)
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
    @test Matrix(L) ≈ M
    @test Matrix(L') ≈ M'
    @test Matrix(transpose(L)) ≈ transpose(M)
    @test Matrix(transpose(L)') ≈ conj(M)

    # @test !ishermitian(L)
    # @test ishermitian(matrix_representation(op + hc, Hcons; lazy=true, projection=true)) #SciMLOperators doesn't check termwise ishermitian for an added operator
end

