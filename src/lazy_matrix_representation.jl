"""
    LazyOperator{O,S,T,P}

A matrix-free (lazy) representation of a symbolic operator on a Hilbert space.
Acts directly on vectors and matrices via without constructing a sparse matrix.

Constructed via `matrix_representation(op, space; lazy=true)`. `LazyOperator` conforms to
the `SciMLOperators.AbstractSciMLOperator` interface.
"""
struct LazyOperator{O,S,T,P} <: SciMLOperators.AbstractSciMLOperator{T}
    op::O
    space::S
    precomp::P
    projection::Bool
    conjugate::Bool
    transpose::Bool
    hermitian::Bool
end

function _show_lazy_operator_expression(io::IO, L::LazyOperator)
    show(IOContext(io, :compact => true), L.op)
    L.transpose && print(io, "^T")
    L.conjugate && print(io, "^*")
end

function Base.show(io::IO, L::LazyOperator)
    _show_lazy_operator_expression(io, L)
    print(io, " acting on ")
    show(IOContext(io, :compact => true), L.space)
end

function LazyOperator(op::O, space::S, precomp::P=_precomputation_before_operator_application(op, space); projection=false, conjugate=false, transpose=false, hermitian=_ishermitian(op), T=mat_eltype(op)
) where {O,S,P}
    LazyOperator{O,S,T,P}(op, space, precomp, projection, conjugate, transpose, hermitian)
end
_ishermitian(x::NCMul) = iszero(x - hc)
_ishermitian(x::NCAdd) = iszero(x - hc)
function _ishermitian(x::OperatorSequence)
    _ishermitian(prod(x.ops))
end
mat_eltype(x::OperatorSequence) = promote_type(map(mat_eltype, x.ops)...)

Base.size(L::LazyOperator) = (dim(L.space), dim(L.space))
Base.size(L::LazyOperator, i::Int) = size(L)[i]
Base.eltype(::LazyOperator{O,S,T,P}) where {O,S,T,P} = T
Base.conj(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=!L.conjugate, transpose=L.transpose, hermitian=L.hermitian)
Base.adjoint(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=!L.conjugate, transpose=!L.transpose, hermitian=L.hermitian)
Base.transpose(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=L.conjugate, transpose=!L.transpose, hermitian=L.hermitian)

SciMLOperators.isconstant(::LazyOperator) = true
SciMLOperators.islinear(::LazyOperator) = true
SciMLOperators.isconvertible(::LazyOperator) = true
SciMLOperators.has_adjoint(::LazyOperator) = true
SciMLOperators.has_mul(::LazyOperator) = true
SciMLOperators.has_mul!(::LazyOperator) = true
SciMLOperators.update_coefficients(L::LazyOperator, u, p, t; kwargs...) = L
SciMLOperators.update_coefficients!(L::LazyOperator, u, p, t; kwargs...) = nothing
LinearAlgebra.ishermitian(L::LazyOperator) = L.hermitian

function _eager_matrix_representation(L::LazyOperator{<:NCMul})
    return _term_matrix_representation(L.op, L.space, EagerRepr(); projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:OperatorSequence})
    return _factorized_term_matrix_representation(L.op, L.space, EagerRepr(); projection=L.projection)
end

function _eager_matrix_representation(L::LazyOperator{<:NCAdd})
    return _matrix_representation_single_space(L.op, L.space, EagerRepr(); projection=L.projection)
end

function Base.convert(::Type{AbstractMatrix}, L::LazyOperator)
    sparse(L)
end

function SparseArrays.sparse(L::LazyOperator)
    M = _eager_matrix_representation(L)
    L.transpose && (M = transpose(M))
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
function _apply_single_term!(y::AbstractVector, x::AbstractVector, space, term, precomp, _coeff, conjugate, transpose, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for n in eachindex(basisstates(space))
        if !transpose
            xn = x[n]
            iszero(xn) && continue
        end
        outind, _amp = _apply_local_operators_index(term, n, space, precomp)
        if !iszero(_amp)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !projection || !iszero(outind)
                if !transpose
                    y[outind] += amp * xn
                else
                    y[n] += amp * x[outind]
                end
            end
        end
    end
end

function _apply_single_term!(y::AbstractMatrix, x::AbstractMatrix, space, term, precomp, _coeff, conjugate, transpose, projection)
    coeff = (conjugate ? conj(_coeff) : _coeff)
    for n in eachindex(basisstates(space))
        outind, _amp = _apply_local_operators_index(term, n, space, precomp)
        if !iszero(_amp)
            amp = (conjugate ? conj(_amp) : _amp) * coeff
            if !projection || !iszero(outind)
                if !transpose
                    @views y[outind, :] .+= amp .* x[n, :]
                else
                    @views y[n, :] .+= amp .* x[outind, :]
                end
            end
        end
    end
end

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCMul}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    _apply_single_term!(y, x, L.space, L.op, L.precomp, α, L.conjugate, L.transpose, L.projection)
    return y
end

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:OperatorSequence}, x::AbstractVecOrMat, α, β)
    # This is for productspaces, where ops is a list of operators applying to each factor space
    rmul!(y, β)
    coeff = prod(op.coeff for op in L.op.ops)
    _apply_single_term!(y, x, L.space, L.op, L.precomp, coeff * α, L.conjugate, L.transpose, L.projection)
    return y
end

function lazy_mul!(y::AbstractVecOrMat, L::LazyOperator{<:NCAdd}, x::AbstractVecOrMat, α, β)
    rmul!(y, β)
    op = L.op
    for (term, coeff) in op.dict
        precomp = _precomputation_before_operator_application(term, L.space)
        _apply_single_term!(y, x, L.space, term, precomp, coeff * α, L.conjugate, L.transpose, L.projection)
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
function _term_matrix_representation(op::NCMul, H::AbstractHilbertSpace, ::LazyRepr; kwargs...)
    LazyOperator(op, H; kwargs...)
end

function _factorized_term_matrix_representation(ops::OperatorSequence, H, ::LazyRepr; kwargs...)
    LazyOperator(ops, H; kwargs...)
end

function _matrix_representation_single_space(op::NCAdd, space, ::LazyRepr; kwargs...)
    LazyOperator(op, space; kwargs...)
end

@testitem "lazy matrix_representation" begin
    using FermionicHilbertSpaces.SciMLOperators
    using LinearAlgebra, SparseArrays
    @fermions f
    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im
    L = matrix_representation(op, Hf; lazy=true)
    M = matrix_representation(op, Hf)
    v = randn(dim(Hf))
    w = randn(ComplexF64, dim(Hf))
    Vm = randn(dim(Hf), 3)
    Wm = randn(ComplexF64, dim(Hf), 3)
    α = 0.7
    β = -0.2
    @test SciMLOperators.isconstant(L)
    @test SciMLOperators.islinear(L)
    @test SciMLOperators.isconvertible(L)
    @test SciMLOperators.has_mul(L)
    @test SciMLOperators.has_mul!(L)
    @test SciMLOperators.has_adjoint(L)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v
    @test L * Vm ≈ M * Vm
    @test L(v, nothing, nothing, nothing) ≈ M * v
    @test L(Vm, nothing, nothing, nothing) ≈ M * Vm
    @test L(copy(w), v, nothing, nothing, nothing) ≈ M * v
    @test L(copy(Wm), Vm, nothing, nothing, nothing) ≈ M * Vm
    @test L(copy(w), v, nothing, nothing, nothing, α, β) ≈ α .* (M * v) .+ β .* w
    @test L(copy(Wm), Vm, nothing, nothing, nothing, α, β) ≈ α .* (M * Vm) .+ β .* Wm
    @test (L + one(L)) * v ≈ (M + I) * v
    @test (2 * L) * v ≈ 2 * (M * v)
    @test sparse(L) == M
    @test !ishermitian(L)
    @test ishermitian(matrix_representation(op + hc, Hf; lazy=true))

    @spin s 1 // 2
    Hs = hilbert_space(s)
    H = tensor_product(Hf, Hs)
    op = f[1]' * f[2] * s[:x] + s[:z] + f[1] + 1im
    L = matrix_representation(op, H; lazy=true)
    M = matrix_representation(op, H)
    v = randn(dim(H))
    Vm = randn(dim(H), 2)
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v
    @test L * Vm ≈ M * Vm

    Hcons = constrain_space(H, NumberConservation(2, [Hf]))
    L = matrix_representation(op, Hcons; lazy=true, projection=true)
    M = matrix_representation(op, Hcons; projection=true)
    v = randn(dim(Hcons))
    Vm = randn(ComplexF64, dim(Hcons), 2)
    @test L * v ≈ M * v
    @test L * Vm ≈ M * Vm
    @test Matrix(L) ≈ M
    @test Matrix(L') ≈ M'
    @test Matrix(transpose(L)) ≈ transpose(M)
    @test Matrix(transpose(L)') ≈ conj(M)

    @test !ishermitian(L)
    @test ishermitian(matrix_representation(op + hc, Hcons; lazy=true, projection=true))
end

