"""
    LazyOperator{O,S,T}

A matrix-free (lazy) representation of a symbolic operator on a Hilbert space.
Acts directly on vectors via `mul!` without constructing a sparse matrix.

Constructed via `lazy_matrix_representation(op, space)` or
`matrix_representation(op, space; lazy=true)`.
"""
struct LazyOperator{O,S,T,P}
    op::O
    space::S
    precomp::P
    projection::Bool
    conjugate::Bool
    transpose::Bool
end

function LazyOperator(op::O, space::S, precomp::P=_precomputation_before_operator_application(op, space); projection=false, conjugate=false, transpose=false) where {O,S,P}
    T = mat_eltype(op)
    LazyOperator{O,S,T,P}(op, space, precomp, projection, conjugate, transpose)
end

Base.size(L::LazyOperator) = (dim(L.space), dim(L.space))
Base.size(L::LazyOperator, i::Int) = size(L)[i]
Base.eltype(::LazyOperator{O,S,T,P}) where {O,S,T,P} = T
Base.adjoint(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=!L.conjugate, transpose=!L.transpose)
Base.transpose(L::LazyOperator) = LazyOperator(L.op, L.space, L.precomp; projection=L.projection, conjugate=L.conjugate, transpose=!L.transpose)

function LinearAlgebra.mul!(y::AbstractVector, L::LazyOperator, x::AbstractVector, α=true, β=false)
    lazy_mul!(y, L, x, α, β)
end
function lazy_mul!(y::AbstractVector, L::LazyOperator{<:NCMul}, x::AbstractVector, α, β)
    y .= β .* y
    op = L.op
    space = L.space
    precomp = L.precomp
    projection = L.projection
    conjugate = L.conjugate
    transpose = L.transpose
    for (n, state) in enumerate(basisstates(space))
        if !transpose
            xn = x[n]
            iszero(xn) && continue
        end
        newstates, newamps = apply_local_operators(op, state, space, precomp)
        for (newstate, _amp) in zip(newstates, newamps)
            if !iszero(_amp)
                amp = conjugate ? conj(_amp) : _amp
                outind = state_index(newstate, space)
                if !projection || !ismissing(outind)
                    if !transpose
                        y[outind] += amp * xn * α
                    else
                        y[n] += amp * x[outind] * α
                    end
                end
            end
        end
    end
    return y
end

function lazy_mul!(y::AbstractVector, L::LazyOperator{<:Vector{<:NCMul}}, x::AbstractVector, α, β)
    # This is for productspaces, where ops is a list of operators applying to each factor space
    y .= β .* y
    ops = L.op
    space = L.space
    precomp = L.precomp
    coeff = prod(op.coeff for op in ops)
    projection = L.projection
    conjugate = L.conjugate
    transpose = L.transpose
    for (n, state) in enumerate(basisstates(space))
        if !transpose
            xn = x[n]
            iszero(xn) && continue
        end
        newstates, newamps = apply_local_operators(ops, state, space, precomp)
        for (newstate, _amp) in zip(newstates, newamps)
            if !iszero(_amp)
                amp = conjugate ? conj(_amp) : _amp
                outind = state_index(newstate, space)
                if !projection || !ismissing(outind)
                    if !transpose
                        y[outind] += amp * xn * α * coeff
                    else
                        y[n] += amp * x[outind] * α * coeff
                    end
                end
            end
        end
    end
    return y
end

function lazy_mul!(y::AbstractVector, L::LazyOperator{<:NCAdd}, x::AbstractVector, α, β)
    y .= β .* y
    op = L.op
    space = L.space
    projection = L.projection
    conjugate = L.conjugate
    transpose = L.transpose
    for (term, coeff) in op.dict
        precomp = _precomputation_before_operator_application(term, space)
        for (n, state) in enumerate(basisstates(space))
            if !transpose
                xn = x[n]
                iszero(xn) && continue
            end
            newstates, newamps = apply_local_operators(term, state, space, precomp)
            for (newstate, _amp) in zip(newstates, newamps)
                if !iszero(_amp)
                    amp = conjugate ? conj(_amp) : _amp
                    outind = state_index(newstate, space)
                    if !projection || !ismissing(outind)
                        if !transpose
                            y[outind] += coeff * amp * xn * α
                        else
                            y[n] += coeff * amp * x[outind] * α
                        end
                    end
                end
            end
        end
    end
    if !iszero(op.coeff) && !iszero(α)
        coeff = conjugate ? conj(op.coeff) : op.coeff
        y .+= coeff .* x .* α
    end
    return y
end

function Base.:*(L::LazyOperator, x::AbstractVector)
    T = promote_type(eltype(L), eltype(x))
    y = similar(x, T)
    mul!(y, L, x)
end
function _term_matrix_representation(op::NCMul, H::AbstractHilbertSpace, ::LazyRepr; kwargs...)
    LazyOperator(op, H)
end

function _matrix_representation_single_space(op::NCAdd, space, ::LazyRepr; kwargs...)
    LazyOperator(op, space)
end


@testitem "lazy matrix_representation" begin
    using LinearAlgebra
    @fermions f
    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im
    L = matrix_representation(op, Hf; lazy=true)
    M = matrix_representation(op, Hf)
    v = randn(dim(Hf))
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v

    using LinearMaps
    @spin s 1 // 2
    Hs = hilbert_space(s)
    H = tensor_product(Hf, Hs)
    op = op * s[:x] + s[:z]
    L = matrix_representation(op, H; lazy=true)
    M = matrix_representation(op, H)
    v = randn(dim(H))
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v

    Hcons = constrain_space(H, NumberConservation(2, [Hf]))
    L = matrix_representation(op, Hcons; lazy=true)
    M = matrix_representation(op, Hcons)
    v = randn(dim(Hcons))
    @test L * v ≈ M * v
    @test Matrix(L) ≈ M
    @test Matrix(L') ≈ M'
    @test Matrix(transpose(L)) ≈ transpose(M)
    @test Matrix(transpose(L)') ≈ conj(M)
end

