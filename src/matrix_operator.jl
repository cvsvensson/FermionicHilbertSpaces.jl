"""
    MatrixOperator(M, H)

The symbolic operator with matrix `M` on the space `H`. It is represented on any space where the groups of `H` lie in
one factor, and on part of a fermionic group `M` must have definite parity.

```julia
Jm = representation(f[1], Heig, Matrix) .* sqrt.(γ.(E .- transpose(E)))
L = representation(-1im * (left(ham) - right(ham)) + dissipator(MatrixOperator(Jm, Heig)), Hlr)
```
"""
struct MatrixOperator{M <: AbstractMatrix, H <: AbstractHilbertSpace} <: AbstractSym
    matrix::M
    space::H
    function MatrixOperator(matrix::AbstractMatrix, space::AbstractHilbertSpace)
        size(matrix) == (dim(space), dim(space)) || throw(DimensionMismatch("The matrix must have the size $((dim(space), dim(space))) of the space, got $(size(matrix))."))
        return new{typeof(matrix), typeof(space)}(matrix, space)
    end
end
Base.:(==)(a::MatrixOperator, b::MatrixOperator) = a.space == b.space && a.matrix == b.matrix
Base.hash(a::MatrixOperator, h::UInt) = hash(a.matrix, hash(a.space, h))
Base.adjoint(a::MatrixOperator) = MatrixOperator(a.matrix', a.space)
Base.show(io::IO, a::MatrixOperator) = print(io, "MatrixOperator(", summary(a.matrix), ")")
symbolic_group(a::MatrixOperator) = _support(group_ids(a.space))
add_tag(a::MatrixOperator, tag) = MatrixOperator(a.matrix, add_tag(a.space, tag))
mat_eltype(::Type{<:MatrixOperator{M}}) where {M} = eltype(M)

# Operators of groups that overlap with those of the matrix's space have no rule with it, and others commute with it.
_commute_or_keep(a, b, x::MatrixOperator, y) = _overlaps(symbolic_group(x), symbolic_group(y)) ? nothing : (_group_isless(b, a) ? Swap(1) : nothing)
NonCommutativeProducts.mul_effect(a::MatrixOperator, b::AbstractSym) = _commute_or_keep(a, b, a, b)
NonCommutativeProducts.mul_effect(a::AbstractSym, b::MatrixOperator) = _commute_or_keep(a, b, b, a)
NonCommutativeProducts.mul_effect(a::MatrixOperator, b::MatrixOperator) = a.space == b.space ? MatrixOperator(a.matrix * b.matrix, a.space) : _commute_or_keep(a, b, a, b)
function NonCommutativeProducts.mul_effect(a::MatrixOperator, s::SymbolicState)
    a.space == s.space && has_ket(s) || return _commute_or_keep(a, s, a, s)
    column = sparse(a.matrix[:, state_index(s.ket, s.space)])
    return AddTerms([_scaled_state(c, SymbolicState(s.space, basisstate(i, s.space), s.bra)) for (i, c) in zip(findnz(column)...)])
end
function NonCommutativeProducts.mul_effect(s::SymbolicState, a::MatrixOperator)
    a.space == s.space && has_bra(s) || return _commute_or_keep(s, a, a, s)
    row = sparse(a.matrix[state_index(s.bra, s.space), :])
    return AddTerms([_scaled_state(c, SymbolicState(s.space, s.ket, basisstate(i, s.space))) for (i, c) in zip(findnz(row)...)])
end
# Products with kets are formed factor by factor, as the matrix maps a basis state to a superposition
_ket_index(::MatrixOperator, factors) = nothing

_has_matrix_factor(op::NCMul) = any(x -> x isa MatrixOperator, op.factors)
_has_matrix_factor(op::NCAdd) = any(_has_matrix_factor, keys(op.dict))
_has_matrix_factor(op) = op isa MatrixOperator
# Terms take the route of `KronTerm`s when a factor of the operator or a factor of the space acts by a matrix
_by_matrices(op, space) = _acts_by_matrices(space) || _has_matrix_factor(op)

# The matrix of `op` on the slot `H` is the product of the matrices of its factors that act by matrices and of the runs
# of other factors between them, whose matrices are `run_matrix(run)`.
function _runs_matrix(run_matrix, op::NCMul, H)
    matrices = []
    run = []
    for x in op.factors
        _is_matrix_factor(x, H) || (push!(run, x); continue)
        isempty(run) || push!(matrices, run_matrix(NCMul(1, run)))
        run = []
        push!(matrices, _factor_matrix(x, H))
    end
    isempty(run) || push!(matrices, run_matrix(NCMul(1, run)))
    return op.coeff * prod(matrices)
end
_is_matrix_factor(x::MatrixOperator, H) = true
_is_matrix_factor(x, H) = false
_factor_matrix(x::MatrixOperator, H) = x.space == H ? x.matrix : embed(x.matrix, x.space => H)
function _matrix_factors_matrix(op::NCMul, H, repr; projection = false, _...)
    H isa TransposedSpace && return transpose(_matrix_factors_matrix(op, parent(H), repr; projection))
    return _runs_matrix(run -> _term_matrix_representation(run, H, repr, NoChunking(); projection), op, H)
end

@testitem "MatrixOperator" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: MatrixOperator, eigenbasis, energies, open_system, dissipator
    @fermions f
    @boson b
    ham = sum(0.3k * f[k]'f[k] for k in 1:3) + sum(0.5 * f[k]'f[k + 1] + hc for k in 1:2) + 0.7 * f[1]'f[1] * f[2]'f[2]
    N = sum(f[k]'f[k] for k in 1:3)
    Heig = eigenbasis(ham, hilbert_space(f, 1:3, NumberConservation()))
    E = energies(Heig)
    Jm = Matrix(representation(f[1], Heig)) .* sqrt.(exp.(-max.(E .- transpose(E), 0)))
    Hm = Matrix(representation(ham, Heig))
    Id = Matrix(I, size(Hm))
    J = MatrixOperator(Jm, Heig)
    Hfull, Hl, Hr, left, right = open_system(Heig)
    for Hlr in (Hfull, tensor_product(Hl, Hr; constraint = left(N) - right(N) => 0))
        # the superoperator ρ ↦ AρB
        sup(A, B) = tensor_product((A, transpose(B)), (Hl, Hr) => Hlr)
        Lref = -1im * (sup(Hm, Id) - sup(Id, Hm)) + sup(Jm, Jm') - (sup(Jm'Jm, Id) + sup(Id, Jm'Jm)) / 2
        @test representation(-1im * (left(ham) - right(ham)) + dissipator(J), Hlr) ≈ Lref
    end

    H = hilbert_space(f, 1:2)
    M = rand(4, 4)
    A = MatrixOperator(M, H)
    @test representation(A, H) ≈ M
    @test representation(A * f[1], H) ≈ M * representation(f[1], H)
    Hb = hilbert_space(b, 3)
    @test representation(A * b', tensor_product(H, Hb)) ≈ kron(Matrix(representation(b', Hb)), M)
    Hn = hilbert_space(f, 1:3, NumberConservation(1))
    Mn = rand(3, 3)
    @test representation(MatrixOperator(Mn, Hn), Hn) ≈ Mn
    H1 = hilbert_space(f, 1:1)
    m = MatrixOperator([0.2 0; 0 -1], H1)
    @test representation(m, Hn) ≈ representation(0.2 * f[1] * f[1]' - f[1]'f[1], Hn)
    @test representation(m, Hn, :lazy) * ones(3) ≈ representation(0.2 * f[1] * f[1]' - f[1]'f[1], Hn) * ones(3)

    k = Kets(H)
    @test representation(A * k[2], H) ≈ M[:, 2]
    P = tensor_product(H, Hb)
    @test_throws ArgumentError representation(MatrixOperator(rand(12, 12), P), P)
end
