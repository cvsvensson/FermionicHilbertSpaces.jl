abstract type AbstractSym end
abstract type AbstractFermionSym <: AbstractSym end

function mat_eltype(::NCAdd{C,NCMul{C2,S,F}}) where {C,C2,S,F}
    promote_type(C, mat_eltype(S))
end
function mat_eltype(::NCMul{C,S,F}) where {C,S,F}
    promote_type(C, mat_eltype(S))
end
mat_eltype(::Type{NCMul{C,S,F}}) where {C,S,F} = promote_type(C, mat_eltype(S))
mat_eltype(::S) where {S} = mat_eltype(S)
mat_eltype(::Type{S}) where {S} = Float64 #Default fallback. Could give errors if a complex number is expected. Override it for specific types if needed.

function operator_indices_and_amplitudes!((outinds, ininds, amps), op, H::AbstractHilbertSpace; kwargs...)
    return operator_indices_and_amplitudes_generic!((outinds, ininds, amps), op, H; kwargs...)
end
_precomputation_before_operator_application(factors, space) = nothing
function operator_indices_and_amplitudes_generic!((outinds, ininds, amps), op::NCMul, space::AbstractHilbertSpace; projection=false)
    precomp = _precomputation_before_operator_application(op, space)
    for (n, state) in enumerate(basisstates(space))
        newstates, newamps = apply_local_operators(op, state, space, precomp)
        for (newstate, amp) in zip(newstates, newamps)
            if !iszero(amp)
                outind = state_index(newstate, space)
                if !projection || !ismissing(outind)
                    push!(outinds, outind)
                    push!(amps, amp)
                    push!(ininds, n)
                end
            end
        end
    end
    return (outinds, ininds, amps)
end

## 
remove_identity(a::NCMul) = a
remove_identity(a::NCAdd) = NCAdd(zero(a.coeff), a.dict)

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

"""
    partition_factors_by_basis(factors::Vector, bases::Vector)

Partition a vector of operator factors into groups by their symbolic basis.
"""
function partition_factors_by_basis(factors::Vector, bases)
    partition = map(bases) do basis
        filter(==(symbolic_group(basis)) ∘ symbolic_group, factors)
    end
    sum(length, partition) == length(factors) || throw(ArgumentError("Not all factors were assigned to a basis."))
    return partition
end


function _matrix_representation(op::NCMul, bases, space::ProductSpace; kwargs...)
    spaces = factors(space)
    partitioned = partition_factors_by_basis(op.factors, bases)
    matrices = map(partitioned, spaces) do factors, space
        if isempty(factors)
            return I(dim(space))
        else
            _term_matrix_representation(NCMul(1, factors), space; kwargs...)
        end
    end
    length(spaces) == 1 && return op.coeff * only(matrices)
    op.coeff * kron(reverse(matrices)...)
end
function _matrix_representation(op::NCMul, bases, space; kwargs...)
    if isempty(op.factors)
        return op.coeff * I(dim(space))
    else
        if length(bases) > 1
            partition = partition_factors_by_basis(op.factors, bases)
            return op.coeff * _factorized_term_matrix_representation(map(ops -> NCMul(1, ops), partition), space; kwargs...)
        else
            return _term_matrix_representation(op, space; kwargs...)
        end
    end
end
# function _matrix_representation(op::NCAdd, bases, space::Union{<:AbstractAtomicHilbertSpace,<:AbstractClusterHilbertSpace}; kwargs...)
#     return _matrix_representation_single_space(op, space; kwargs...)
# end
# function _matrix_representation(op::NCAdd, bases, space::ProductSpace; kwargs...)
#     sum(_matrix_representation(term, bases, space; kwargs...) for term in NCterms(op)) + op.coeff * I(dim(space))
# end
function _matrix_representation(op::NCAdd, bases, space; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space(op, space; kwargs...)
    end
    sum(_matrix_representation(term, bases, space; kwargs...) for term in NCterms(op)) + op.coeff * I(dim(space))
end
function _matrix_representation(op, bases, space; kwargs...) #Assume op is a single symbolic operator
    _matrix_representation(NCMul(1, [op]), bases, space; kwargs...)
end

function _matrix_representation_single_space(op::NCAdd, space; kwargs...)
    outinds = Int[]
    ininds = Int[]
    AT = mat_eltype(op)
    amps = AT[]
    N = dim(space)
    sizehint!(outinds, N)
    sizehint!(ininds, N)
    sizehint!(amps, N)
    for (term, coeff) in op.dict
        operator_indices_and_amplitudes!((outinds, ininds, amps), coeff * term, space; kwargs...)
    end
    if !iszero(op.coeff)
        append!(ininds, 1:N)
        append!(outinds, 1:N)
        append!(amps, Fill(op.coeff, N))
    end
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end


function _term_matrix_representation(op, H::AbstractHilbertSpace; kwargs...)
    _outinds = Int[]
    _ininds = Int[]
    AT = mat_eltype(op)
    _amps = AT[]
    N = dim(H)
    sizehint!(_outinds, N)
    sizehint!(_ininds, N)
    sizehint!(_amps, N)
    (outinds, ininds, amps) = operator_indices_and_amplitudes!((_outinds, _ininds, _amps), op, H; kwargs...)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end
function _factorized_term_matrix_representation(ops::Vector, H; kwargs...)
    _outinds = Int[]
    _ininds = Int[]
    AT = promote_type([mat_eltype(op) for op in ops]...)
    _amps = AT[]
    N = dim(H)
    sizehint!(_outinds, N)
    sizehint!(_ininds, N)
    sizehint!(_amps, N)
    (outinds, ininds, amps) = operator_indices_and_amplitudes!((_outinds, _ininds, _amps), ops, H; kwargs...)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end

function apply_local_operator(op::NCMul{C,F}, state::AbstractFockState, space::AbstractHilbertSpace; kwargs...) where {C,F<:AbstractFermionSym}
    # Apply sequence of fermion operators (homogeneous type)
    digitpositions = collect(Iterators.reverse(_find_position(f, space) for f in op.factors))
    daggers = collect(Iterators.reverse(s.creation for s in op.factors))
    new_state, amp = togglefermions(digitpositions, daggers, state)
    return (new_state, amp)
end

# Fallback for other types
function apply_local_operator(op, state, space; kwargs...)
    error("apply_local_operator not implemented for operator type $(typeof(op)) on space type $(typeof(space))")
end

## Automatic basis inference from operator
"""
    extract_symbolic_bases(op)

Extract all unique symbolic bases from an operator expression.
Returns a set of unique bases found in the operator.
"""
function symbolic_groups(op::NCMul)
    bases = Set()
    for factor in op.factors
        basis = symbolic_group(factor)
        push!(bases, basis)
    end
    return bases
end

function symbolic_groups(op::NCAdd)
    bases = Set()
    for (term, coeff) in op.dict
        term_bases = symbolic_groups(term)
        union!(bases, term_bases)
    end
    return bases
end

function symbolic_groups(op)
    [symbolic_group(op)]
end

"""
    matrix_representation(op, space::AbstractHilbertSpace; kwargs...)

Return the matrix representation of symbolic operator `op` in Hilbert space `space`.

This is the main entry point for converting symbolic expressions (fermionic, Majorana,
spin, or mixed products supported by the space) into sparse/dense matrices.

# Examples
```julia
@fermions f
H = hilbert_space(f, 1:2)
op = f[1]' * f[2] + hc
M = matrix_representation(op, H)
size(M) == (dim(H), dim(H))
```
"""
function matrix_representation(op, space::AbstractHilbertSpace; kwargs...)
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(dim(space))
    end
    op_groups = symbolic_groups(op)
    space_groups = unique(map(symbolic_group, atomic_factors(space)))
    all(in(space_groups), op_groups) || throw(ArgumentError("Symbolic bases in operator do not match the atomic groups of the provided space. Operator groups: $op_groups, space groups: $space_groups"))
    return _matrix_representation(op, space_groups, space; kwargs...)
end
trivial_operator(op::Union{UniformScaling,Number}) = true
trivial_operator(op::NCMul) = length(op.factors) == 0
trivial_operator(op::NCAdd) = length(op.dict) == 0
trivial_operator(op) = false
get_trivial_op_coeff(op::NCMul) = op.coeff
get_trivial_op_coeff(op::NCAdd) = op.coeff
get_trivial_op_coeff(op) = op

@testitem "Multi-space matrix representation" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: SpinSpace, SpinState

    # Test 1: Fermion ⊗ Spin system
    @fermions f
    @spin s

    Nf = 2
    Hf = hilbert_space(f, 1:Nf)
    Hs = SpinSpace{1 // 2}(s)
    H = tensor_product(Hf, Hs)

    # Test simple product operator: f[1]' * f[1] ⊗ S_z
    op = f[1]' * f[1] * s[:z]
    M = matrix_representation(op, H)
    @test M == kron(matrix_representation(s[:z], Hs), matrix_representation(f[1]' * f[1], Hf))

    # Verify dimensions: (2^2) × 2 = 8×8
    @test size(M) == (8, 8)

    # Check that the matrix is sparse
    @test M isa SparseMatrixCSC

    # Test 2: Sum of mixed operators
    ops = [f[1]' * f[1] * s[:z], f[2]' * f[2] * s[:+], 2im]
    op_mixed = sum(ops)
    M_mixed = matrix_representation(op_mixed, H)
    @test size(M_mixed) == (8, 8)
    @test M_mixed == sum(matrix_representation(op, H) for op in ops)


    # Test 3: Verify the result is hermitian for hermitian operators
    op_herm = f[1]' * f[2] * s[:z] + hc
    M_herm = matrix_representation(op_herm, H)
    @test M_herm ≈ M_herm'  # Should be hermitian
    @test M_herm == matrix_representation(op_herm', H)
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
    M = matrix_representation(op, H)
    @test M == kron(reverse([matrix_representation(f[1]' * f[1], Hf), matrix_representation(g[1]' * g[1], Hg), matrix_representation(s[:z], Hs)])...)

    # Should have dimension 2 × 2 × 2 = 8
    @test size(M) == (8, 8)
    @test M isa SparseMatrixCSC

    @test_throws ArgumentError matrix_representation(op, Hs)
    @test_throws ArgumentError matrix_representation(op, Hf)
    @test_throws ArgumentError matrix_representation(op, Hg)
end
