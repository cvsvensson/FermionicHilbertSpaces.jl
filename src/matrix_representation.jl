abstract type AbstractFermionSym <: AbstractSym end

struct EagerRepr end
struct LazyRepr end

function mat_eltype(::NCAdd{C,NCMul{C2,S,F}}) where {C,C2,S,F}
    promote_type(C, mat_eltype(S))
end
function mat_eltype(::NCMul{C,S,F}) where {C,S,F}
    promote_type(C, mat_eltype(S))
end
mat_eltype(::Type{NCMul{C,S,F}}) where {C,S,F} = promote_type(C, mat_eltype(S))
mat_eltype(::S) where {S} = mat_eltype(S)
mat_eltype(::Type{S}) where {S} = Float64 #Default fallback. Could give errors if a complex number is expected. Override it for specific types if needed.

_concretize(op::NCMul) = op
_concretize(op::NCMul{C,AbstractSym,F}) where {C,F} = NCMul(op.coeff, Tuple(op.factors))
_concretize(op::NCMul{C,Any,F}) where {C,F} = NCMul(op.coeff, Tuple(op.factors))
_concretize(op::OperatorSequence) = OperatorSequence(map(_concretize, op.ops))
function operator_indices_and_amplitudes!((outinds, ininds, amps), op, space::AbstractHilbertSpace; kwargs...)
    concrete_op = _concretize(op) # op is often an NCMul with Abstract types. We try to make it concrete here, as the operator will be applied to all basis states, so the overhead of concretization is likely worth it
    precomp = _precomputation_before_operator_application(concrete_op, space)
    return operator_indices_and_amplitudes_generic!((outinds, ininds, amps), concrete_op, space, precomp; kwargs...)
end
_precomputation_before_operator_application(factors, space) = nothing

function _apply_local_operators_index(op, index::Integer, space, precomp)
    state = basisstate(index, space)
    newstate, amp = _apply_local_operators(op, state, space, precomp)
    newindex = state_index(newstate, space)
    return newindex, amp
end
function _apply_local_operators(op, state, space, precomp)
    apply_local_operators(op, state, space, precomp; transpose=false)
end

function operator_indices_and_amplitudes_generic!((outinds, ininds, amps), op, space::AbstractHilbertSpace, precomp; projection)
    for inind in eachindex(basisstates(space))
        outind, amp = _apply_local_operators_index(op, inind, space, precomp)
        if !iszero(amp)
            if iszero(outind)
                if projection
                    continue
                else
                    throw(ArgumentError("Operator maps outside of the provided space. Set projection=true to ignore those states."))
                end
            else
                push!(outinds, outind)
                push!(amps, amp)
                push!(ininds, inind)
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
        filter(==(basis) ∘ symbolic_group, factors)
    end
    sum(length, partition) == length(factors) || throw(ArgumentError("Not all factors were assigned to a basis."))
    return partition
end


function _matrix_representation(op::NCMul, bases, space::ProductSpace, repr; kwargs...)
    spaces = factors(space)
    partitioned = partition_factors_by_basis(op.factors, bases)
    matrices = Iterators.map(partitioned, spaces, eachindex(spaces)) do factors, space, n
        coeff = n == 1 ? op.coeff : one(op.coeff)
        if isempty(factors)
            return coeff * I(dim(space))
        else
            _term_matrix_representation(NCMul(coeff, factors), space, repr; kwargs...)
        end
    end
    length(spaces) == 1 && return first(matrices)
    return foldl(kron, Iterators.reverse(matrices))
end
function _matrix_representation(op::NCMul, bases, space, repr; kwargs...)
    if isempty(op.factors)
        return op.coeff * I(dim(space))
    else
        if length(bases) > 1
            partition = partition_factors_by_basis(op.factors, bases)
            vecops = OperatorSequence(map((n, ops) -> NCMul(n == 1 ? op.coeff : one(op.coeff), ops), eachindex(partition), partition))
            return _factorized_term_matrix_representation(vecops, space, repr; kwargs...)
        else
            return _term_matrix_representation(op, space, repr; kwargs...)
        end
    end
end
function _matrix_representation(op::NCAdd, bases, space, repr; kwargs...)
    if length(bases) == 1
        return _matrix_representation_single_space(op, space, repr; kwargs...)
    end
    sum(_matrix_representation(term, bases, space, repr; kwargs...) for term in NCterms(op)) + op.coeff * I(dim(space))
end
function _matrix_representation(op, bases, space, repr; kwargs...) #Assume op is a single symbolic operator
    _matrix_representation(NCMul(1, [op]), bases, space, repr; kwargs...)
end

function _matrix_representation_single_space(op::NCAdd, space, ::EagerRepr; kwargs...)
    outinds = Int[]
    ininds = Int[]
    AT = mat_eltype(op)
    amps = AT[]
    N = dim(space)
    length_guess = Int(floor(1 + log2(length(op.dict) + 1))) * N # mild increase with number of terms
    sizehint!(outinds, length_guess)
    sizehint!(ininds, length_guess)
    sizehint!(amps, length_guess)
    for (term, coeff) in op.dict
        operator_indices_and_amplitudes!((outinds, ininds, amps), coeff * term, space; kwargs...)
    end
    if !iszero(op.coeff)
        append!(ininds, 1:N)
        append!(outinds, 1:N)
        append!(amps, Fill(op.coeff, N))
    end
    isconcretetype(eltype(amps)) && return SparseArrays.sparse!(outinds, ininds, amps, N, N)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end


function _term_matrix_representation(op, H::AbstractHilbertSpace, ::EagerRepr; kwargs...)
    _outinds = Int[]
    _ininds = Int[]
    AT = mat_eltype(op)
    _amps = AT[]
    N = dim(H)
    sizehint!(_outinds, N)
    sizehint!(_ininds, N)
    sizehint!(_amps, N)
    (outinds, ininds, amps) = operator_indices_and_amplitudes!((_outinds, _ininds, _amps), op, H; kwargs...)
    isconcretetype(eltype(amps)) && return SparseArrays.sparse!(outinds, ininds, amps, N, N)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end
function _factorized_term_matrix_representation(ops::OperatorSequence, H, ::EagerRepr; kwargs...)
    _outinds = Int[]
    _ininds = Int[]
    AT = promote_type([mat_eltype(op) for op in ops.ops]...)
    _amps = AT[]
    N = dim(H)
    sizehint!(_outinds, N)
    sizehint!(_ininds, N)
    sizehint!(_amps, N)
    (outinds, ininds, amps) = operator_indices_and_amplitudes!((_outinds, _ininds, _amps), ops, H; kwargs...)
    isconcretetype(eltype(amps)) && return SparseArrays.sparse!(outinds, ininds, amps, N, N)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end

# function apply_local_operator(op::NCMul{C,F}, state::AbstractFockState, space::AbstractHilbertSpace, (pos, daggers)) where {C,F<:AbstractFermionSym}
#     new_state, amp = togglefermions(Iterators.reverse(pos), Iterators.reverse(daggers), state)
#     return new_state, amp
# end


## Automatic basis inference from operator
"""
    extract_symbolic_bases(op)

Extract all unique symbolic bases from an operator expression.
Returns a set of unique bases found in the operator.
"""
function symbolic_groups(op::NCMul, bases=Set())
    for factor in op.factors
        push!(bases, symbolic_group(factor))
    end
    return bases
end

function symbolic_groups(op::NCAdd)
    bases = Set()
    for term in keys(op.dict)
        symbolic_groups(term, bases)
    end
    return bases
end

function symbolic_groups(op) # Assume op is a single symbolic operator
    [symbolic_group(op)]
end

"""
    matrix_representation(op, space::AbstractHilbertSpace; kwargs...)

Return the matrix representation of symbolic operator `op` in Hilbert space `space`.

Keyword arguments include `lazy` (default `false`) to return a `LazyOperator` that computes the matrix-vector product on demand, and `projection` (default `false`) which if true will ignore the error thrown when the operator maps outside the space.

# Examples
```julia
@fermions f
H = hilbert_space(f, 1:2)
op = f[1]' * f[2] + hc
M = matrix_representation(op, H)
size(M) == (dim(H), dim(H))
```
"""
function matrix_representation(op, space::AbstractHilbertSpace; lazy=false, projection=false, kwargs...)
    repr = lazy ? LazyRepr() : EagerRepr()
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(dim(space))
    end
    op_groups = symbolic_groups(op)
    space_groups = unique(Iterators.map(group_id, factors(space)))
    all(in(space_groups), op_groups) || throw(ArgumentError("Symbolic bases in operator do not match the atomic groups of the provided space. Operator groups: $op_groups, space groups: $space_groups"))
    return _matrix_representation(op, space_groups, space, repr; projection, kwargs...)
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
