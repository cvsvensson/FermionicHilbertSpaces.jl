abstract type AbstractFermionSym <: AbstractSym end

struct EagerDenseRepr end
struct EagerSparseRepr end

struct LazyRepr{T}
    input::T
end
LazyRepr() = LazyRepr(missing)

function mat_eltype(::NCAdd{C,NCMul{C2,S,F}}) where {C,C2,S,F}
    promote_type(C, mat_eltype(S))
end
function mat_eltype(::NCMul{C,S,F}) where {C,S,F}
    promote_type(C, mat_eltype(S))
end
mat_eltype(::Type{NCMul{C,S,F}}) where {C,S,F} = promote_type(C, mat_eltype(S))
mat_eltype(::S) where {S} = mat_eltype(S)
mat_eltype(::Type{S}) where {S} = Float64 #Default fallback. Could give errors if a complex number is expected. Override it for specific types if needed.

_precomputation_before_operator_application(factors, space) = nothing

function operator_indices_and_amplitudes!(accumulator, op, space::AbstractHilbertSpace; kwargs...)
    precomp = _precomputation_before_operator_application(op, space)
    return operator_indices_and_amplitudes_generic!(accumulator, op, space, precomp; kwargs...)
end

function operator_indices_and_amplitudes_generic!(accumulator, op, space::AbstractHilbertSpace, precomp; projection)
    for (inind, state) in enumerate(basisstates(space))
        newstate, amp = _apply_local_operators(op, state, space, precomp)
        if !iszero(amp)
            outind = state_index(newstate, space)
            if iszero(outind)
                if projection
                    continue
                else
                    throw(ArgumentError("Operator maps outside of the provided space. Set projection=true to ignore those states."))
                end
            else
                push_inds_amps!(accumulator, outind, inind, amp)
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
function push_inds_amps!(m::Matrix, outind, inind, amp)
    m[outind, inind] += amp
end


_apply_local_operators(op::Missing, state, space, precomp) = state, 1
function _apply_local_operators(op, state, space, precomp)
    apply_local_operators(op, state, space, precomp; transpose=false)
end

function apply_local_operators(_op::NCMul, state, space, precomp; transpose)
    if !transpose
        return foldr(_op.factors, init=(state, _op.coeff)) do op, (state, amp)
            newstate, _amp = apply_local_operator(op, state, space, precomp)
            return newstate, amp * _amp
        end
    elseif transpose
        return foldl(_op.factors; init=(state, _op.coeff)) do (state, amp), op
            newstate, _amp = apply_local_operator(op', state, space, precomp)
            return newstate, amp * conj(_amp)
        end
    end
end


"""
    partition_factors_by_basis(factors::Vector, bases::Vector)

Partition a vector of operator factors into groups by their symbolic basis.
"""
function partition_factors_by_basis(factors::Vector, bases)
    partition = Iterators.map(bases) do basis
        v = filter(==(basis) ∘ symbolic_group, factors)
        length(v) > 0 ? Tuple(v) : v #missing
    end
    sum(length, skipmissing(partition)) == length(factors) || throw(ArgumentError("Not all factors were assigned to a basis."))
    return partition
end
struct TupleConcretizer end
struct VecConcretizer end
struct NoConcretizer end
___concretize(factors, ::TupleConcretizer) = Tuple(factors)
___concretize(factors::AbstractVector, ::VecConcretizer) = map(identity, factors)
___concretize(factors::Tuple, ::VecConcretizer) = collect(factors)
___concretize(factors, ::NoConcretizer) = factors

function partition_product(op::NCMul, bases, spaces, concr=VecConcretizer())
    used_coeff = false
    n = 0
    ops = map(bases, spaces) do basis, space
        v = filter(==(basis) ∘ symbolic_group, op.factors)
        if length(v) > 0
            vc = ___concretize(v, concr)
            coeff = used_coeff ? one(op.coeff) : op.coeff
            used_coeff = true
            n += length(v)
            return NCMul(coeff, vc)
        end
        return missing
    end
    n == length(op.factors) || throw(ArgumentError("Not all factors were assigned to a basis."))
    return ProductOperator(___concretize(ops, concr), spaces)
end

import FillArrays: Eye
function _matrix_representation(op::NCMul, bases, space::ProductSpace, repr; kwargs...)
    spaces = factors(space)
    length(spaces) == 1 && return _term_matrix_representation(op, local_space, repr; kwargs...)

    prodop = partition_product(op, bases, spaces)
    matrices = map(prodop.ops, prodop.spaces) do op, local_space
        ismissing(op) && return Eye(dim(local_space)) #I(dim(local_space))
        _term_matrix_representation(op, local_space, repr; kwargs...)
    end

    mergedmatrices = _merge_diags(matrices)
    length(spaces) == 1 && return first(mergedmatrices)
    return foldl(kron, Iterators.reverse(mergedmatrices))
end
function _merge_diags(matrices)
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

function _matrix_representation(op::NCMul, bases, space, repr; kwargs...)
    if isempty(op.factors)
        return op.coeff * I(dim(space))
    else
        if length(bases) > 1
            prodop = partition_product(op, bases, factors(space))
            return _factorized_term_matrix_representation(prodop, space, repr; kwargs...)
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
function _matrix_representation(op::Missing, bases, space, repr; kwargs...)
    Eye(dim(space))
end

function matrix_accumulator(op::NCAdd, space, ::EagerSparseRepr)
    length_guess = Int(floor(1 + log2(length(op.dict) + 1))) * dim(space) # mild increase with number of terms
    return sparse_matrix_accumulator(mat_eltype(op), length_guess)
end
matrix_accumulator(op::Union{NCMul,ProductOperator}, space, ::EagerSparseRepr) = sparse_matrix_accumulator(mat_eltype(op), dim(space))
function sparse_matrix_accumulator(::Type{T}, N) where T
    outinds = Int[]
    ininds = Int[]
    amps = T[]
    sizehint!(outinds, N)
    sizehint!(ininds, N)
    sizehint!(amps, N)
    return (outinds, ininds, amps)
end
function add_identity!!((outinds, ininds, amps), coeff, space)
    N = dim(space)
    append!(ininds, 1:N)
    append!(outinds, 1:N)
    append!(amps, Fill(coeff, N))
end
function finalize!((outinds, ininds, amps), space)
    N = dim(space)
    isconcretetype(eltype(amps)) && return SparseArrays.sparse!(outinds, ininds, amps, N, N)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end

function matrix_accumulator(op, space, ::EagerDenseRepr)
    T = mat_eltype(op)
    N = dim(space)
    zeros(T, N, N)
end
function add_identity!!(m::Matrix{T}, coeff, space) where T
    m .+= coeff * I(size(m, 1))
end
function finalize!(m::Matrix, space)
    return m
end


function _matrix_representation_single_space(op::NCAdd, space, repr::Union{EagerSparseRepr,EagerDenseRepr}; kwargs...)
    accumulator = matrix_accumulator(op, space, repr)
    for (term, coeff) in op.dict
        operator_indices_and_amplitudes!(accumulator, coeff * term, space; kwargs...)
    end
    if !iszero(op.coeff)
        add_identity!!(accumulator, op.coeff, space)
    end
    finalize!(accumulator, space)
end


function _term_matrix_representation(op, H::AbstractHilbertSpace, repr::Union{EagerSparseRepr,EagerDenseRepr}; kwargs...)
    _accumulator = matrix_accumulator(op, H, repr)
    accumulator = operator_indices_and_amplitudes!(_accumulator, op, H; kwargs...)
    finalize!(accumulator, H)
end
function _factorized_term_matrix_representation(ops::ProductOperator, H, repr::Union{EagerSparseRepr,EagerDenseRepr}; kwargs...)
    _accumulator = matrix_accumulator(ops, H, repr)
    accumulator = operator_indices_and_amplitudes!(_accumulator, ops, H; kwargs...)
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
    representation(op_or_state, space::AbstractHilbertSpace; kwargs...)

Return a concrete representation of `op_or_state` in Hilbert space `space`.

- For symbolic operators, returns a sparse matrix (or lazy operator when `lazy=true`).
- For `AbstractBasisState`, returns a one-hot sparse column vector.
- For `SymbolicState` (ket/bra), returns the corresponding column/row vector.
"""
function representation(state::AbstractBasisState, space::AbstractHilbertSpace; kwargs...)
    vector_representation(state, space)
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
function matrix_representation(op, space::AbstractHilbertSpace, type=EagerSparseRepr(); projection=false, kwargs...)
    repr = _process_type(type)
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(dim(space))
    end
    op_groups = symbolic_groups(op)
    space_groups = group_ids(space)
    all(in(space_groups), op_groups) || throw(ArgumentError("Symbolic bases in operator do not match the atomic groups of the provided space. Operator groups: $op_groups, space groups: $space_groups"))
    return _matrix_representation(op, space_groups, space, repr; projection, kwargs...)
end
_process_type(t) = t
function _process_type(s::Symbol)
    s == :sparse && return EagerSparseRepr()
    s == :dense && return EagerDenseRepr()
    s == :lazy && return LazyRepr()
    throw(ArgumentError("Invalid type argument: $s. Expected :sparse, :dense, or :lazy."))
end


# group_ids(space::ProductSpace) = unique(Iterators.map(group_id, factors(space)))
group_ids(space::ProductSpace) = map(group_id, factors(space))
group_ids(space::Union{AbstractAtomicHilbertSpace,AbstractGroupedHilbertSpace}) = (group_id(space),)
group_ids(space::AbstractHilbertSpace) = group_ids(parent(space))

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


@testitem "Dense/sparse/lazy matrix representation agree" begin
    using SparseArrays, LinearAlgebra
    import FermionicHilbertSpaces.SciMLOperators: concretize
    @fermions f
    @spin s 1 // 2

    Hf = hilbert_space(f, 1:2)
    op = f[1]' * f[2] + 1im * f[2]' * f[1] + 2
    Md = matrix_representation(op, Hf, :dense)
    Ms = matrix_representation(op, Hf)
    Ml = concretize(matrix_representation(op, Hf, :lazy))
    @test Md == Ms
    @test Md == Ml

    Hs = hilbert_space(s)
    Hprod = tensor_product(Hf, Hs)
    op_prod = f[1]' * f[2] * s[:z] + 2 * f[2]' * f[1] * s[:x] + 1im
    Md = matrix_representation(op_prod, Hprod, :dense)
    Ms = matrix_representation(op_prod, Hprod)
    Ml = concretize(matrix_representation(op_prod, Hprod, :lazy))
    @test Md == Ms
    @test Md == Ml
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
