abstract type AbstractSym end
abstract type AbstractFermionSym <: AbstractSym end

function mat_eltype(ncadd::NCAdd{C,NCMul{C2,S,F}}) where {C,C2,S<:AbstractSym,F}
    isconcretetype(S) && return promote_type(C, mat_eltype(S))
    return _mat_eltype(ncadd)
end
function mat_eltype(ncmul::NCMul{C,S,F}) where {C,S<:AbstractSym,F}
    isconcretetype(S) && return promote_type(C, mat_eltype(S))
    return _mat_eltype(ncmul)
end
mat_eltype(::Type{NCMul{C,S,F}}) where {C,S<:AbstractSym,F} = promote_type(C, mat_eltype(S))
mat_eltype(::S) where {S<:AbstractSym} = mat_eltype(S)

function _mat_eltype(ncmul::NCMul{C}) where C
    factor_valtypes = [mat_eltype(f) for f in ncmul.factors]
    return promote_type(C, factor_valtypes...)
end
function _mat_eltype(ncadd::NCAdd{C,<:NCMul}) where C
    term_valtypes = [mat_eltype(term) for term in NCterms(ncadd)]
    return promote_type(C, term_valtypes...)
end

function operator_inds_amps!((outinds, ininds, amps), op, H::AbstractHilbertSpace; kwargs...)
    return operator_inds_amps_generic!((outinds, ininds, amps), op, H; kwargs...)
end

function operator_inds_amps_generic!((outinds, ininds, amps), op::NCMul{C,F}, space::AbstractHilbertSpace; projection=false) where {C,F}
    for (n, state) in enumerate(basisstates(space))
        newstate, amp = apply_local_operators(op.factors, state, space)
        if !iszero(amp)
            outind = state_index(newstate, space)
            if !projection || !ismissing(outind)
                push!(outinds, outind)
                push!(amps, amp * op.coeff)
                push!(ininds, n)
            end
        end
    end
    return (outinds, ininds, amps)
end

function operator_inds_amps_generic!((outinds, ininds, amps), op::NCMul{C,F}, H::AbstractFockHilbertSpace; projection=false) where {C,F<:AbstractFermionSym}
    ordering = mode_ordering(H)
    digitpositions = collect(Iterators.reverse(getindex(ordering, f.label) for f in op.factors))
    daggers = collect(Iterators.reverse(s.creation for s in op.factors))
    mc = -op.coeff
    pc = op.coeff
    for (n, f) in enumerate(basisstates(H))
        newstate, amp = togglefermions(digitpositions, daggers, f)
        if !iszero(amp)
            outind = state_index(newstate, H)
            if !projection || !ismissing(outind)
                ismissing(outind) && throw(ArgumentError("State $newstate not found in basis."))
                push!(outinds, outind)
                push!(amps, isone(amp) ? pc : mc)
                push!(ininds, n)
            end
        end
    end
    return (outinds, ininds, amps)
end


function _matrix_representation(op::NCAdd{C}, ordering, states, fock_to_ind; kwargs...) where C
    outinds = Int[]
    ininds = Int[]
    AT = mat_eltype(op)
    amps = AT[]
    sizehint!(outinds, length(states))
    sizehint!(ininds, length(states))
    sizehint!(amps, length(states))
    for (operator, coeff) in op.dict
        operator_inds_amps!((outinds, ininds, amps), coeff * operator, ordering, states, fock_to_ind; kwargs...)
    end
    if !iszero(op.coeff)
        append!(ininds, eachindex(states))
        append!(outinds, eachindex(states))
        append!(amps, Fill(op.coeff, length(states)))
    end
    return SparseArrays.sparse!(outinds, ininds, amps, length(states), length(states))
end
operator_inds_amps!((outinds, ininds, amps), op::AbstractFermionSym, args...; kwargs...) = operator_inds_amps!((outinds, ininds, amps), NCMul(1, [op]), args...; kwargs...)

@testitem "Instantiating symbolic fermions" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: eval_in_basis
    @fermions f
    N = 4
    labels = 1:N
    H = FockHilbertSpace(labels)
    fmb = fermions(H)
    fockstates = map(FockNumber, 0:2^N-1)
    get_mat(op) = matrix_representation(op, H)
    @test all(get_mat(f[l]) == fmb[l] for l in labels)
    @test all(get_mat(f[l]') == fmb[l]' for l in labels)
    @test all(get_mat(f[l]') == get_mat(f[l])' for l in labels)
    @test all(get_mat(f[l]'') == get_mat(f[l]) for l in labels)
    @test all(get_mat(f[l]' * f[l]) == get_mat(f[l])' * get_mat(f[l]) for l in labels)
    @test all(get_mat(f[l]' * f[l]) == fmb[l]' * fmb[l] for l in labels)

    newmat = get_mat(sum(f[l]' * f[l] for l in labels))
    mat = sum(fmb[l]' * fmb[l] for l in labels)
    @test newmat == mat

    @test all(eval_in_basis(f[l], fmb) == fmb[l] for l in labels)
    @test all(eval_in_basis(f[l]', fmb) == fmb[l]' for l in labels)
    @test all(eval_in_basis(f[l]' * f[l], fmb) == fmb[l]'fmb[l] for l in labels)
    @test all(eval_in_basis(f[l] + f[l]', fmb) == fmb[l] + fmb[l]' for l in labels)
end

## Convert to expression
eval_in_basis(a::NCMul, f) = a.coeff * mapfoldl(Base.Fix2(eval_in_basis, f), *, a.factors)
eval_in_basis(a::NCAdd, f) = a.coeff * I + mapfoldl(Base.Fix2(eval_in_basis, f), +, NCterms(a))

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
function partition_factors_by_basis(factors::Vector, bases::Vector)
    partition = map(bases) do basis
        [filter(==(basis) ∘ get_symbolic_basis, factors)...]
    end
    sum(length, partition) == length(factors) || throw(ArgumentError("Not all factors were assigned to a basis."))
    return partition
end

"""
    matrix_representation(op, basis_space_pairs::AbstractVector{<:Pair})

Compute the matrix representation of a symbolic operator on a product hilbert space.The `basis_space_pairs` argument is a vector of pairs `symbolic_basis => hilbert_space`, where each symbolic basis (e.g., from `@fermions f`, `@spin s`, etc.) is mapped to its corresponding Hilbert space.

# Example
```julia
@fermions f
@spin s
Hf = FockHilbertSpace([1, 2])
Hs = SpinSpace{1//2}(0)
op = f[1]' * f[1] * s[0](:z)
M = matrix_representation(op, [f => Hf, s => Hs])
```
"""
function matrix_representation(op, basis_space_pairs::AbstractVector{<:Pair}; kwargs...)
    bases = [get_symbolic_basis(first(p)) for p in basis_space_pairs] # we call get_symbolic_basis because for e.g. a boson 'b', it represents both the basis and the operator, so we need to extract an invariant
    spaces = [last(p) for p in basis_space_pairs]
    _matrix_representation(op, bases, spaces; kwargs...)
end

function _matrix_representation(op::NCMul, bases, spaces; kwargs...)
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

function _matrix_representation(op::NCAdd, bases, spaces; kwargs...)
    if length(spaces) == 1
        return _matrix_representation_single_space(op, only(spaces); kwargs...)
    end
    sum(_matrix_representation(term, bases, spaces; kwargs...) for term in NCterms(op)) + op.coeff * I(prod(dim.(spaces)))
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
        operator_inds_amps!((outinds, ininds, amps), coeff * term, space; kwargs...)
    end
    if !iszero(op.coeff)
        append!(ininds, 1:N)
        append!(outinds, 1:N)
        append!(amps, Fill(op.coeff, N))
    end
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end
function _matrix_representation(op, bases, spaces; kwargs...) #Assume op is a single symbolic operator
    _matrix_representation(NCMul(1, [op]), bases, spaces; kwargs...)
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
    (outinds, ininds, amps) = operator_inds_amps!((_outinds, _ininds, _amps), op, H; kwargs...)
    return SparseArrays.sparse!(outinds, ininds, identity.(amps), N, N)
end


function apply_local_operator(op::NCMul{C,F}, state::FockNumber, space::AbstractFockHilbertSpace; kwargs...) where {C,F<:AbstractFermionSym}
    # Apply sequence of fermion operators (homogeneous type)
    ordering = mode_ordering(space)
    digitpositions = collect(Iterators.reverse(getindex(ordering, f.label) for f in op.factors))
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
function extract_symbolic_bases(op::NCMul)
    bases = Set()
    for factor in op.factors
        basis = get_symbolic_basis(factor)
        push!(bases, basis)
    end
    return bases
end

function extract_symbolic_bases(op::NCAdd)
    bases = Set()
    for (term, coeff) in op.dict
        term_bases = extract_symbolic_bases(term)
        union!(bases, term_bases)
    end
    return bases
end

function extract_symbolic_bases(op)
    [get_symbolic_basis(op)]
end

function infer_basis_space_pairs(op, spaces::AbstractVector{<:AbstractHilbertSpace})
    # Extract bases from operator
    unique_bases = collect(extract_symbolic_bases(op))
    # Match them to spaces
    length(unique_bases) == length(spaces) || throw(ArgumentError("Number of unique symbolic bases in operator ($(length(unique_bases))) does not match number of provided spaces ($(length(spaces)))"))
    bases = map(spaces) do space
        matches = filter(basis -> _sym_space_match(basis, space), unique_bases)
        if length(matches) == 1
            return matches[1]
        else
            throw(ArgumentError("Could not uniquely match space $space to a basis. Matches found: $matches"))
        end
    end
    bases, spaces
end

"""
    matrix_representation(op, spaces::AbstractVector{<:AbstractHilbertSpace})

Compute the matrix representation with automatic basis inference.
The function will attempt to match symbolic bases in the operator to the provided spaces.

# Example
```julia
@fermions f
@spin s
H_f = FockHilbertSpace([1, 2])
H_s = SpinSpace{1//2}(0)
op = f[1]' * f[1] * s[0](:z)
M = matrix_representation(op, [H_f, H_s])
```
"""
function matrix_representation(op, spaces::AbstractVector{<:AbstractHilbertSpace}; kwargs...)
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(prod(dim.(spaces)))
    end
    bases, spaces = infer_basis_space_pairs(op, spaces)
    return _matrix_representation(op, bases, spaces; kwargs...)
end
function matrix_representation(op, space::AbstractHilbertSpace; kwargs...)
    if trivial_operator(op)
        return get_trivial_op_coeff(op) * I(dim(space))
    end
    bases, spaces = infer_basis_space_pairs(op, [space])
    return _matrix_representation(op, bases, spaces; kwargs...)
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
    Hf = FockHilbertSpace(1:Nf)
    Hs = SpinSpace{1 // 2}(s)

    # Test simple product operator: f[1]' * f[1] ⊗ S_z
    op = f[1]' * f[1] * s[:z]
    M = matrix_representation(op, [f => Hf, s => Hs])
    @test M == matrix_representation(op, [Hf, Hs])

    # Verify dimensions: (2^2) × 2 = 8×8
    @test size(M) == (8, 8)

    # Check that the matrix is sparse
    @test M isa SparseMatrixCSC

    # Test 2: Sum of mixed operators
    op_mixed = f[1]' * f[1] * s[:z] + f[2]' * f[2] * s[:+]
    M_mixed = matrix_representation(op_mixed, [f => Hf, s => Hs])
    @test size(M_mixed) == (8, 8)
    @test M_mixed == matrix_representation(op_mixed, [Hf, Hs])


    # Test 3: Verify the result is hermitian for hermitian operators
    op_herm = f[1]' * f[2] * s[:z] + hc
    M_herm = matrix_representation(op_herm, [f => Hf, s => Hs])
    @test M_herm ≈ M_herm'  # Should be hermitian
    @test M_herm == matrix_representation(op_herm, [Hf, Hs])
end

@testitem "Three-space product" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: SpinSpace

    @fermions f
    @fermions g  # Different fermion species, commuting with f
    @spin s

    Hf = FockHilbertSpace([1])
    Hg = FockHilbertSpace([1])
    Hs = SpinSpace{1 // 2}(s.name)

    # Operator acting on all three spaces
    op = f[1]' * f[1] * g[1]' * g[1] * s[:z]
    M = matrix_representation(op, [f => Hf, g => Hg, s => Hs])
    @test M == kron(reverse([matrix_representation(f[1]' * f[1], Hf), matrix_representation(g[1]' * g[1], Hg), matrix_representation(s[:z], Hs)])...)

    # Should have dimension 2 × 2 × 2 = 8
    @test size(M) == (8, 8)
    @test M isa SparseMatrixCSC

    @test_throws MethodError matrix_representation(op, [f => Hs])
    @test_throws FieldError matrix_representation(op, [f => Hf])
    @test_throws FieldError matrix_representation(op, [f => Hg])
    @test_throws MethodError matrix_representation(op, [f => Hf, g => Hs, s => Hf])
end
