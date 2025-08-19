abstract type AbstractFermionSym end

Base.valtype(::NCAdd{C,S}) where {C,S} = promote_type(C, valtype(S))
Base.valtype(::NCMul{C,S}) where {C,S<:AbstractFermionSym} = promote_type(C, valtype(S))
Base.valtype(::Type{NCMul{C,S,F}}) where {C,S<:AbstractFermionSym,F} = promote_type(C, valtype(S))

## Instantiating sparse matrices
matrix_representation(op, H::AbstractFockHilbertSpace) = matrix_representation(op, mode_ordering(H), basisstates(H), Dict(Iterators.map(reverse, enumerate(basisstates(H)))))
matrix_representation(op::Number, H::AbstractFockHilbertSpace) = op * I(size(H, 1))

function matrix_representation(op::Union{<:NCMul,<:AbstractFermionSym}, labels, states, fock_to_ind)
    outinds = Int[]
    ininds = Int[]
    AT = valtype(op)
    amps = AT[]
    sizehint!(outinds, length(states))
    sizehint!(ininds, length(states))
    sizehint!(amps, length(states))
    operator_inds_amps!((outinds, ininds, amps), op, labels, states, fock_to_ind)
    SparseArrays.sparse!(outinds, ininds, identity.(amps), length(states), length(states))
end
matrix_representation(op, labels, states) = matrix_representation(op, labels, states, Dict(Iterators.map(reverse, enumerate(states))))

function operator_inds_amps!((outinds, ininds, amps), op, ordering, states::AbstractVector{SingleParticleState}, fock_to_ind)
    isquadratic(op) && isnumberconserving(op) && return operator_inds_amps_free_fermion!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
    return operator_inds_amps_generic!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
end

function operator_inds_amps!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
    return operator_inds_amps_generic!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
end

function operator_inds_amps_free_fermion!((outinds, ininds, amps), op::NCMul, ordering, states::AbstractVector{SingleParticleState}, fock_to_ind)
    if length(op.factors) != 2
        throw(ArgumentError("Only two-fermion operators supported for free fermions"))
    end
    fockstates = (SingleParticleState(getindex(ordering, op.factors[1].label)), SingleParticleState(getindex(ordering, op.factors[2].label)))
    inind = fock_to_ind[fockstates[2]]
    outind = fock_to_ind[fockstates[1]]
    sign = (-1)^op.factors[2].creation
    push!(outinds, outind)
    push!(ininds, inind)
    push!(amps, sign * op.coeff)
    return (outinds, ininds, amps)
end

function operator_inds_amps_generic!((outinds, ininds, amps), op::NCMul, ordering, states, fock_to_ind)
    digitpositions = collect(Iterators.reverse(getindex(ordering, f.label) for f in op.factors))
    daggers = collect(Iterators.reverse(s.creation for s in op.factors))
    mc = -op.coeff
    pc = op.coeff
    for (n, f) in enumerate(states)
        newfockstate, amp = togglefermions(digitpositions, daggers, f)
        if !iszero(amp)
            push!(outinds, fock_to_ind[newfockstate])
            push!(amps, isone(amp) ? pc : mc)
            push!(ininds, n)
        end
    end
    return (outinds, ininds, amps)
end

# promote_array(v) = convert(Array{eltype(promote(map(zero, unique(typeof(v) for v in v))...))}, v)

function matrix_representation(op::NCAdd{C}, ordering, states, fock_to_ind) where C
    outinds = Int[]
    ininds = Int[]
    AT = valtype(op)
    amps = AT[]
    sizehint!(outinds, length(states))
    sizehint!(ininds, length(states))
    sizehint!(amps, length(states))
    for (operator, coeff) in op.dict
        operator_inds_amps!((outinds, ininds, amps), coeff * operator, ordering, states, fock_to_ind)
    end
    if !iszero(op.coeff)
        append!(ininds, eachindex(states))
        append!(outinds, eachindex(states))
        append!(amps, fill(op.coeff, length(states)))
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

    @test all(matrix_representation(sum(f[l]' * f[l] for l in labels), H.jw.ordering, FermionicHilbertSpaces.fixed_particle_number_fockstates(N, n)) == n * I for n in 1:N)

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
