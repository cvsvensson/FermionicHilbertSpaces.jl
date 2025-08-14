abstract type AbstractOrdering end
struct NormalOrdering <: AbstractOrdering end
struct NaiveOrdering <: AbstractOrdering end

abstract type AbstractFermionSym end
ordered_product(a::AbstractFermionSym, b::AbstractFermionSym, ::NaiveOrdering) = FermionMul(1, [a, b])

struct FermionMul{C,S<:AbstractFermionSym}
    coeff::C
    factors::Vector{S}
    function FermionMul(coeff::C, factors) where {C}
        new{C,eltype(factors)}(coeff, factors)
    end
end
Base.convert(::Type{FermionMul{C,S}}, x::FermionMul{<:Any,S}) where {C,S} = FermionMul(convert(C, x.coeff), x.factors)
function canonicalize!(a::FermionMul)
    if iszero(a.coeff)
        return a.coeff
    elseif length(a.factors) == 1 && isone(a.coeff)
        return only(a.factors)
    end
    return a
end

function Base.show(io::IO, x::FermionMul)
    isscalar(x) && print(io, x.coeff)
    print_coeff = !isone(x.coeff)
    if print_coeff
        v = x.coeff
        if isreal(v)
            neg = real(v) < 0
            if neg isa Bool
                print(io, real(v))
            else
                print(io, "(", v, ")")
            end
        else
            print(io, "(", v, ")")
        end
    end
    for (n, x) in enumerate(x.factors)
        if print_coeff || n > 1
            print(io, "*")
        end
        print(io, x)
    end
end
Base.iszero(x::FermionMul) = iszero(x.coeff)

Base.:(==)(a::FermionMul, b::Number) = isscalar(a) && a.coeff == b
Base.:(==)(a::FermionMul, b::FermionMul) = a.coeff == b.coeff && a.factors == b.factors
Base.:(==)(a::FermionMul, b::AbstractFermionSym) = isone(a.coeff) && length(a.factors) == 1 && only(a.factors) == b
Base.:(==)(b::AbstractFermionSym, a::FermionMul) = a == b
Base.hash(a::FermionMul, h::UInt) = hash(a.coeff, hash(a.factors, h))
FermionMul(f::FermionMul) = f
FermionMul(f::AbstractFermionSym) = FermionMul(1, [f])
mutable struct FermionAdd{C,D}
    coeff::C
    dict::D
end
Base.:(==)(a::FermionAdd, b::FermionAdd) = a.coeff == b.coeff && a.dict == b.dict
Base.hash(a::FermionAdd, h::UInt) = hash(a.coeff, hash(a.dict, h))
function filter_scalars!(f::FermionAdd)
    for (k, v) in f.dict
        if isscalar(k)
            f.coeff += k.coeff * v
            delete!(f.dict, k)
        end
    end
    f
end
function canonicalize!(_f::FermionAdd{_C,D}, filter_scalars=true) where {_C,D}
    C = promote_type(_C, valtype(D))
    f = FermionAdd(C(_f.coeff), _f.dict)
    filter_scalars && filter_scalars!(f)
    if length(f.dict) == 0
        return f.coeff
    elseif length(f.dict) == 1 && iszero(f.coeff)
        k, v = only(f.dict)
        return v * k
    else
        return FermionAdd{C,D}(f.coeff, f.dict)
    end
end
canonicalize!(a::AbstractFermionSym) = a
canonicalize!(a::Number) = a
const SM = Union{AbstractFermionSym,FermionMul}
const SMA = Union{AbstractFermionSym,FermionMul,FermionAdd}

function show_compact_sum(io, x::FermionAdd, max_terms=3)
    println(io, "Sum with ", length(x.dict), " terms: ")
    N = min(max_terms, length(x.dict))
    args = sum(v * k for (k, v) in Iterators.take(pairs(x.dict), N))
    show(io, args)
    if N < length(x.dict)
        print(io, " + ...")
    end
    return nothing
end
function Base.show(io::IO, x::FermionAdd)
    if length(x.dict) > 3
        return show_compact_sum(io, x)
    end
    compact = get(io, :compact, false)
    args = arguments(x)
    print_one = !iszero(x.coeff)
    if print_one
        if isreal(x.coeff)
            print(io, real(x.coeff), "I")
        else
            print(io, "(", x.coeff, ")", "I")
        end
        args = args[2:end]
    end
    print_sign(s) = compact ? print(io, s) : print(io, " ", s, " ")
    for (n, arg) in enumerate(args)
        k = prod(arg.factors)
        v = arg.coeff
        should_print_sign = (n > 1 || print_one)
        if isreal(v)
            v = real(v)
            neg = v < 0
            if neg isa Bool
                if neg
                    print_sign("-")
                    print(io, -real(v) * k)
                else
                    should_print_sign && print_sign("+")
                    print(io, real(v) * k)
                end
            else
                should_print_sign && print_sign("+")
                print(io, "(", v, ")*", k)
            end
        else
            should_print_sign && print_sign("+")
            print(io, "(", v, ")*", k)
        end
    end
    return nothing
end
print_num(io::IO, x) = isreal(x) ? print(io, real(x)) : print(io, "(", x, ")")

Base.:+(a::Number, b::SM) = iszero(a) ? b : FermionAdd(a, to_add(b))
Base.:+(a::UniformScaling, b::SM) = iszero(a) ? b : FermionAdd(a.λ, to_add(b))
Base.:+(a::SM, b::Union{Number,UniformScaling}) = b + a
Base.:+(a::FermionMul, b::AbstractFermionSym) = a + 1 * b
Base.:+(a::AbstractFermionSym, b::FermionMul) = 1 * a + b
Base.:+(a::AbstractFermionSym, b::AbstractFermionSym) = (1 * a) + (1 * b)
function Base.:+(a::FermionMul{CA,KA}, b::FermionMul{CB,KB}) where {CA,CB,KA,KB}
    C = promote_type(CA, CB)
    K = Union{FermionMul{C,KA},FermionMul{C,KB}}
    D = Dict{K,C}
    if a.factors == b.factors
        coeff = a.coeff + b.coeff
        return canonicalize!(FermionAdd(0, D(FermionMul(1, a.factors) => coeff)))
    end
    at, bt = to_add_tuple(a), to_add_tuple(b)
    return canonicalize!(FermionAdd(0, D(at..., bt...)))
end
Base.:+(a::SM, b::FermionAdd) = canonicalize!(FermionAdd(b.coeff, (_merge(+, to_add(a), b.dict; filter=iszero))))
function add!(a::FermionAdd, b::FermionAdd)
    a.coeff += b.coeff
    a.dict = __merge!(+, a.dict, b.dict; filter=iszero)
    return a
end
function add!(a::FermionAdd, b::SM)
    a.dict = __merge!(+, a.dict, to_add_tuple(b); filter=iszero)
    return a
end
add!(a::FermionAdd, b::Number) = (a.coeff += b; return a)
add!(a::FermionAdd, b::UniformScaling) = (a.coeff += b.λ; return a)
to_add(a::FermionMul, coeff=1) = Dict(FermionMul(1, a.factors) => a.coeff * coeff)
to_add(a::AbstractFermionSym, coeff=1) = Dict(FermionMul(a) => coeff)
to_add_tuple(a::FermionMul, coeff=1) = (FermionMul(1, a.factors) => a.coeff * coeff,)
to_add_tuple(a::AbstractFermionSym, coeff=1) = (FermionMul(a) => coeff,)

Base.:+(a::Number, b::FermionAdd) = iszero(a) ? b : FermionAdd(a + b.coeff, b.dict)
Base.:+(a::UniformScaling, b::FermionAdd) = iszero(a) ? b : FermionAdd(a.λ + b.coeff, b.dict)
Base.:+(a::FermionAdd, b::Union{Number,SM,UniformScaling}) = b + a
Base.:/(a::SMA, b::Number) = inv(b) * a
Base.:-(a::Union{Number,UniformScaling}, b::SMA) = a + (-b)
Base.:-(a::SMA, b::Union{Number,SMA,UniformScaling}) = a + (-b)
Base.:-(a::SMA) = -1 * a
function fermionterms(a::FermionAdd)
    (v * k for (k, v) in pairs(a.dict))
end
fermionterms(a::FermionMul) = (a,)
fermionterms(a::AbstractFermionSym) = (a,)
function allterms(a::FermionAdd)
    [a.coeff, [v * k for (k, v) in pairs(a.dict)]...]
end
function Base.:+(a::FermionAdd, b::FermionAdd)
    coeff = a.coeff + b.coeff
    dict = _merge(+, a.dict, b.dict; filter=iszero)
    FermionAdd(coeff, dict)
end
Base.:^(a::Union{FermionMul,FermionAdd}, b) = Base.power_by_squaring(a, b)

ordered_product(x::Number, y::Number, ordering) = x * y
ordered_product(x::Number, a::AbstractFermionSym, ordering) = iszero(x) ? 0 : FermionMul(x, [a])
ordered_product(x::Number, a::FermionMul, ordering) = iszero(x) ? 0 : FermionMul(x * a.coeff, a.factors)
ordered_product(x::Number, a::FermionAdd, ordering) = iszero(x) ? 0 : FermionAdd(x * a.coeff, Dict(k => v * x for (k, v) in a.dict))
ordered_product(a::FermionMul, b::FermionMul, ::NaiveOrdering) = FermionMul(a.coeff * b.coeff, vcat(a.factors, b.factors))
ordered_product(a::SMA, x::Number, ordering) = ordered_product(x, a, ordering)

ordered_product(a::AbstractFermionSym, bs::FermionMul, ordering) = ordered_product(1 * a, bs, ordering)
ordered_product(as::FermionMul, b::AbstractFermionSym, ordering) = ordered_product(as, 1 * b, ordering)
ordered_product(as::FermionMul, bs::FermionMul, ::NormalOrdering) = canonicalize!(normal_order(ordered_product(as, bs, NaiveOrdering())))
additive_coeff(a::FermionAdd) = a.coeff
additive_coeff(a::SM) = 0

Base.:*(a::SMA, b::SMA) = ordered_product(a, b, NormalOrdering())
Base.:*(a::SMA, x::Number) = ordered_product(x, a, NaiveOrdering())
Base.:*(x::Number, a::SMA) = ordered_product(x, a, NaiveOrdering())
function ordered_product(a::FermionAdd, b::SM, ordering::AbstractOrdering)
    c = zero(a)
    return trymul!(c, a, b, ordering)
end
function ordered_product(a::SM, b::FermionAdd, ordering)
    c = zero(b)
    return trymul!(c, a, b, ordering)
end
function ordered_product(a::FermionAdd, b::FermionAdd, ordering)
    c = zero(a)
    return trymul!(c, a, b, ordering)
end
function trymul!(c::FermionAdd, a::SMA, b::SMA, ordering)
    acoeff = additive_coeff(a)
    bcoeff = additive_coeff(b)
    if !iszero(acoeff)
        c = tryadd!(c, ordered_product(acoeff, b, ordering))
    end
    if !iszero(bcoeff)
        c = tryadd!(c, ordered_product(a, bcoeff, ordering))
        c = tryadd!(c, -acoeff * bcoeff) # We've double counted this term so subtract it
    end
    for bterm in fermionterms(b)
        for aterm in fermionterms(a)
            newterm = ordered_product(aterm, bterm, ordering)
            c = tryadd!(c, newterm)
        end
    end
    return canonicalize!(c)
end

"""
    tryadd!(c::FermionAdd, term)

This function tries to add `term` to `c` in place. If it fails, it catches the error and returns the out of place sum.
"""
function tryadd!(c::FermionAdd, term)
    try
        return add!(c, term)
    catch e
        @debug e
    end
    c + term
end
tryadd!(c::SM, term) = c + term
tryadd!(c::Number, term) = c + term


Base.adjoint(x::FermionMul) = length(x.factors) == 0 ? FermionMul(adjoint(x.coeff), x.factors) : adjoint(x.coeff) * foldr(*, Iterators.reverse(Iterators.map(adjoint, x.factors)))
function Base.adjoint(x::FermionAdd)
    newx = zero(x)
    newx.coeff = adjoint(x.coeff)
    for (f, v) in x.dict
        add!(newx, v' * f')
    end
    newx
end
Base.zero(::FermionAdd{C,D}) where {C,D} = FermionAdd(zero(C), D())

function sorted_noduplicates(v)
    I = eachindex(v)
    for i in I[1:end-1]
        isequal(v[i], v[i+1]) && return false
    end
    return true
end

## Normal ordering
function bubble_sort(a::FermionAdd; start=1)
    c = zero(a)
    c.coeff = a.coeff
    for term in fermionterms(a)
        add!(c, bubble_sort(term; start))
    end
    return c
end

function bubble_sort(a::FermionMul; start=1)
    if length(a.factors) == 1
        return a
    end
    muloraddvec::Union{Number,SMA} = a

    swapped = false
    i = max(0, start - 1)
    triple_prod(a, b, c) = ordered_product(ordered_product(a, b, NaiveOrdering()), c, NaiveOrdering())
    while !swapped && i < length(eachindex(a.factors)) - 1
        i += 1
        if a.factors[i] > a.factors[i+1] || isequal(a.factors[i], a.factors[i+1])
            swapped = true
            product = ordered_product(a.factors[i], a.factors[i+1], NormalOrdering())
            left_factors = FermionMul(a.coeff, a.factors[1:i-1])
            right_factors = FermionMul(1, a.factors[i+2:end])
            muloraddvec = triple_prod(left_factors, product, right_factors)
        end
    end
    if !swapped
        return a
    end
    bubble_sort(muloraddvec; start=i - 1)
end

normal_order(a::SMA) = bubble_sort(a)
normal_order(a::Number) = a
bubble_sort(a::Number; kwargs...) = a

isscalar(x::FermionMul) = iszero(x.coeff) || (length(x.factors) == 0)
isscalar(x::FermionAdd) = length(x.dict) == 0 || all(isscalar, keys(x.dict)) || all(iszero(values(x.dict)))
isscalar(x::AbstractFermionSym) = false

Base.valtype(::FermionMul{C}) where C = C
Base.valtype(op::FermionAdd{C}) where {C} = promote_type(C, valtype(op.dict), valtype(first(keys(op.dict))))
Base.valtype(::AbstractFermionSym) = Int

## Instantiating sparse matrices
_labels(a::FermionMul) = [s.label for s in a.factors]
matrix_representation(op, H::AbstractFockHilbertSpace) = matrix_representation(op, mode_ordering(H), basisstates(H), Dict(Iterators.map(reverse, enumerate(basisstates(H)))))
matrix_representation(op::Number, H::AbstractFockHilbertSpace) = op * I(size(H, 1))

function matrix_representation(op::Union{<:FermionMul,<:AbstractFermionSym}, labels, states, fock_to_ind)
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

function operator_inds_amps_free_fermion!((outinds, ininds, amps), op::FermionMul, ordering, states::AbstractVector{SingleParticleState}, fock_to_ind)
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

function operator_inds_amps_generic!((outinds, ininds, amps), op::FermionMul, ordering, states, fock_to_ind)
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

function matrix_representation(op::FermionAdd{C}, ordering, states, fock_to_ind) where C
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
operator_inds_amps!((outinds, ininds, amps), op::AbstractFermionSym, args...; kwargs...) = operator_inds_amps!((outinds, ininds, amps), FermionMul(1, [op]), args...; kwargs...)

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
eval_in_basis(a::FermionMul, f) = a.coeff * mapfoldl(Base.Fix2(eval_in_basis, f), *, a.factors)
eval_in_basis(a::FermionAdd, f) = a.coeff * I + mapfoldl(Base.Fix2(eval_in_basis, f), +, fermionterms(a))

## 
remove_identity(a::SM) = a
remove_identity(a::FermionAdd) = FermionAdd(zero(a.coeff), a.dict)

##
TermInterface.head(a::Union{FermionMul,FermionAdd}) = operation(a)
TermInterface.iscall(::Union{FermionMul,FermionAdd}) = true
TermInterface.isexpr(::Union{FermionMul,FermionAdd}) = true

TermInterface.operation(::FermionMul) = (*)
TermInterface.operation(::FermionAdd) = (+)
TermInterface.arguments(a::FermionMul) = [a.coeff, a.factors...]
TermInterface.arguments(a::FermionAdd) = iszero(a.coeff) ? fermionterms(a) : allterms(a)
TermInterface.sorted_arguments(a::FermionAdd) = iszero(a.coeff) ? sort(fermionterms(a), by=x -> x.factors) : [a.coeff, sort(fermionterms(a); by=x -> x.factors)...]
TermInterface.children(a::Union{FermionMul,FermionAdd}) = arguments(a)
TermInterface.sorted_children(a::Union{FermionMul,FermionAdd}) = sorted_arguments(a)

TermInterface.maketerm(::Type{<:FermionMul}, ::typeof(*), args, metadata) = *(args...)
TermInterface.maketerm(::Type{<:FermionAdd}, ::typeof(+), args, metadata) = +(args...)

TermInterface.head(::T) where {T<:AbstractFermionSym} = T
TermInterface.iscall(::AbstractFermionSym) = true
TermInterface.isexpr(::AbstractFermionSym) = true
TermInterface.maketerm(::Type{Q}, head::Type{T}, args, metadata) where {Q<:Union{AbstractFermionSym,<:FermionMul,<:FermionAdd},T<:AbstractFermionSym} = T(args...)


#From SymbolicUtils
function _merge(f, d, others...; filter=x -> false)
    T = Union{promote_type(valtype(d), valtype.(others)...)}
    K = Union{keytype(d),keytype.(others)...}
    d2 = Dict{K,T}(d)
    return __merge!(f, d2, others...; filter)
end
function __merge!(f::F, d, others...; filter=x -> false) where {F}
    acc = d
    for other in others
        for (k, v) in other
            v = f(v)
            ak = get(acc, k, nothing)
            if ak !== nothing
                v = ak + v
            end
            if filter(v)
                delete!(acc, k)
            else
                acc[k] = v
            end
        end
    end
    acc
end


## Symmetries
isnumberconserving(x::AbstractFermionSym) = false
isnumberconserving(x::FermionMul) = iszero(sum(s -> 2s.creation - 1, x.factors))
isnumberconserving(x::FermionAdd) = all(isnumberconserving, fermionterms(x))

isparityconserving(x::AbstractFermionSym) = false
isparityconserving(x::FermionMul) = iseven(length(x.factors))
isparityconserving(x::FermionAdd) = all(isparityconserving, fermionterms(x))

isquadratic(::AbstractFermionSym) = false
isquadratic(x::FermionMul) = length(x.factors) == 2
isquadratic(x::FermionAdd) = all(isquadratic, fermionterms(x))

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

Base.copy(x::FermionAdd) = FermionAdd(copy(x.coeff), copy(x.dict))
@testitem "Consistency between + and add!" begin
    import FermionicHilbertSpaces: add!
    @fermions f
    a = 1.0 * f[2] * f[1] + 1 + f[1]
    for b in [1.0, 1, f[1], 1.0 * f[1], f[2] * f[1], a]
        a2 = copy(a)
        a3 = add!(a2, b)
        @test a + b == a3
        @test a2 == a3
        @test_throws InexactError add!(a, 1im * b)
    end
    @test a == 1.0 * f[2] * f[1] + 1 + f[1]
end
