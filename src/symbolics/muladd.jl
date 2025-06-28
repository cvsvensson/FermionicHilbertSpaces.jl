abstract type AbstractFermionSym end
Base.:*(a::AbstractFermionSym, b::AbstractFermionSym) = ordered_prod(a, b)
unordered_prod(a::AbstractFermionSym, b::AbstractFermionSym) = FermionMul(1, [a, b])

struct FermionMul{C,F<:AbstractFermionSym}
    coeff::C
    factors::Vector{F}
    ordered::Bool
    function FermionMul(coeff::C, factors) where {C}
        ordered = issorted(factors) && sorted_noduplicates(factors)
        new{C,eltype(factors)}(coeff, factors, ordered)
    end
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

struct FermionAdd{C,D}
    coeff::C
    dict::D
    function FermionAdd(coeff::C, dict::D) where {C,D}
        for (k, v) in dict
            if isscalar(k)
                coeff += k.coeff * v
            end
        end
        dict = filter(p -> !(isscalar(first(p))), dict)
        if length(dict) == 0
            coeff
        elseif length(dict) == 1 && iszero(coeff)
            k, v = first(dict)
            v * k
        else
            new{typeof(coeff),D}(coeff, dict)
        end
    end
end
Base.:(==)(a::FermionAdd, b::FermionAdd) = a.coeff == b.coeff && a.dict == b.dict
Base.hash(a::FermionAdd, h::UInt) = hash(a.coeff, hash(a.dict, h))

const SM = Union{AbstractFermionSym,FermionMul}
const SMA = Union{AbstractFermionSym,FermionMul,FermionAdd}

function show_compact_sum(io, x::FermionAdd, max_terms=3)
    println(io, "Sum with ", length(x.dict), " terms: ")
    N = min(max_terms, length(x.dict))
    args = sum(sorted_arguments(x)[1:N])
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
    args = sorted_arguments(x)
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
Base.:+(a::SM, b::Number) = b + a
Base.:+(a::SM, b::SM) = FermionAdd(0, (_merge(+, to_add(a), to_add(b); filter=iszero)))
Base.:+(a::SM, b::FermionAdd) = FermionAdd(b.coeff, (_merge(+, b.dict, to_add(a); filter=iszero)))
Base.:+(a::FermionAdd, b::SM) = b + a
Base.:/(a::SMA, b::Number) = inv(b) * a
to_add(a::FermionMul, coeff=1) = Dict(FermionMul(1, a.factors) => a.coeff * coeff)
to_add(a::AbstractFermionSym, coeff=1) = Dict(FermionMul(a) => coeff)

Base.:+(a::Number, b::FermionAdd) = iszero(a) ? b : FermionAdd(a + b.coeff, b.dict)
Base.:+(a::FermionAdd, b::Number) = b + a
Base.:-(a::Number, b::SMA) = a + (-b)
Base.:-(a::SMA, b::Number) = a + (-b)
Base.:-(a::SMA, b::SMA) = a + (-b)
Base.:-(a::SMA) = -1 * a
function fermionterms(a::FermionAdd)
    [v * k for (k, v) in pairs(a.dict)]
end
function allterms(a::FermionAdd)
    [a.coeff, [v * k for (k, v) in pairs(a.dict)]...]
end
function Base.:+(a::FermionAdd, b::FermionAdd)
    a.coeff + foldr((f, b) -> f + b, fermionterms(a); init=b)
end
Base.:^(a::Union{FermionMul,FermionAdd}, b) = Base.power_by_squaring(a, b)


Base.:*(x::Number, a::AbstractFermionSym) = iszero(x) ? 0 : FermionMul(x, [a])
Base.:*(x::Number, a::FermionMul) = iszero(x) ? 0 : FermionMul(x * a.coeff, a.factors)
Base.:*(x::Number, a::FermionAdd) = iszero(x) ? 0 : FermionAdd(x * a.coeff, Dict(k => v * x for (k, v) in collect(a.dict)))
Base.:*(a::SMA, x::Number) = x * a

Base.:*(a::AbstractFermionSym, bs::FermionMul) = (1 * a) * bs
Base.:*(as::FermionMul, b::AbstractFermionSym) = (b' * as')'
Base.:*(as::FermionMul, bs::FermionMul) = order_mul(unordered_prod(as, bs))
Base.adjoint(x::FermionMul) = length(x.factors) == 0 ? FermionMul(adjoint(x.coeff), x.factors) : adjoint(x.coeff) * foldr(*, reverse(adjoint.(x.factors)))
Base.:*(a::FermionAdd, b::SM) = (b' * a')'
function Base.:*(a::SM, b::FermionAdd)
    a * b.coeff + sum((v * a) * f for (f, v) in b.dict)
end
function Base.:*(a::FermionAdd, b::FermionAdd)
    a.coeff * b + sum((va * fa) * b for (fa, va) in a.dict)
end

Base.adjoint(x::FermionAdd) = adjoint(x.coeff) + sum(f' for f in fermionterms(x))


unordered_prod(a::FermionMul, b::FermionAdd) = b.coeff * a + sum(unordered_prod(a, f) for f in fermionterms(b))
unordered_prod(a::FermionAdd, b::FermionMul) = a.coeff * b + sum(unordered_prod(f, b) for f in fermionterms(a))
unordered_prod(a::FermionAdd, b::FermionAdd) = sum(unordered_prod(f, g) for f in allterms(a), g in allterms(b))
unordered_prod(a::FermionMul, b::FermionMul) = FermionMul(a.coeff * b.coeff, vcat(a.factors, b.factors))
unordered_prod(a, b, xs...) = foldl(*, xs; init=(*)(a, b))
unordered_prod(x::Number, a::SMA) = x * a
unordered_prod(a::SMA, x::Number) = x * a
unordered_prod(x::Number, y::Number) = x * y

function sorted_noduplicates(v)
    I = eachindex(v)
    for i in I[1:end-1]
        isequal(v[i], v[i+1]) && return false
    end
    return true
end

## Normal ordering
ordering_product(ordered_leftmul::Number, right_mul) = ordered_leftmul * order_mul(right_mul)
bubble_sort(a::FermionAdd) = a.coeff + sum(bubble_sort(f) for f in fermionterms(a))

function bubble_sort(a::FermionMul)
    if a.ordered || length(a.factors) == 1
        return a
    end
    swapped = true
    muloraddvec::Union{Number,SMA} = a

    swapped = false
    i = first(eachindex(a.factors)) - 1
    while !swapped && i < length(eachindex(a.factors)) - 1
        i += 1
        if a.factors[i] > a.factors[i+1] || isequal(a.factors[i], a.factors[i+1])
            swapped = true
            product = a.factors[i] * a.factors[i+1]
            left_factors = FermionMul(a.coeff, a.factors[1:i-1])
            right_factors = FermionMul(1, a.factors[i+2:end])
            muloraddvec = unordered_prod(left_factors, product, right_factors)
        end
    end
    bubble_sort(muloraddvec)
end
bubble_sort(a::Number) = a

order_mul(a::FermionMul) = bubble_sort(a)
order_mul(x::Number) = x

isscalar(x::FermionMul) = iszero(x.coeff) || (length(x.factors) == 0)
isscalar(x::FermionAdd) = length(x.dict) == 0 || all(isscalar, keys(x.dict)) || all(iszero(values(x.dict)))
isscalar(x::AbstractFermionSym) = false

Base.valtype(::FermionMul{C}) where C = C
Base.valtype(op::FermionAdd{C}) where {C} = promote_type(C, valtype(op.dict), valtype(first(keys(op.dict))))
Base.valtype(::AbstractFermionSym) = Int

## Instantiating sparse matrices
_labels(a::FermionMul) = [s.label for s in a.factors]
matrix_representation(op, H::AbstractFockHilbertSpace) = matrix_representation(op, H.jw.ordering, focknumbers(H), focknumbers(H))
matrix_representation(op::Number, H::AbstractFockHilbertSpace) = op * I(size(H, 1))

function matrix_representation(op::Union{<:FermionMul,<:AbstractFermionSym}, labels, outstates, instates)
    outinds = Int[]
    ininds = Int[]
    AT = valtype(op)
    amps = AT[]
    sizehint!(outinds, length(instates))
    sizehint!(ininds, length(instates))
    sizehint!(amps, length(instates))
    operator_inds_amps!((outinds, ininds, amps), op, labels, outstates, instates)
    SparseArrays.sparse!(outinds, ininds, identity.(amps), length(outstates), length(instates))
end
matrix_representation(op, labels, instates) = matrix_representation(op, labels, instates, instates)

function operator_inds_amps!((outinds, ininds, amps), op::FermionMul{C}, label_to_site_index, outstates, instates; fock_to_outind=Dict(map(reverse, enumerate(outstates)))) where {C}
    digitpositions = collect(Iterators.reverse(label_to_site_index[f.label] for f in op.factors)) #reverse(siteindices(_labels(op), jw))
    daggers = collect(Iterators.reverse(s.creation for s in op.factors))
    mc = -op.coeff
    pc = op.coeff
    for (n, f) in enumerate(instates)
        newfockstate, amp = togglefermions(digitpositions, daggers, f)
        if !iszero(amp)
            push!(outinds, fock_to_outind[newfockstate])
            push!(amps, amp == 1 ? pc : mc)
            push!(ininds, n)
        end
    end
    return (outinds, ininds, amps)
end

# promote_array(v) = convert(Array{eltype(promote(map(zero, unique(typeof(v) for v in v))...))}, v)

function matrix_representation(op::FermionAdd{C}, label_to_site_index, outstates, instates) where C
    fock_to_outind = Dict(Iterators.map(reverse, enumerate(outstates)))
    outinds = Int[]
    ininds = Int[]
    AT = valtype(op)
    amps = AT[]
    N = length(op.dict)
    sizehint!(outinds, N * length(instates))
    sizehint!(ininds, N * length(instates))
    sizehint!(amps, N * length(instates))
    for (factor, coeff) in op.dict
        operator_inds_amps!((outinds, ininds, amps), coeff * factor, label_to_site_index, outstates, instates; fock_to_outind=fock_to_outind)
    end
    if !iszero(op.coeff)
        append!(ininds, eachindex(instates))
        append!(outinds, eachindex(outstates))
        append!(amps, fill(op.coeff, length(instates)))
    end
    return SparseArrays.sparse!(outinds, ininds, amps, length(outstates), length(instates))
end
operator_inds_amps!((outinds, ininds, amps), op::AbstractFermionSym, label_to_site_index, outstates, instates; kwargs...) = operator_inds_amps!((outinds, ininds, amps), FermionMul(1, [op]), label_to_site_index, outstates, instates; kwargs...)

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

    @test all(matrix_representation(sum(f[l]' * f[l] for l in labels), H.jw, FermionicHilbertSpaces.fixed_particle_number_fockstates(N, n)) == n * I for n in 1:N)

    @test all(eval_in_basis(f[l], fmb) == fmb[l] for l in labels)
    @test all(eval_in_basis(f[l]', fmb) == fmb[l]' for l in labels)
    @test all(eval_in_basis(f[l]' * f[l], fmb) == fmb[l]'fmb[l] for l in labels)
    @test all(eval_in_basis(f[l] + f[l]', fmb) == fmb[l] + fmb[l]' for l in labels)
end

## Convert to expression
eval_in_basis(a::FermionMul, f) = a.coeff * mapfoldl(Base.Fix2(eval_in_basis, f), *, a.factors)
eval_in_basis(a::FermionAdd, f) = a.coeff * I + mapfoldl(Base.Fix2(eval_in_basis, f), +, fermionterms(a))

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
# _merge(f::F, d, others...; filter=x -> false) where {F} = _merge!(f, Dict{SM,Any}(d), others...; filter=filter)
_merge(f::F, d, others...; filter=x -> false) where {F} = _merge!(f, Dict{SM,Union{promote_type(valtype(d), valtype.(others)...)}}(d), others...; filter=filter)

function _merge!(f::F, d, others...; filter=x -> false) where {F}
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
