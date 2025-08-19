function filter_scalars!(d::Dict{K,V}) where {K<:NCMul,V}
    coeff = zero(V)
    for (k, v) in d
        if isscalar(k)
            coeff += k.coeff * v
            delete!(d, k)
        end
    end
    return coeff
end
function filter_zeros!(d::Dict{K,V}) where {K<:NCMul,V}
    for (k, v) in d
        if iszero(v)
            delete!(d, k)
        end
    end
    return d
end

mutable struct NCAdd{C,K,D}
    coeff::C
    dict::D
    function NCAdd(coeff::C, dict::Dict{K,C}; keep_zeros=false) where {C,K}
        keep_zeros || filter_zeros!(dict)
        coeff += filter_scalars!(dict)
        new{C,K,Dict{K,C}}(coeff, dict)
    end
end
function NCAdd(coeff::C, dict::Dict{K,V}; kwargs...) where {C,K,V}
    T = promote_type(C, V)
    NCAdd(T(coeff), Dict{K,T}(dict); kwargs...)
end
const MulAdd = Union{NCMul,NCAdd}
function filter_scalars!(x::NCAdd)
    add!!(x, filter_scalars!(x.dict))
end
filter_zeros!(x::NCAdd) = (filter_zeros!(x.dict); return x)
Base.iszero(x::NCAdd) = iszero(x.coeff) && all(iszero, values(x.dict))
Base.:(==)(a::NCAdd, b::NCAdd) = a.coeff == b.coeff && a.dict == b.dict
Base.:(==)(a::NCAdd, b::Number) = a.coeff == b && isempty(a.dict)
Base.:(==)(a::Number, b::NCAdd) = a == b.coeff && isempty(b.dict)
function Base.hash(a::NCAdd, h::UInt)
    # if it's only a number, hash should equal hash of number
    if isempty(a.dict)
        return hash(a.coeff, h)
    end
    # if coeff is zero and there is only one term, it should hash equals to the corresponding NCMul
    if iszero(a.coeff) && length(a.dict) == 1
        ncmul, coeff = only(a.dict)
        return hash(NCMul(coeff, ncmul.factors), h)
    end
    return hash(a.coeff, hash(a.dict, h))
end

isscalar(x::NCAdd) = length(x.dict) == 0 || all(isscalar, keys(x.dict)) || all(iszero(values(x.dict)))
Base.copy(x::NCAdd) = NCAdd(copy(x.coeff), copy(x.dict))

function show_compact_sum(io, x::NCAdd, max_terms=3)
    println(io, "Sum with ", length(x.dict), " terms: ")
    N = min(max_terms, length(x.dict))
    args = sum(v * k for (k, v) in Iterators.take(pairs(x.dict), N))
    show(io, args)
    if N < length(x.dict)
        print(io, " + ...")
    end
    return nothing
end
function Base.show(io::IO, x::NCAdd)
    if length(x.dict) > 3
        return show_compact_sum(io, x)
    end
    compact = get(io, :compact, false)
    print_one = !iszero(x.coeff) || length(x.dict) == 0
    if print_one
        if isreal(x.coeff)
            print(io, real(x.coeff), "I")
        else
            print(io, "(", x.coeff, ")", "I")
        end
        # args = args[2:end]
    end
    print_sign(s) = compact ? print(io, s) : print(io, " ", s, " ")
    for (n, (k, v)) in enumerate(pairs(x.dict))
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


Base.:+(a::Number, b::NCAdd) = iszero(a) ? b : NCAdd(a + b.coeff, b.dict)
Base.:+(a::UniformScaling, b::NCAdd) = iszero(a) ? b : NCAdd(a.λ + b.coeff, b.dict)
Base.:+(a::NCAdd, b::B) where B<:Union{Number,UniformScaling} = b + a
Base.:+(a::NCAdd, b::B) where B<:NCMul = b + a
Base.:/(a::MulAdd, b::Number) = inv(b) * a
Base.:-(a::Union{Number,UniformScaling}, b::MulAdd) = a + (-b)
Base.:-(a::MulAdd, b::Union{Number,MulAdd,UniformScaling}) = a + (-b)
Base.:-(a::NCAdd) = NCAdd(-a.coeff, Dict(k => -v for (k, v) in pairs(a.dict)))
function Base.:+(a::NCAdd, b::NCAdd)
    coeff = a.coeff + b.coeff
    dict = mergewith(+, a.dict, b.dict)
    NCAdd(coeff, dict)
end


function add!!(a::NCAdd, b::NCMul)
    key = NCMul(1, b.factors)
    coeff = b.coeff
    newdict, ret = modify!!(a.dict, key) do val
        isnothing(val) && return coeff
        return something(val, 0) + coeff
    end
    newdict === a.dict && return a
    return NCAdd(a.coeff, newdict)
end
function add!!(a::NCAdd, b::NCAdd)
    newdict = mergewith!!(+, a.dict, b.dict)
    newdict === a.dict && return add!!(a, b.coeff)
    return NCAdd(a.coeff + b.coeff, newdict)
end
function add!!(a::NCAdd{C}, b::C2) where {C,C2<:Number}
    promote_type(C, C2) <: C && (a.coeff += b; return a)
    return a + b
end
function add!!(a::NCAdd{C}, b::UniformScaling{C2}) where {C,C2<:Number}
    promote_type(C, C2) <: C && return (a.coeff += b.λ; return a)
    return a + b
end


function NCterms(a::NCAdd)
    (v * k for (k, v) in pairs(a.dict))
end
additive_coeff(a::NCAdd) = a.coeff
additive_coeff(a::NCMul) = 0

Base.:*(x::Number, a::NCAdd) = NCAdd(x * a.coeff, Dict(k => v * x for (k, v) in a.dict))
Base.:*(a::NCAdd, x::Number) = x * a

function Base.:*(a::NCAdd, b::NCMul)
    c = zero(a)
    filter_scalars!(filter_zeros!(mul!!(c, a, b)))
end
function Base.:*(a::NCMul, b::NCAdd)
    c = zero(b)
    filter_scalars!(filter_zeros!(mul!!(c, a, b)))
end
function Base.:*(a::NCAdd, b::NCAdd)
    c = zero(a)
    filter_scalars!(filter_zeros!(mul!!(c, a, b)))
end
function mul!!(c::NCAdd, a::MulAdd, b::MulAdd)
    acoeff = additive_coeff(a)
    bcoeff = additive_coeff(b)
    if !iszero(acoeff)
        c = add!!(c, acoeff * b)
    end
    if !iszero(bcoeff)
        c = add!!(c, a * bcoeff)
        c = add!!(c, -acoeff * bcoeff) # We've double counted this term so subtract it
    end
    for bterm in NCterms(b)
        for aterm in NCterms(a)
            newterm = cat(aterm, bterm)
            c = add!!(c, newterm)
        end
    end
    if eager(c)
        c = bubble_sort!(c, Ordering(c))
    end
    return c
end
add!!(c::NCMul, term) = c + term
add!!(c::Number, term) = c + term

function Base.adjoint(x::NCAdd)
    newx = zero(x)
    newx.coeff = adjoint(x.coeff)
    for (f, v) in x.dict
        newx = add!!(newx, v' * f')
    end
    newx
end
Base.zero(::NCAdd{C,K,D}) where {C,K,D} = NCAdd(zero(C), D())

function sorted_noduplicates(v)
    I = eachindex(v)
    for i in I[1:end-1]
        isequal(v[i], v[i+1]) && return false
    end
    return true
end
