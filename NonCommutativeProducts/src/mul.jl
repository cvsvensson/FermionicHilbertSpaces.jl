
mutable struct NCMul{C,S,F}
    coeff::C
    factors::F
    function NCMul(coeff::C, factors::F) where {C,F}
        new{C,eltype(factors),F}(coeff, factors)
    end
end
Base.convert(::Type{NCMul{C,S,F}}, x::NCMul{<:Any,S,F}) where {C,S,F} = NCMul(convert(C, x.coeff), x.factors)
Base.convert(::Type{NCMul{C,S,F}}, x::NCMul) where {C,S,F} = NCMul(convert(C, x.coeff), F(x.factors))
Base.copy(x::NCMul) = NCMul(copy(x.coeff), copy(x.factors))
isscalar(x::NCMul) = length(x.factors) == 0
Base.promote_rule(::Type{NCMul{C,S,F}}, x::Type{NCMul{C2,S2,F2}}) where {C,S,F,C2,S2,F2} = NCMul{promote_type(C, C2),promote_type(S, S2),promote_type(F, F2)}

function Base.show(io::IO, x::NCMul)
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
Base.iszero(x::NCMul) = iszero(x.coeff)

Base.:(==)(a::NCMul, b::Number) = isscalar(a) && a.coeff == b
Base.:(==)(a::NCMul, b::NCMul) = a.coeff == b.coeff && a.factors == b.factors
Base.hash(a::NCMul, h::UInt) = isone(a.coeff) && length(a.factors) == 1 ? hash(only(a.factors), h) : hash(a.coeff, hash(a.factors, h))
NCMul(f::NCMul) = f

NCterms(a::NCMul) = (a,)
Base.:-(a::NCMul) = NCMul(-a.coeff, a.factors)

Base.:*(x::Number, a::NCMul) = NCMul(x * a.coeff, a.factors)
Base.:*(m::NCMul, x::Number) = x * m
Base.:*(a::NCMul, b::NCMul) = NCMul(a.coeff * b.coeff, vcat(a.factors, b.factors))

function Base.adjoint(x::NCMul)
    length(x.factors) == 0 && return NCMul(adjoint(x.coeff), x.factors)
    ncmul = NCMul(adjoint(x.coeff), collect(Iterators.reverse(Iterators.map(adjoint, x.factors))))
    if eager(x)
        return bubble_sort(ncmul, Ordering(ncmul))
    end
    return ncmul
end