NCMul(f::NCAdd) = (length(f.dict) == 1 && iszero(f.coeff) && return prod(only(f.dict))) || throw(ArgumentError("Cannot convert NCAdd to NCMul: $f"))


function Base.:(==)(a::NCAdd, b::NCMul)
    iszero(a.coeff) || return false
    length(a.dict) == 1 || return false
    ncmul, coeff = only(a.dict)
    NCMul(coeff, ncmul.factors) == b
end
Base.:(==)(a::NCMul, b::NCAdd) = b == a

function Base.:+(_a::NCMul, _b::NCMul)
    a, b = promote(_a, _b)
    if a.factors == b.factors
        return NCAdd(0, Dict(NCMul(1, a.factors) => a.coeff + b.coeff))
    end
    return NCAdd(0, Dict(NCMul(1, a.factors) => a.coeff, NCMul(1, b.factors) => b.coeff))
end

Base.:+(a::Number, b::NCMul) = NCAdd(a, to_add(b))
Base.:+(a::UniformScaling, b::NCMul) = NCAdd(a.λ, to_add(b))
Base.:+(a::NCMul, b::Union{Number,UniformScaling}) = b + a
Base.:+(a::NCMul, b::NCAdd) = NCAdd(b.coeff, mergewith!!(+, to_add(a), b.dict))

to_add(a::NCMul, coeff=1) = Dict(NCMul(1, a.factors) => a.coeff * coeff)
# to_add(a, coeff=1) = Dict(NCMul(a) => coeff)
# to_add_tuple(a::NCMul, coeff=1) = (NCMul(1, a.factors) => a.coeff * coeff,)
# to_add_tuple(a, coeff=1) = (NCMul(a) => coeff,)


function Base.:^(a::Union{NCAdd,NCMul}, b::Int)
    ret = Base.power_by_squaring(a, b)
    eager(ret) && return bubble_sort(ret, Ordering(ret))
    return ret
end

macro nc0(T)
    quote
        NonCommutativeProducts.NCMul(f::$(esc(T))) = NCMul(1, [f])

        Base.:+(x::$(esc(T)), y::$(esc(T))) = NCMul(x) + NCMul(y)
        Base.:+(x::$(esc(T)), y::Union{Number,UniformScaling,NCMul,NCAdd}) = NCMul(x) + y
        Base.:+(x::Union{Number,UniformScaling,NCMul,NCAdd}, y::$(esc(T))) = y + x

        NonCommutativeProducts.add!!(x::NCAdd, y::$(esc(T))) = add!!(x, NCMul(y))

        Base.:-(x::$(esc(T)), y::$(esc(T))) = NCMul(x) - NCMul(y)
        Base.:-(x::$(esc(T))) = NCMul(-1, [x])
        Base.:-(x::Union{Number,UniformScaling,NCAdd}, y::$(esc(T))) = x - NCMul(y)
        Base.:-(x::$(esc(T)), y::Union{Number,UniformScaling,NCAdd}) = NCMul(x) - y

        Base.:*(x::Union{Number,UniformScaling,NCAdd}, y::$(esc(T))) = x * NCMul(y)
        Base.:*(x::$(esc(T)), y::Union{Number,UniformScaling,NCAdd}) = NCMul(x) * y

        Base.:^(a::$(esc(T)), b) = NCMul(a)^b
        Base.convert(::Type{NCMul{C,S,F}}, x::$(esc(T))) where {C,S<:$(esc(T)),F} = NCMul(one(C), S[x])

        Base.:(==)(a::$(esc(T)), b::Union{NCMul,NCAdd}) = NCMul(a) == b
        Base.:(==)(a::Union{NCMul,NCAdd}, b::$(esc(T))) = b == a
    end
end
macro nc(T)
    quote
        @nc0 $(esc(T))
        Base.:*(x::$(esc(T)), y::$(esc(T))) = NCMul(1, [x, y])
        Base.:*(x::$(esc(T)), y::NCMul) = NCMul(y.coeff, pushfirst!!(copy(y.factors), x))
        Base.:*(x::NCMul, y::$(esc(T))) = NCMul(x.coeff, push!!(copy(x.factors), y))
        NonCommutativeProducts.eager(::A) where A<:$(esc(T)) = eager(A)
        NonCommutativeProducts.eager(::Type{<:$(esc(T))}) = false
    end
end

eager(::Type{NCMul{C,T,F}}) where {C,T,F} = eager(T)
eager(::NCMul{C,T}) where {C,T} = eager(T)
eager(::NCAdd{C,S}) where {C,S} = eager(S)

Ordering(::NCMul{C,T}) where {C,T} = Ordering(T)
Ordering(::Type{NCMul{C,T,F}}) where {C,T,F} = Ordering(T)
Ordering(::NCAdd{C,S}) where {C,S} = Ordering(S)

macro nc_eager(T, ordering)
    quote
        @nc0 $(esc(T))
        NonCommutativeProducts.Ordering(::A) where A<:$(esc(T)) = Ordering(A)
        NonCommutativeProducts.Ordering(::Type{<:$(esc(T))}) = $(esc(ordering))
        NonCommutativeProducts.eager(::A) where A<:$(esc(T)) = eager(A)
        NonCommutativeProducts.eager(::Type{<:$(esc(T))}) = true
        Base.:*(x::$(esc(T)), y::$(esc(T))) = bubble_sort!(NCMul(1, [x, y]), $(esc(ordering)))

        Base.:*(x::$(esc(T)), y::NCMul) = NCMul(x) * y
        Base.:*(x::NCMul, y::$(esc(T))) = x * NCMul(y)
    end
end

