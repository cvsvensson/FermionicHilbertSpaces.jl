
struct ScalarMul{T}
    λ::T
end
scalar(s::ScalarMul) = s.λ

struct AddTerms{T}
    terms::T
end
struct Swap{T}
    λ::T
end


function bubble_sort(a::NCMul, ordering; kwargs...)
    if length(a.factors) == 0
        return a
    end
    bubble_sort(NCAdd(0, to_add(a)), ordering; kwargs...)
end
function bubble_sort(ncadd::NCAdd, ordering)
    terms = collect(v * copy(k) for (k, v) in pairs(ncadd.dict))
    sorted_terms = bubble_sort!(terms, ordering)
    newadd = zero(ncadd)
    for term in sorted_terms
        newadd = add!!(newadd, term)
    end
    return filter_scalars!(filter_zeros!(ncadd.coeff + newadd))
end

function bubble_sort!(terms::AbstractVector{<:NCMul}, ordering)
    n = 1
    while n <= length(terms)
        done = false
        count = 0
        start = 1
        while !done && n <= length(terms)
            count += 1
            terms, done, start = bubble_sort!(terms, n, ordering; start)
            if iszero(terms[n].coeff)
                done = false
                start = 1
                deleteat!(terms, n)
            end
            # if count > 100
            #     @warn "Bubble sort took too long, stopping early"
            #     break
            # end
        end
        n += 1
    end
    return terms
end

function bubble_sort!(terms::AbstractVector{<:NCMul}, index, ordering; start=1)
    no_effect = true
    ncmul = terms[index]
    if length(ncmul.factors) <= 1
        done = true
        return terms, done, start
    end
    i = max(0, start - 1)
    while no_effect && i < length(ncmul.factors) - 1
        i += 1
        a, b = ncmul.factors[i], ncmul.factors[i+1]
        effect = mul_effect(a, b, ordering)
        isnothing(effect) && continue

        no_effect = false
        ncmul, newterms = splice!!_and_add(ncmul, i:i+1, effect)
        if length(newterms) > 0
            terms = append!!(terms, newterms)
        end
        terms = setindex!!(terms, ncmul, index)
    end
    done = no_effect
    newstart = i - 1
    return terms, done, newstart
end


function mysplice!!(v::V, i, replacement::W) where {V,W}
    T = promote_type(eltype(W), eltype(V))
    if T <: eltype(V)
        splice!(v, i, replacement)
        return v
    else
        return Vector{T}(vcat(v[1:first(i)-1], replacement, v[last(i)+1:end]))
    end
end
function splice!!(ncmul::NCMul, i)
    splice!(ncmul.factors, i)
    return ncmul
end
function splice!!(ncmul::NCMul, i, term::NCMul)
    ncmul.coeff *= term.coeff
    newfactors = mysplice!!(ncmul.factors, i, term.factors)
    newfactors === ncmul.factors && return ncmul
    return NCMul(ncmul.coeff, newfactors)
end
function splice!!(ncmul::NCMul{C,S}, i, term::S) where {C,S}
    splice!!(ncmul, i, NCMul(1, (term,)))
end
function splice!!(ncmul::NCMul, i, coeff::Number)
    deleteat!(ncmul.factors, i)
    ncmul.coeff *= coeff
    return ncmul
end
function splice!!(ncmul::NCMul, i, swap::Swap)
    length(i) == 2 && i[2] == i[1] + 1 || throw(ArgumentError("Invalid index for swap"))
    a, b = ncmul.factors[i]
    splice!!(ncmul, i, NCMul(swap.λ, [b, a]))
end
function splice!!_and_add(ncmul::NCMul, i, terms::AddTerms)
    newterms = [splice!!(copy(ncmul), i, term) for term in Iterators.drop(terms.terms, 1)]
    ncmul = splice!!(ncmul, i, first(terms.terms))
    return ncmul, newterms
end
function splice!!_and_add(ncmul::T, i, effect) where T<:NCMul
    ncmul = splice!!(ncmul, i, effect)
    return ncmul, T[]
end

bubble_sort(a::Number, ordering; kwargs...) = a
function mul_effect end
