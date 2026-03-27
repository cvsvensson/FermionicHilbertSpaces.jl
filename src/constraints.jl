
abstract type AbstractConstraint end

"""
    NoSymmetry()

Constraint that leaves a Hilbert space unchanged.
"""
struct NoSymmetry <: AbstractConstraint end

struct ProductConstraint{C} <: AbstractConstraint
    constraints::C
end

Base.:*(sym1::AbstractConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1, sym2))
Base.:*(sym1::AbstractConstraint, sym2::ProductConstraint) = ProductConstraint((sym1, sym2.constraints...))
Base.:*(sym1::ProductConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1.constraints..., sym2))
Base.:*(sym1::ProductConstraint, sym2::ProductConstraint) = ProductConstraint((sym1.constraints..., sym2.constraints...))
branch_constraint(constraint::ProductConstraint, space) = ProductConstraint(map(cons -> branch_constraint(cons, space), constraint.constraints))


"""
    AdditiveConstraint(allowed_values, subspaces=missing, functions)

Constraint enforcing that the sum of user-specified per-subspace contributions lies
in `allowed_values`.

The constraint is evaluated on the factors used during state generation. For a
composite Hilbert space this means `atomic_factors(space)`.
"""
struct AdditiveConstraint{T,H,F} <: AbstractConstraint
    allowed_values::T
    subspaces::H
    functions::F
end
AdditiveConstraint(allowed_values, functions) = AdditiveConstraint(allowed_values, missing, functions)
AdditiveConstraint(allowed_values, subspace::AbstractHilbertSpace, functions) = AdditiveConstraint(allowed_values, atomic_factors(subspace), functions)
AdditiveConstraint(allowed_values, subspace::AbstractClusterHilbertSpace, functions) = AdditiveConstraint(allowed_values, atomic_factors(subspace), functions)

"""
    NumberConservation(total=missing, subspaces=missing, weights=missing)

Constraint enforcing conservation of a (possibly weighted) particle number.
`total` can be a single value or collection of allowed values.
"""
struct NumberConservation{T,H,W} <: AbstractConstraint
    total::T
    subspaces::H
    weights::W
    function NumberConservation(_total=missing, _subspaces=missing, _weights=missing)
        total = _normalize_constraint_values(_total)
        subspace = _normalize_constraint_subspace(_subspaces)
        weights = _normalize_constraint_values(_weights)
        new{typeof(total),typeof(subspace),typeof(weights)}(total, subspace, weights)
    end
end
_normalize_constraint_subspace(subspace::AbstractHilbertSpace) = (subspace,)
_normalize_constraint_subspace(subspaces) = map(hilbert_space, subspaces)
_normalize_constraint_subspace(subspaces::Missing) = subspaces
NumberConservation(H::AbstractHilbertSpace) = NumberConservation(missing, (H,), missing)
# NumberConservation(n) = NumberConservation(n, missing, missing)
# NumberConservation(H::AbstractHilbertSpace) = NumberConservation(missing, atomic_factors(H), missing)
# NumberConservation() = NumberConservation(missing, missing, missing)
# NumberConservation(total, subspace::AbstractHilbertSpace) = NumberConservation(total, (subspace,), missing)
# NumberConservation(total, subspace::AbstractHilbertSpace) = NumberConservation(total, (subspace,), missing)
# NumberConservation(total, subspace::AbstractClusterHilbertSpace) = NumberConservation(total, (subspace,), missing)
# NumberConservation(total, spaces) = NumberConservation(total, spaces, missing)

"""
    ParityConservation(parities=[-1, 1], subspaces=missing)

Constraint enforcing allowed fermion parities, optionally on selected subspaces.
"""
struct ParityConservation{H} <: AbstractConstraint
    allowed_parities::Vector{Int}
    subspaces::H
    function ParityConservation(_allowed=[-1, 1], _subspaces=missing)
        allowed = _normalize_constraint_values(_allowed)
        allowed in Set([[-1, 1], [1], [-1]]) || throw(ArgumentError("Allowed parities must be a subset of [-1, 1]"))
        subspace = _normalize_constraint_subspace(_subspaces)
        new{typeof(subspace)}(allowed, subspace)
    end
end
# ParityConservation() = ParityConservation([-1, 1], missing)
# ParityConservation(H::AbstractHilbertSpace) = ParityConservation([-1, 1], H)
# ParityConservation(ps::AbstractVector{Int}) = ParityConservation(Vector{Int}(ps), missing)
# ParityConservation(p::Int) = ParityConservation([p], missing)

_format_subspaces(s) = ismissing(s) ? nothing : isa(s, Union{Tuple,AbstractVector}) ? "$(length(s)) subspaces" : "subspaces"
_indent(s, p) = join([(i > 1 ? p : "") * l for (i, l) in enumerate(split(s, '\n'))], '\n')
_parity(p) = p == [-1, 1] ? "any" : p == [1] ? "even" : p == [-1] ? "odd" : string(p)

function Base.show(io::IO, nc::NumberConservation{T,H,W}) where {T,H,W}
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(nc.total) ? nothing : sprint(show, nc.total),
            ismissing(nc.subspaces) ? nothing : _format_subspaces(nc.subspaces),
            ismissing(nc.weights) ? nothing : "weighted"])
        print(io, "NumberConservation(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(nc.total) ? nothing : "total: " * (isa(nc.total, AbstractVector) && length(nc.total) == 1 ? string(only(nc.total)) : sprint(show, nc.total)),
            ismissing(nc.subspaces) ? nothing : "$(_format_subspaces(nc.subspaces))",
            ismissing(nc.weights) ? nothing : "weights: " * sprint(show, nc.weights)])
        isempty(lines) ? print(io, "NumberConservation()") : print(io, "NumberConservation(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, pc::ParityConservation{H}) where {H}
    compact = get(io, :compact, false)
    if compact
        parts = [_parity(pc.allowed_parities)]
        !ismissing(pc.subspaces) && push!(parts, "subspaces")
        print(io, "ParityConservation(", join(parts, ", "), ")")
    else
        lines = ["allowed_parities: $(_parity(pc.allowed_parities))"]
        !ismissing(pc.subspaces) && push!(lines, ", $(_format_subspaces(pc.subspaces))")
        print(io, "ParityConservation(", join(lines, ""), ")")
    end
end

function Base.show(io::IO, ac::AdditiveConstraint{T,H,F}) where {T,H,F}
    nf = isa(ac.functions, Tuple) ? length(ac.functions) : 1
    if get(io, :compact, false)
        pre = ismissing(ac.allowed_values) ? "" : sprint(show, ac.allowed_values) * ", "
        print(io, "AdditiveConstraint(", pre, "$nf function(s))")
    else
        lines = ["  allowed_values: " * (ismissing(ac.allowed_values) ? "missing" : sprint(show, ac.allowed_values)),
            "  functions: $nf function(s)"]
        !ismissing(ac.subspaces) && push!(lines, "  subspaces: $(_format_subspaces(ac.subspaces))")
        print(io, "AdditiveConstraint(\n", join(lines, "\n"), "\n)")
    end
end

function Base.show(io::IO, pc::ProductConstraint{C}) where {C}
    nconstraints = length(pc.constraints)
    nshow = min(nconstraints, 6)
    if get(io, :compact, false)
        print(io, "ProductConstraint(")
        for (i, c) in enumerate(pc.constraints[1:nshow])
            i > 1 && print(io, " * ")
            show(IOContext(io, :compact => true), c)
        end
        nconstraints > nshow && print(io, " * ... ($(nconstraints - nshow) more)")
        print(io, ")")
    else
        print(io, "ProductConstraint:\n")
        for (i, c) in enumerate(pc.constraints[1:nshow])
            is_last = (i == nshow) && (nconstraints == nshow)
            prefix = is_last ? "  └─ " : "  ├─ "
            if isa(c, ProductConstraint)
                nested = join(split(sprint(show, c), '\n')[2:end], '\n')
                print(io, prefix, "ProductConstraint:\n", _indent(nested, is_last ? "     " : "  │  "))
            else
                print(io, prefix, sprint(show, c; context=:compact => true))
            end
            is_last || print(io, "\n")
        end
        if nconstraints > nshow
            nshow > 0 && print(io, "\n")
            print(io, "  └─ ... (", nconstraints - nshow, " more)")
        end
    end
end

unique_split_state(state, mapper) = only(first(split_state(state, mapper)))

# function sector_function(cons::NumberConservation{T,S,Missing}, space::AbstractHilbertSpace, spaces) where {T,S}
#     subspaces = S === Missing ? spaces : cons.subspaces
#     mapper = state_mapper(space, subspaces)
#     function number(state)
#         subs = unique_split_state(state, mapper)
#         sum(particle_number, subs)
#     end
# end
# function sector_function(cons::NumberConservation{T,S,W}, space::AbstractHilbertSpace, spaces) where {T,S,W}
#     subspaces = S === Missing ? spaces : cons.subspaces
#     mapper = state_mapper(space, subspaces)
#     function number(state)
#         subs = unique_split_state(state, mapper)
#         # mapreduce((s, w) -> particle_number(s) * w, +, subs, cons.weights)
#         _additive_function_application(subs, WeightedFunction(particle_number, cons.weights))
#     end
# end
function sector_function(cons::C, space::AbstractHilbertSpace, spaces) where {C<:Union{<:NumberConservation,<:ParityConservation,<:AdditiveConstraint}}
    subspaces = ismissing(cons.subspaces) ? spaces : cons.subspaces
    subspaces = ismissing(subspaces) ? (space,) : subspaces
    mapper = state_mapper(space, subspaces)
    func = _function_on_substates(cons)
    function number(state)
        subs = unique_split_state(state, mapper)
        _additive_function_application(subs, func)
    end
end
function sector_function(constraint::ProductConstraint, space::AbstractHilbertSpace, spaces)
    subspace_functions = map(cons -> sector_function(cons, space, spaces), constraint.constraints)
    state -> map(f -> f(state), subspace_functions)
end

_function_on_substates(::NumberConservation{<:Any,<:Any,Missing}) = particle_number
_function_on_substates(cons::NumberConservation{<:Any,<:Any,W}) where {W} = WeightedFunction(particle_number, cons.weights)
_function_on_substates(::ParityConservation) = parity
_function_on_substates(cons::AdditiveConstraint) = cons.functions


function branch_constraint(constraint::ParityConservation, spaces)
    possible_numbers = ismissing(constraint.subspaces) ? (0:sum(maximum_particles, spaces)) : (0:sum(nbr_of_modes, constraint.subspaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    additive_branch_constraint(allowed_numbers, particle_number, constraint.subspaces, spaces)
end

function branch_constraint(constraint::NumberConservation{T,H,W}, spaces) where {T,H,W}
    subspaces = H === Missing ? spaces : constraint.subspaces
    if W === Missing
        total = T === Missing ? (0:sum(maximum_particles, subspaces)) : constraint.total
        return additive_branch_constraint(total, particle_number, subspaces, spaces)
    end
    T === Missing && throw(ArgumentError("Total particle number must be specified when using weighted number branch constraint"))
    additive_branch_constraint(constraint.total, WeightedFunction(particle_number, constraint.weights), subspaces, spaces)
end


# sector_function(::ParityConservation{Missing}, space::ProductSpace) = f -> prod(parity, f.states)
# sector_function(::NumberConservation{T,Missing,Missing}, space) where {T} = f -> particle_number(f)
# sector_function(::ParityConservation{Missing}, space) = f -> parity(f)

# function sector_function(constraint::ProductConstraint, space)
#     subspace_functions = map(cons -> sector_function(cons, space), constraint.constraints)
#     state -> map(f -> f(state), subspace_functions)
# end
sectors(::AbstractConstraint) = nothing
has_sectors(N::NumberConservation) = true
has_sectors(C::AdditiveConstraint) = true
has_sectors(P::ParityConservation) = true
has_sectors(c::ProductConstraint) = any(has_sectors, c.constraints)


"""
    constrain_space(space, constraint; kwargs...)
    constrain_space(space, states)

Build a constrained Hilbert space from `space`, either by applying a constraint or
by explicitly providing a list of allowed basis states.
"""
constrain_space
default_sorter(space, constraint) = identity

@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = NumberConservation() * ParityConservation()
    @fermions f
    H = hilbert_space(f, labels, qn)
    @test collect(quantumnumbers(H)) == [(n, (-1)^n) for n in 0:4]
    qn = prod(NumberConservation(missing, hilbert_space(f[l])) for l in labels)
    H = hilbert_space(f, labels, qn)
    @test dim(H) == 2^4
    @test all(isone ∘ dim, sectors(H))
end
