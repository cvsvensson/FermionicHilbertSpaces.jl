
abstract type AbstractConstraint end

"""
    NoSymmetry()

Constraint that leaves a Hilbert space unchanged.
"""
struct NoSymmetry <: AbstractConstraint end

struct ProductConstraint{C} <: AbstractConstraint
    constraints::C
end

Base.:*(sym1::AbstractConstraint, sym2::AbstractConstraint) = ProductConstraint([sym1, sym2])
Base.:*(sym1::AbstractConstraint, sym2::ProductConstraint) = ProductConstraint([sym1, sym2.constraints...])
Base.:*(sym1::ProductConstraint, sym2::AbstractConstraint) = ProductConstraint([sym1.constraints..., sym2])
Base.:*(sym1::ProductConstraint, sym2::ProductConstraint) = ProductConstraint([sym1.constraints..., sym2.constraints...])
function branch_constraint(constraint::ProductConstraint, space)
    prunable = filter(supports_branch_pruning, constraint.constraints)
    ProductConstraint(map(cons -> branch_constraint(cons, space), prunable))
end
supports_branch_pruning(c::ProductConstraint) = any(supports_branch_pruning, c.constraints)
supports_filtering(c::ProductConstraint) = any(supports_filtering, c.constraints)
supports_sector_grouping(c::ProductConstraint) = all(supports_sector_grouping, c.constraints)


struct FilterConstraint{H,FS,F} <: AbstractConstraint
    subspaces::H
    subspace_functions::FS
    reducer::F
    function FilterConstraint(subspaces::H, subspace_functions::FS, reducer::F) where {H,FS,F}
        new{H,FS,F}(subspaces, subspace_functions, reducer)
    end
end
function FilterConstraint(reducer::F) where {F<:Function}
    FilterConstraint(missing, missing, reducer)
end
supports_branch_pruning(::FilterConstraint) = false
supports_filtering(::FilterConstraint) = true
supports_sector_grouping(c::FilterConstraint) = false
function filter_function(constraint::FilterConstraint{Missing,Missing,F}, ::AbstractHilbertSpace) where F
    return constraint.reducer
end
function filter_function(constraint::FilterConstraint, space::AbstractHilbertSpace)
    mapper = state_mapper(space, constraint.subspaces)
    function _filter_function(state)
        subs = unique_split_state(state, mapper)
        values = Iterators.map((s, f) -> f(s), subs, constraint.subspace_functions)
        constraint.reducer(values)
    end
end
function filter_function(constraint::FilterConstraint{<:Any,<:Function}, space::AbstractHilbertSpace)
    mapper = state_mapper(space, constraint.subspaces)
    f = constraint.subspace_functions
    function _filter_function(state)
        subs = unique_split_state(state, mapper)
        values = Iterators.map(f, subs)
        constraint.reducer(values)
    end
end

struct BlockConstraint{F<:FilterConstraint} <: AbstractConstraint
    filter::F
end
function BlockConstraint(subspaces::H, subspace_functions::FS, reducer::F) where {H,FS,F}
    BlockConstraint(FilterConstraint(subspaces, subspace_functions, reducer))
end
function BlockConstraint(reducer::F) where {F<:Function}
    BlockConstraint(FilterConstraint(missing, missing, reducer))
end
supports_branch_pruning(::BlockConstraint) = false
supports_filtering(::BlockConstraint) = true
supports_sector_grouping(::BlockConstraint) = true
function sector_function(constraint::BlockConstraint, space::AbstractHilbertSpace)
    return filter_function(constraint.filter, space)
end
function filter_function(constraint::BlockConstraint, space::AbstractHilbertSpace)
    sec = filter_function(constraint.filter, space)
    return !ismissing ∘ sec
end


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
supports_branch_pruning(::AdditiveConstraint) = true
supports_filtering(::AdditiveConstraint{<:Any,Missing}) = false
supports_filtering(::AdditiveConstraint) = true
supports_sector_grouping(::AdditiveConstraint{<:Any,Missing}) = false
supports_sector_grouping(::AdditiveConstraint) = true

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
_normalize_constraint_subspace(subspaces) = subspaces
_normalize_constraint_subspace(subspaces::Missing) = subspaces
NumberConservation(H::AbstractHilbertSpace) = NumberConservation(missing, (H,), missing)
supports_branch_pruning(::NumberConservation) = true
supports_filtering(::NumberConservation) = true
supports_sector_grouping(::NumberConservation) = true

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
supports_branch_pruning(::ParityConservation) = true
supports_filtering(::ParityConservation) = true
supports_sector_grouping(::ParityConservation) = true

unique_split_state(state, mapper) = only(first(split_state(state, mapper)))

function sector_function(cons::C, space::AbstractHilbertSpace) where {C<:Union{<:NumberConservation,<:ParityConservation,<:AdditiveConstraint}}
    subspaces = ismissing(cons.subspaces) ? (space,) : cons.subspaces
    mapper = state_mapper(space, subspaces)
    allowed_vals = allowed_values(cons, space)
    allowed = in(allowed_vals)
    function number(state)
        subs = unique_split_state(state, mapper)
        val = _apply_constraint_function(subs, cons)
        allowed(val) ? val : missing
    end
end
function sector_function(constraint::ProductConstraint, space::AbstractHilbertSpace)
    subspace_functions = map(cons -> sector_function(cons, space), constraint.constraints)
    function sector(state)
        sectors = map(f -> f(state), subspace_functions)
        any(ismissing, sectors) && return missing
        return sectors
    end
end

_apply_constraint_function(substates, ::NumberConservation{<:Any,<:Any,Missing}) = sum(particle_number, substates)
_apply_constraint_function(substates, cons::NumberConservation{<:Any,<:Any,W}) where {W} = mapreduce((s, w) -> particle_number(s) * w, +, substates, cons.weights)
_apply_constraint_function(substates, ::ParityConservation) = prod(parity, substates)
_apply_constraint_function(substates, cons::AdditiveConstraint{<:Any,<:Any,<:Function}) = sum(cons.functions, substates)
function _apply_constraint_function(substates, cons::AdditiveConstraint)
    mapreduce((s, f) -> f(s), +, substates, cons.functions)
end


function branch_constraint(constraint::ParityConservation, spaces)
    possible_numbers = ismissing(constraint.subspaces) ? (0:sum(maximum_particles, spaces)) : (0:sum(nbr_of_modes, constraint.subspaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    cons = NumberConservation(allowed_numbers, constraint.subspaces, missing)
    branch_constraint(cons, spaces)
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

sectors(::AbstractConstraint) = nothing


"""
    constrain_space(space, constraint; kwargs...)
    constrain_space(space, states)

Build a constrained Hilbert space from `space`, either by applying a constraint or
by explicitly providing a list of allowed basis states.
"""
constrain_space

@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = NumberConservation() * ParityConservation()
    @fermions f
    H = hilbert_space(f, labels, qn)
    @test collect(quantumnumbers(H)) == [[n, (-1)^n] for n in 0:4]
    qn = prod(NumberConservation(missing, hilbert_space(f[l])) for l in labels)
    H = hilbert_space(f, labels, qn)
    @test dim(H) == 2^4
    @test all(isone ∘ dim, sectors(H))
end
