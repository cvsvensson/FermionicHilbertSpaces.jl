
"""
    NoSymmetry()

Constraint that leaves a Hilbert space unchanged.
"""
struct NoSymmetry <: AbstractConstraint end
supports_sector_grouping(::NoSymmetry) = false
supports_filtering(::NoSymmetry) = false
supports_branch_pruning(::NoSymmetry) = false

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

supply_missing_constraint_info(constraint::ProductConstraint, space, spaces) = ProductConstraint(map(cons -> supply_missing_constraint_info(cons, space, spaces), constraint.constraints))

"""
    FilterConstraint(reducer)
    FilterConstraint(reducer, functions, spaces)

Constraint that keeps complete basis states for which `reducer` returns `true`.

With only `reducer`, the reducer is called directly as `reducer(state)`. When
`functions` and `spaces` are provided, the state is split into the selected
spaces, each function is applied to its corresponding subspace, and the
reducer is called on the resulting values. A single function is applied to all
spaces; a collection of functions supplies one function per subspace.
"""
struct FilterConstraint{R,FS,H} <: AbstractConstraint
    reducer::R
    maps::FS
    spaces::H
    function FilterConstraint(reducer::R, maps::FS, spaces::H) where {H,FS,R}
        new{R,FS,H}(reducer, maps, spaces)
    end
end
function FilterConstraint(reducer::R; maps=missing, spaces=missing) where {R<:Function}
    FilterConstraint(reducer, maps, spaces)
end
supports_branch_pruning(::FilterConstraint) = false
supports_filtering(::FilterConstraint) = true
supports_sector_grouping(c::FilterConstraint) = false
function filter_function(constraint::FilterConstraint{F,Missing,Missing}, ::AbstractHilbertSpace) where F
    return constraint.reducer
end
function filter_function(constraint::FilterConstraint, space::AbstractHilbertSpace)
    mapper = state_mapper(space, constraint.spaces)
    function _filter_function(state)
        subs = unique_split_state(state, mapper)
        values = Iterators.map((s, f) -> f(s), subs, constraint.maps)
        constraint.reducer(values)
    end
end
function filter_function(constraint::FilterConstraint{<:Any,<:Function}, space::AbstractHilbertSpace)
    mapper = state_mapper(space, constraint.spaces)
    f = constraint.maps
    function _filter_function(state)
        subs = unique_split_state(state, mapper)
        values = Iterators.map(f, subs)
        constraint.reducer(values)
    end
end
supply_missing_constraint_info(constraint::FilterConstraint{<:Any,<:Any,Missing}, space, spaces) = FilterConstraint(constraint.reducer, constraint.maps, spaces)
supply_missing_constraint_info(constraint::FilterConstraint{<:Any,Missing,Missing}, space, spaces) = FilterConstraint(constraint.reducer, constraint.maps, missing) # if subspace functions are missing, the reducer acts directly on the full state
supply_missing_constraint_info(constraint::FilterConstraint, space, spaces) = constraint


"""
    SectorConstraint(reducer, maps=missing, spaces=missing)

Constraint that groups complete basis states into sectors according to a
custom rule. It has the same `reducer`, `maps`, and `spaces` interface
as [`FilterConstraint`](@ref), but `reducer` must return the sector label for a
state. Returning `missing` discards the state; states with the same non-missing
label are placed in the same sector.

When `maps` and `spaces` are provided, the state is split into the
selected spaces, the maps are applied to them, and the reducer receives
the resulting values. A single function is applied to all spaces; a
collection of maps supplies one function per subspace.
"""
struct SectorConstraint{F<:FilterConstraint} <: AbstractConstraint
    filter::F
end
function SectorConstraint(reducer, maps, spaces)
    SectorConstraint(FilterConstraint(reducer, maps, spaces))
end
function SectorConstraint(reducer; maps=missing, spaces=missing)
    SectorConstraint(FilterConstraint(reducer, maps, spaces))
end

supports_branch_pruning(::SectorConstraint) = false
supports_filtering(::SectorConstraint) = true
supports_sector_grouping(::SectorConstraint) = true
function sector_function(constraint::SectorConstraint, space::AbstractHilbertSpace)
    return filter_function(constraint.filter, space)
end
function filter_function(constraint::SectorConstraint, space::AbstractHilbertSpace)
    sec = filter_function(constraint.filter, space)
    return !ismissing ∘ sec
end

supply_missing_constraint_info(constraint::SectorConstraint, space, spaces) = supply_missing_constraint_info(constraint.filter, space, spaces) |> SectorConstraint


"""
    AdditiveConstraint(allowed_values; spaces=missing, maps=missing)

Constraint enforcing that the sum of user-specified per-subspace contributions lies
in `allowed_values`.

The constraint is evaluated on the factors used during state generation. For a
composite Hilbert space this means `atomic_factors(space)`.
"""
struct AdditiveConstraint{T,H,F} <: AbstractConstraint
    allowed_values::T
    spaces::H
    maps::F
end
AdditiveConstraint(allowed_values; spaces=missing, maps=missing) = AdditiveConstraint(allowed_values, spaces, maps)
AdditiveConstraint(allowed_values, maps) = AdditiveConstraint(allowed_values; maps)
supports_branch_pruning(::AdditiveConstraint) = true
supports_filtering(::AdditiveConstraint{<:Any,Missing}) = false
supports_filtering(::AdditiveConstraint) = true
supports_sector_grouping(::AdditiveConstraint{<:Any,Missing}) = false
supports_sector_grouping(::AdditiveConstraint) = true

supply_missing_constraint_info(constraint::AdditiveConstraint{<:Any,Missing}, space, spaces) = AdditiveConstraint(constraint.allowed_values, spaces, constraint.maps)
supply_missing_constraint_info(constraint::AdditiveConstraint, space, spaces) = constraint

"""
    NumberConservation(allowed, spaces, weights)
    NumberConservation(allowed=missing; spaces=missing, weights=missing)

Constraint enforcing conservation of a (possibly weighted) particle number.
`total` can be a single value or collection of allowed values.
"""
struct NumberConservation{T,H,W} <: AbstractConstraint
    total::T
    spaces::H
    weights::W
    function NumberConservation(_total, _spaces, _weights)
        total = _normalize_constraint_values(_total)
        subspace = _normalize_constraint_subspace(_spaces)
        weights = _normalize_constraint_values(_weights)
        new{typeof(total),typeof(subspace),typeof(weights)}(total, subspace, weights)
    end
end
_normalize_constraint_subspace(subspace::AbstractHilbertSpace) = (subspace,)
_normalize_constraint_subspace(spaces) = spaces
_normalize_constraint_subspace(spaces::Missing) = spaces
supports_branch_pruning(::NumberConservation) = true
supports_filtering(::NumberConservation) = true
supports_sector_grouping(::NumberConservation) = true
supply_missing_constraint_info(constraint::NumberConservation{<:Any,Missing}, space, spaces) = NumberConservation(constraint.total, spaces, constraint.weights)
supply_missing_constraint_info(constraint::NumberConservation, space, spaces) = constraint
NumberConservation(allowed=missing; weights=missing, spaces=missing) = NumberConservation(allowed, spaces, weights)
NumberConservation(allowed, spaces; weights=missing) = NumberConservation(allowed; spaces, weights)
NumberConservation(space::AbstractHilbertSpace) = NumberConservation(missing; spaces=(space,), weights=missing)

"""
    ParityConservation(parities=[-1, 1], spaces=missing)

Constraint enforcing allowed fermion parities, optionally on selected spaces.
"""
struct ParityConservation{H} <: AbstractConstraint
    allowed_parities::Vector{Int}
    spaces::H
    function ParityConservation(_allowed, _spaces)
        allowed = _normalize_constraint_values(_allowed)
        allowed in Set([[-1, 1], [1], [-1]]) || throw(ArgumentError("Allowed parities must be a subset of [-1, 1]"))
        subspace = _normalize_constraint_subspace(_spaces)
        new{typeof(subspace)}(allowed, subspace)
    end
end
ParityConservation(parities=[-1, 1]; spaces=missing) = ParityConservation(parities, spaces)
ParityConservation(space::AbstractHilbertSpace) = ParityConservation(; spaces=(space,))
supports_branch_pruning(::ParityConservation) = true
supports_filtering(::ParityConservation) = true
supports_sector_grouping(::ParityConservation) = true
supply_missing_constraint_info(constraint::ParityConservation{Missing}, space, spaces) = ParityConservation(constraint.allowed_parities, spaces)
supply_missing_constraint_info(constraint::ParityConservation, space, spaces) = constraint
unique_split_state(state, mapper) = only(first(split_state(state, mapper)))

function sector_function(cons::C, space::AbstractHilbertSpace) where {C<:Union{<:NumberConservation,<:ParityConservation,<:AdditiveConstraint}}
    spaces = ismissing(cons.spaces) ? (space,) : cons.spaces
    mapper = state_mapper(space, spaces)
    allowed_vals = allowed_values(cons, space, mapper)
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
function filter_function(constraint::ProductConstraint, space::AbstractHilbertSpace)
    subspace_functions = map(constraint.constraints) do cons
        if supports_sector_grouping(cons)
            sector = sector_function(cons, space)
            return state -> !ismissing(sector(state))
        end
        filter_function(cons, space)
    end
    state -> all(f -> f(state), subspace_functions)
end

_apply_constraint_function(substates, ::NumberConservation{<:Any,<:Any,Missing}) = sum(particle_number, substates; init=0)
_apply_constraint_function(substates, cons::NumberConservation{<:Any,<:Any,W}) where {W} = mapreduce((s, w) -> particle_number(s) * w, +, substates, cons.weights; init=0)
_apply_constraint_function(substates, ::ParityConservation) = prod(parity, substates; init=1)
_apply_constraint_function(substates, cons::AdditiveConstraint{<:Any,<:Any,<:Function}) = sum(cons.maps, substates; init=0)
function _apply_constraint_function(substates, cons::AdditiveConstraint)
    mapreduce((s, f) -> f(s), +, substates, cons.maps)
end


function branch_constraint(constraint::ParityConservation, spaces)
    possible_numbers = ismissing(constraint.spaces) ? (0:sum(maximum_particles, spaces)) : (0:sum(maximum_particles, constraint.spaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    cons = NumberConservation(allowed_numbers, constraint.spaces, missing)
    branch_constraint(cons, spaces)
end

function branch_constraint(constraint::NumberConservation{T,H,W}, spaces) where {T,H,W}
    subspaces = H === Missing ? spaces : constraint.spaces
    if W === Missing
        return additive_branch_constraint(constraint.total, particle_number, subspaces, spaces)
    end
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

@testitem "ProductConstraint filtering" begin
    using FermionicHilbertSpaces: FilterConstraint, particle_number
    @fermions f
    H = hilbert_space(f, 1:2)
    constraint = NumberConservation(1) * FilterConstraint(state -> state != basisstate(2, H))

    Hconstrained = constrain_space(H, constraint)

    @test dim(Hconstrained) == 1
    @test all(particle_number(state) == 1 for state in basisstates(Hconstrained))
end
