
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

unique_split_state(state, mapper) = only(first(split_state(state, mapper)))

function sector_function(cons::C, space::AbstractHilbertSpace, spaces) where {C<:Union{<:NumberConservation,<:ParityConservation,<:AdditiveConstraint}}
    subspaces = ismissing(cons.subspaces) ? spaces : cons.subspaces
    subspaces = ismissing(subspaces) ? (space,) : subspaces
    mapper = state_mapper(space, subspaces)
    function number(state)
        subs = unique_split_state(state, mapper)
        _apply_constraint_function(subs, cons)
    end
end
function sector_function(constraint::ProductConstraint, space::AbstractHilbertSpace, spaces)
    subspace_functions = map(cons -> sector_function(cons, space, spaces), constraint.constraints)
    state -> map(f -> f(state), subspace_functions)
end

_apply_constraint_function(substates, ::NumberConservation{<:Any,<:Any,Missing}) = sum(particle_number, substates)
_apply_constraint_function(substates, cons::NumberConservation{<:Any,<:Any,W}) where {W} = mapreduce((s, w) -> particle_number(s) * w, +, substates, cons.weights)
_apply_constraint_function(substates, ::ParityConservation) = prod(parity, substates)
_apply_constraint_function(substates, cons::AdditiveConstraint{<:Any,<:Any,<:Function}) = sum(cons.f, substates)
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
