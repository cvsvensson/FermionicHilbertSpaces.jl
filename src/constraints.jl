
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
    NumberConservation(total=missing, subspaces=missing, weights=missing)

Constraint enforcing conservation of a (possibly weighted) particle number.
`total` can be a single value or collection of allowed values.
"""
struct NumberConservation{T,H,W} <: AbstractConstraint
    total::T
    subspaces::H
    weights::W
end
NumberConservation(n) = NumberConservation(n, missing, missing)
NumberConservation(H::AbstractHilbertSpace) = NumberConservation(missing, H, missing)
NumberConservation() = NumberConservation(missing, missing, missing)
NumberConservation(total, subspace::AbstractHilbertSpace) = NumberConservation(total, (subspace,), missing)
NumberConservation(total, subspace::AbstractClusterHilbertSpace) = NumberConservation(total, atomic_factors(subspace), missing)
NumberConservation(total, spaces) = NumberConservation(total, spaces, missing)

"""
    ParityConservation(parities=[-1, 1], subspaces=missing)

Constraint enforcing allowed fermion parities, optionally on selected subspaces.
"""
struct ParityConservation{H} <: AbstractConstraint
    allowed_parities::Vector{Int}
    subspaces::H
end
ParityConservation() = ParityConservation([-1, 1], missing)
ParityConservation(H::AbstractHilbertSpace) = ParityConservation([-1, 1], H)
ParityConservation(ps::AbstractVector{Int}) = ParityConservation(Vector{Int}(ps), missing)
ParityConservation(p::Int) = ParityConservation([p], missing)

function sector_function(cons::NumberConservation{T,S,W}, space) where {T,S,W}
    subspaces = S === Missing ? factors(space) : cons.subspaces
    issub = BitVector(map(in(subspaces), factors(space)))
    _weights = W === Missing ? ones(Int, sum(issub)) : cons.weights
    weights = zeros(eltype(_weights), length(factors(space)))
    weights[issub] .= _weights
    return f -> mapreduce((n, w) -> particle_number(substate(n, f)) * w, +, 1:length(weights), weights)
    # return f -> mapreduce((f, w) -> particle_number(f) * w, +, f.states, weights)
end

sector_function(::ParityConservation{Missing}, space::ProductSpace) = f -> prod(parity, f.states)
sector_function(::NumberConservation{T,Missing,Missing}, space) where {T} = f -> particle_number(f)
sector_function(::ParityConservation{Missing}, space) = f -> parity(f)

function sector_function(constraint::ProductConstraint, space)
    subspace_functions = map(cons -> sector_function(cons, space), constraint.constraints)
    state -> map(f -> f(state), subspace_functions)
end
sectors(::AbstractConstraint) = nothing
has_sectors(N::NumberConservation) = true
has_sectors(P::ParityConservation) = true
has_sectors(c::ProductConstraint) = any(has_sectors, c.constraints)

function constrain_space(space, constraint::AbstractConstraint; kwargs...)
    sortby = default_sorter(space, constraint)
    states = generate_states(space, constraint; kwargs...)
    isnothing(sortby) || sort!(states, by=sortby)
    has_sectors(constraint) || return ConstrainedSpace(space, states)
    block_space(space, states, sector_function(constraint, space))
end

"""
    constrain_space(space, constraint; kwargs...)
    constrain_space(space, states)

Build a constrained Hilbert space from `space`, either by applying a constraint or
by explicitly providing a list of allowed basis states.
"""
constrain_space
default_processor(space, _) = nothing
default_sorter(space, constraint) = nothing


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
