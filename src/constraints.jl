
abstract type AbstractConstraint end
struct NoSymmetry <: AbstractConstraint end

struct ProductConstraint{C} <: AbstractConstraint
    constraints::C
end

Base.:*(sym1::AbstractConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1, sym2))
Base.:*(sym1::AbstractConstraint, sym2::ProductConstraint) = ProductConstraint((sym1, sym2.constraints...))
Base.:*(sym1::ProductConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1.constraints..., sym2))
Base.:*(sym1::ProductConstraint, sym2::ProductConstraint) = ProductConstraint((sym1.constraints..., sym2.constraints...))
branch_constraint(constraint::ProductConstraint, space) = ProductConstraint(map(cons -> branch_constraint(cons, space), constraint.constraints))


struct NumberConservation{T,H} <: AbstractConstraint
    total::T
    subspaces::H
end
NumberConservation(n) = NumberConservation(n, nothing)
NumberConservation(H::AbstractHilbertSpace) = NumberConservation(nothing, H)
NumberConservation() = NumberConservation(nothing, nothing)
NumberConservation(total, subspace::AbstractHilbertSpace) = NumberConservation(total, (subspace,))
NumberConservation(total, subspace::AbstractClusterHilbertSpace) = NumberConservation(total, atomic_factors(subspace))

struct ParityConservation{H} <: AbstractConstraint
    allowed_parities::Vector{Int}
    subspaces::H
end
ParityConservation() = ParityConservation([-1, 1], nothing)
ParityConservation(H::AbstractHilbertSpace) = ParityConservation([-1, 1], H)
ParityConservation(ps::AbstractVector{Int}) = ParityConservation(Vector{Int}(ps), nothing)
ParityConservation(p::Int) = ParityConservation([p], nothing)

function sector_function(constraint::NumberConservation{T,Nothing}, space::ProductSpace) where {T}
    f -> sum(particle_number, f.states)
end
function sector_function(constraint::ParityConservation{Nothing}, space::ProductSpace)
    f -> prod(parity, f.states)
end
function sector_function(constraint::NumberConservation{T,Nothing}, space) where {T}
    f -> particle_number(f)
end
function sector_function(constraint::ParityConservation{Nothing}, space)
    f -> parity(f)
end
function sector_function(constraint::NumberConservation, space)
    positions = map(Base.Fix2(_find_position, space), constraint.subspaces)
    mask = focknbr_from_site_indices(positions)
    f -> particle_number(f & mask)
end
function sector_function(constraint::ParityConservation, space)
    positions = map(Base.Fix2(_find_position, space), constraint.subspaces)
    mask = focknbr_from_site_indices(positions)
    f -> parity(f & mask)
end
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
    # leaf_processor = default_processor(space, constraint)
    states = generate_states(space, constraint; kwargs...)
    isnothing(sortby) || sort!(states, by=sortby)
    has_sectors(constraint) || return ConstrainedSpace(space, states)
    block_space(space, states, sector_function(constraint, space))
end
default_processor(space, _) = nothing
default_sorter(space, constraint) = nothing


@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = NumberConservation() * ParityConservation()
    @fermions f
    H = hilbert_space(f, labels, qn)
    @test collect(quantumnumbers(H)) == [(n, (-1)^n) for n in 0:4]
    qn = prod(NumberConservation(nothing, hilbert_space(f[l])) for l in labels)
    H = hilbert_space(f, labels, qn)
    @test dim(H) == 2^4
    @test all(isone ∘ dim, sectors(H))
end
