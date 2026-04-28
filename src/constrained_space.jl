
struct ConstrainedSpace{B,H,S} <: AbstractHilbertSpace{B}
    parent::H
    states::S
    state_index::Dict{B,Int}
end
_find_position(Hsub::AbstractHilbertSpace, H::ConstrainedSpace) = _find_position(Hsub, H.parent)
Base.:(==)(H1::ConstrainedSpace, H2::ConstrainedSpace) = H1.parent == H2.parent && H1.states == H2.states && H1.state_index == H2.state_index
Base.hash(H::ConstrainedSpace, h::UInt) = hash(H.parent, hash(H.states, hash(H.state_index, h)))
Base.parent(H::ConstrainedSpace) = H.parent
function ConstrainedSpace(space::ConstrainedSpace, states)
    intersect(states, space.states)
    ConstrainedSpace(parent(space), states)
end
function ConstrainedSpace(space::H, states::S) where {H,S}
    B = eltype(states)
    state_index = Dict{B,Int}(s => i for (i, s) in enumerate(states))
    ConstrainedSpace{B,H,S}(space, states, state_index)
end

dim(H::ConstrainedSpace) = length(H.states)
basisstates(H::ConstrainedSpace) = H.states
basisstate(n::Int, H::ConstrainedSpace) = H.states[n]
state_index(state, H::ConstrainedSpace) = get(H.state_index, state, missing)
atomic_factors(H::ConstrainedSpace) = atomic_factors(parent(H))
groups(H::ConstrainedSpace) = groups(parent(H))
factors(H::ConstrainedSpace) = factors(parent(H))
isconstrained(H::ConstrainedSpace) = true
combine_states(substates, sp::ConstrainedSpace) = combine_states(substates, parent(sp))
partial_trace_phase_factor(s1, s2, sp::ConstrainedSpace) = partial_trace_phase_factor(s1, s2, parent(sp))
atomic_substate(n, f, space::ConstrainedSpace) = atomic_substate(n, f, parent(space))
constrain_space(space::AbstractHilbertSpace, ::NoSymmetry) = space
constrain_space(space::AbstractHilbertSpace, states::AbstractVector{B}, constraint::AbstractConstraint=NoSymmetry()) where B<:AbstractBasisState = constrain_space(space, constraint, states)

function constrain_space(space, constraint::AbstractConstraint, states = basisstates(space))
    if supports_sector_grouping(constraint)
        f = sector_function(constraint, space)
        return sector_space(space, states, f, constraint)
    elseif supports_filtering(constraint)
        f = filter_function(constraint, space)
        filtered_states = collect(Iterators.filter(f, states))
        return ConstrainedSpace(space, filtered_states)
    else
        throw(ArgumentError("Constraint $(constraint) is not supported for constraining spaces."))
    end
end
_find_position(op::AbstractSym, H::ConstrainedSpace) = _find_position(op, parent(H))
add_tag(H::ConstrainedSpace, tag) = ConstrainedSpace(add_tag(parent(H), tag), H.states, H.state_index)

allowed_values(::NumberConservation{Missing}, space) = 0:maximum_particles(space)
allowed_values(constraint::NumberConservation{T}, space) where T = constraint.total
allowed_values(p::ParityConservation, space) = p.allowed_parities
allowed_values(c::AdditiveConstraint, space) = c.allowed_values

@testitem "constrain_space" begin
    @fermions f
    H = hilbert_space(f, 1:2)
    @test dim(constrain_space(H, NumberConservation())) == 4
    @test dim(constrain_space(H, NumberConservation(1))) == 2
    @test dim(constrain_space(H, NumberConservation(0:1))) == 3

    @test dim(constrain_space(H, ParityConservation())) == 4
    @test dim(constrain_space(H, ParityConservation(1))) == 2

    H1 = constrain_space(H, NumberConservation())
    H2 = constrain_space(H1, NumberConservation(0:1))
    H3 = constrain_space(H2, NumberConservation(1:2))
    @test basisstates(H3) == basisstates(constrain_space(H, NumberConservation(1)))

    #repeat for bosons
    @bosons b
    H = hilbert_space(b, 1:2, 2)
    @test dim(constrain_space(H, NumberConservation())) == 4
    @test dim(constrain_space(H, NumberConservation(1))) == 2
    @test dim(constrain_space(H, NumberConservation(0:1))) == 3

    @test dim(constrain_space(H, ParityConservation())) == 4
    @test dim(constrain_space(H, ParityConservation(1))) == 2

    H1 = constrain_space(H, NumberConservation())
    H2 = constrain_space(H1, NumberConservation(0:1))
    H3 = constrain_space(H2, NumberConservation(1:2))
    @test basisstates(H3) == basisstates(constrain_space(H, NumberConservation(1)))
end

@testitem "FilterConstraint agrees with NumberConservation" begin
    using FermionicHilbertSpaces: SectorConstraint, FilterConstraint, particle_number
    @fermions f
    N = 4
    H = hilbert_space(f, 1:N)
    Hs = (hilbert_space(f, 1:2), hilbert_space(f, 3:4))

    sector_constraint = SectorConstraint(state -> begin
        n = particle_number(state)
        n in 1:2 ? n : missing
    end)
    filter_constraint = FilterConstraint(in(1:2) ∘ particle_number)

    Hnumber = hilbert_space(f, 1:N, NumberConservation(1:2))
    Hfrom_tensor = tensor_product(Hs, constraint=filter_constraint)
    Hfrom_constrain = constrain_space(H, filter_constraint)
    Hfrom_tensor_sector = tensor_product(Hs, constraint=sector_constraint)
    Hfrom_constrain_sector = constrain_space(H, sector_constraint)
    states = Set(basisstates(Hnumber))
    @test Set(basisstates(Hfrom_tensor)) == states
    @test Set(basisstates(Hfrom_constrain)) == states
    @test Set(basisstates(Hfrom_tensor_sector)) == states
    @test Set(basisstates(Hfrom_constrain_sector)) == states

    @test collect(quantumnumbers(Hfrom_constrain_sector)) == collect(quantumnumbers(Hnumber))
    @test collect(quantumnumbers(Hfrom_constrain_sector)) == collect(quantumnumbers(Hnumber))

    # Now with subregions and weights
    Hnumber = hilbert_space(f, 1:N, NumberConservation(1:2, [f[1], f[2]], [1, -1]))
    filter_constraint = FilterConstraint([f[1], f[2]], [particle_number, particle_number], in(-(1:2)) ∘ only ∘ diff ∘ collect)
    sector_constraint = SectorConstraint([f[1], f[2]], [particle_number, particle_number], ns -> begin
        n1, n2 = collect(ns)
        n1 - n2 in 1:2 ? n1 - n2 : missing
    end)
    Hfrom_tensor = tensor_product(Hs, constraint=filter_constraint)
    Hfrom_constrain = constrain_space(H, filter_constraint)
    Hfrom_tensor_sector = tensor_product(Hs, constraint=sector_constraint)
    Hfrom_constrain_sector = constrain_space(H, sector_constraint)

    states = Set(basisstates(Hnumber))
    @test Set(basisstates(Hfrom_tensor)) == states
    @test Set(basisstates(Hfrom_constrain)) == states
    @test Set(basisstates(Hfrom_tensor_sector)) == states
    @test Set(basisstates(Hfrom_constrain_sector)) == states
end

state_mapper(H::ConstrainedSpace, Hs) = state_mapper(parent(H), Hs)
mode_ordering(H::ConstrainedSpace) = mode_ordering(parent(H))

_apply_local_operators(ops, state, space::ConstrainedSpace, precomp) = _apply_local_operators(ops, state, space.parent, precomp)
_precomputation_before_operator_application(ops, space::ConstrainedSpace) = _precomputation_before_operator_application(ops, parent(space))

@testitem "Constrained space" begin
    import FermionicHilbertSpaces: constrain_space, CombineFockNumbersProcessor, subregion
    N = 5
    @fermions f
    H = hilbert_space(f, 1:N)
    sym = sum(f[k]' * f[k] for k in 1:N)
    m = matrix_representation(sym, H)
    @test dim(H) == 2^N
    @test size(m) == (2^N, 2^N)

    #total number of particles is 1
    Hc = constrain_space(H, NumberConservation(1))
    Hc2 = tensor_product(H.modes; constraint=NumberConservation(1))
    @test Set(basisstates(Hc)) == Set(basisstates(Hc2))
    @test dim(Hc) == N
    mc = matrix_representation(sym, Hc)
    @test size(mc) == (N, N)

    @test m == numberoperator(H)

    H2 = hilbert_space(f, 1:N, NumberConservation(1))
    @test numberoperator(H2) == matrix_representation(sym, H2)

    Nsub = 3
    Hsub = subregion(H.modes[1:Nsub], H2)
    @test dim(Hsub) == Nsub + 1
end

_apply_local_operators(ops, state, space::SectorHilbertSpace, precomp) = _apply_local_operators(ops, state, space.parent, precomp)
add_tag(H::SectorHilbertSpace, tag) = SectorHilbertSpace(add_tag(parent(H), tag), H.ordered_basis_states, H.state_to_index, H.qn_to_states, H.constraint)

"""
    localize_constraint(c, H::SectorHilbertSpace)

Return a version of constraint `c` that is bound to `H` as its subspace, so that
when `sector_function` is later called on a larger combined space it knows to extract
the contribution from `H` only.  Already-localized (non-`Missing` subspace) or
unsupported constraints are returned unchanged.
"""
localize_constraint(c::AbstractConstraint, ::SectorHilbertSpace) = c
function localize_constraint(c::NumberConservation{<:Any,Missing,W}, H::SectorHilbertSpace) where W
    allowed = collect(keys(H.qn_to_states))
    NumberConservation(allowed, H, c.weights)
end
function localize_constraint(c::ParityConservation{Missing}, H::SectorHilbertSpace)
    ParityConservation(c.allowed_parities, H)
end
function localize_constraint(c::ProductConstraint, H::SectorHilbertSpace)
    qns = collect(keys(H.qn_to_states))
    localized = map(enumerate(c.constraints)) do (i, sub_c)
        allowed_i = unique!(map(qn -> qn[i], qns))
        _localize_sub_constraint(sub_c, allowed_i, H)
    end
    ProductConstraint(localized)
end
_localize_sub_constraint(c::NumberConservation{<:Any,<:Any,W}, allowed, H) where W = NumberConservation(allowed, H, c.weights)
_localize_sub_constraint(c::ParityConservation, allowed, H) = ParityConservation(allowed, H)
_localize_sub_constraint(c::AbstractConstraint, _allowed, _H) = c
function localize_constraint(c::SectorConstraint{<:FilterConstraint{Missing,Missing}}, H::SectorHilbertSpace)
    # Bind the free-standing reducer to H as its single subspace.
    # sector_function on the larger space will split the state, extract the H-substate,
    # apply the original reducer to it, and return the label via `only`.
    SectorConstraint((H,), (c.filter.reducer,), only)
end

"""
    _combined_sector_constraint(spaces)

Given a collection of Hilbert spaces, construct a combined constraint from the stored
constraints of any `SectorHilbertSpace` inputs (localized to their respective subspaces).
Returns `missing` if no usable sector information exists.
"""
function _combined_sector_constraint(spaces)
    sector_inputs = filter(s -> s isa SectorHilbertSpace && !isnothing(s.constraint), spaces)
    isempty(sector_inputs) && return missing
    local_constraints = map(H -> localize_constraint(H.constraint, H), sector_inputs)
    combined = length(local_constraints) == 1 ? only(local_constraints) : reduce(*, local_constraints)
    (supports_sector_grouping(combined) && supports_filtering(combined)) ? combined : missing
end