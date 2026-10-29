struct ConstrainedSpace{B,H,S,I} <: AbstractSubspace{B,H}
    parent::H
    states::S
    index::I
    ConstrainedSpace{B,H,S,I}(parent, states, index) where {B,H,S,I} = new{B,H,S,I}(parent, states, index)
end

function ConstrainedSpace(space, states; constraint=NoSymmetry(), index=default_subspace_index)
    parent, members = unnest(space, states)
    stored = _stored_states(members)
    ConstrainedSpace(parent, stored, build_subspace_index(index, constraint, parent, stored))
end
ConstrainedSpace(space::H, states::S, index) where {H,S} = ConstrainedSpace{eltype(states),H,S,typeof(index)}(space, states, index)
ConstrainedSpace(::AbstractSubspace, states, index) = throw(nested_subspace_error())

Base.parent(H::ConstrainedSpace) = H.parent
subspace_states(H::ConstrainedSpace) = H.states
subspace_index(H::ConstrainedSpace) = H.index
rewrap(H::ConstrainedSpace, parent) = ConstrainedSpace(parent, H.states, H.index)

constrain_space(space::AbstractHilbertSpace, ::NoSymmetry; index=default_subspace_index) = space
constrain_space(space::AbstractHilbertSpace, states::Union{AbstractVector,BasisStates}; index=default_subspace_index) = constrain_space(space, NoSymmetry(), states; index)

constrain_space(space, ::NoSymmetry, states; index=default_subspace_index) = ConstrainedSpace(space, states; index)
function constrain_space(space, constraint::AbstractConstraint, states=basisstates(space); index=default_subspace_index)
    if supports_sector_grouping(constraint)
        f = sector_function(constraint, space)
        return sector_space(space, states, f, constraint; index)
    elseif supports_filtering(constraint)
        f = filter_function(constraint, space)
        filtered_states = collect(Iterators.filter(f, states))
        return ConstrainedSpace(space, filtered_states; constraint, index)
    else
        throw(ArgumentError("Cannot constrain a space with an unsupported constraint, got `$constraint`."))
    end
end

"""
    build_subspace_index(build, constraint, parent, states)

`build(constraint, parent, states)`, after checking that it gives the first, middle and last of
`states` their positions. Throws an `ArgumentError` otherwise.
"""
function build_subspace_index(build, constraint, parent, states)
    index = build(constraint, parent, states)
    n = length(states)
    for i in unique!([1, cld(n, 2), n])
        n == 0 && break
        state = states[firstindex(states)+i-1]
        position = state_position(index, state, parent)
        position == i || throw(ArgumentError("The subspace index $(typeof(index)) gives position $position to the state `$state` at position $i."))
    end
    return index
end

sector_constraint(c, qn) = c
sector_constraint(::Nothing, qn) = NoSymmetry()
sector_constraint(c::NumberConservation, qn) = NumberConservation([qn], c.spaces, c.weights)
sector_constraint(c::ParityConservation, qn) = ParityConservation([qn], c.spaces)
sector_constraint(c::ProductConstraint, qn) = ProductConstraint(map(sector_constraint, c.constraints, qn))

allowed_values(::NumberConservation{Missing,Missing,Missing}, space, mapper) = 0:maximum_particles(space)
function allowed_values(nc::NumberConservation{Missing}, space, mapper)
    Iterators.map(basisstates(space)) do state
        subs = unique_split_state(state, mapper)
        _apply_constraint_function(subs, nc)
    end |> unique
end
allowed_values(constraint::NumberConservation{T}, space, mapper) where T = constraint.total
allowed_values(p::ParityConservation, space, mapper) = p.allowed_parities
allowed_values(c::AdditiveConstraint, space, mapper) = c.allowed_values

"""
    has_arithmetic_state_index(H)

Whether `state_index(state, H)` is computed from `state` alone, without a lookup table, and is
the exact index in `1:dim(H)` of each basis state of `H` and `0` for other states of type `statetype(H)`.
"""
has_arithmetic_state_index(H) = false

const INDEX_TABLE_RATIO = 32 # maximum table entries per state
const INDEX_TABLE_MAX_LENGTH = 2^28

"""
    root_keyed_index(space, states)

An `IndexTable` of the positions of `states` when it is small enough, otherwise a
`RootIndexDict`, or `nothing` when `space` has no arithmetic `state_index`.
Throws an `ArgumentError` if a state appears more than once.
"""
function root_keyed_index(space, states)
    has_arithmetic_state_index(space) || return nothing
    B = statetype(space)
    (eltype(states) == B || all(state -> same_state_kind(state, B), states)) || return nothing
    d = dim(space)
    if d isa Int && d <= min(INDEX_TABLE_RATIO * length(states), INDEX_TABLE_MAX_LENGTH)
        table = fill_positions!(zeros(Int32, d), space, states)
        return isnothing(table) ? nothing : IndexTable(table)
    end
    positions = fill_positions!(sizehint!(Dict{typeof(d),Int}(), length(states)), space, states)
    return isnothing(positions) ? nothing : RootIndexDict(positions)
end
function fill_positions!(positions, space, states)
    for (i, state) in enumerate(states)
        k = state_index(state, space)
        iszero(k) && return nothing
        iszero(get(positions, k, 0)) || throw(_duplicate_state_error(state))
        positions[k] = i
    end
    return positions
end
# The position of each of `states`, which must appear once each.
function position_dict(states)
    d = Dict{eltype(states),Int}()
    for (i, state) in enumerate(states)
        get!(d, state, i) == i || throw(_duplicate_state_error(state))
    end
    return d
end
_duplicate_state_error(state) = ArgumentError("The state `$state` appears more than once.")
"""
    IndexTable(positions)

A subspace index whose `positions[state_index(state, parent)]` is the position of `state`, or
`0` for a state of `parent` outside the subspace.
"""
struct IndexTable
    positions::Vector{Int32}
end
function state_position(table::IndexTable, state, parent)
    k = state_index(state, parent)
    iszero(k) ? 0 : Int(table.positions[k])
end
"""
    RootIndexDict(positions)

A subspace index whose `positions[state_index(state, parent)]` is the position of `state`. It
is keyed by the integer index rather than by the state because integers hash and compare faster.
"""
struct RootIndexDict{K}
    positions::Dict{K,Int}
end
state_position(d::RootIndexDict, state, parent) = get(d.positions, state_index(state, parent), 0)
state_position(d::Dict, state, parent) = get(d, state, 0)
# Whether `state` has the type `B` up to type parameters, so that `state_index` of a space with states of type `B` accepts it.
same_state_kind(state, ::Type{B}) where {B} = _type_family(typeof(state)) === _type_family(B)
_type_family(T) = (parentmodule(T), nameof(T))
# Returns `state` if it is of the kind of the states of the subspace `H`, and throws a `MethodError` otherwise.
_check_state_kind(state, H) = (state isa statetype(H) || same_state_kind(state, statetype(H))) ? state : throw(MethodError(state_index, (state, H)))
same_state_kind(state::ProductState, ::Type{ProductState{S}}) where {S<:Tuple} = state.states isa Tuple && length(state.states) == fieldcount(S) && all(map(same_state_kind, state.states, fieldtypes(S)))

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

    # explicit states, alone or as candidates for a constraint
    @fermions f
    H = hilbert_space(f, 1:3)
    states = collect(basisstates(H))[2:6]
    Hs = constrain_space(H, states)
    @test basisstates(Hs) == states
    @test Hs == constrain_space(H, NoSymmetry(), states)
    Hc = constrain_space(H, NumberConservation(1), states)
    one_particle = Set(basisstates(constrain_space(H, NumberConservation(1))))
    @test basisstates(Hc) == filter(in(one_particle), states)
    @test_throws MethodError constrain_space(H, states, NumberConservation(1))
end

@testitem "Constraining a ConstrainedSpace keeps only its states" begin
    using FermionicHilbertSpaces: ConstrainedSpace
    @fermions f
    H = hilbert_space(f, 1:2)
    H1 = ConstrainedSpace(H, collect(basisstates(H))[2:3])
    kept = collect(basisstates(H1))
    outside = first(s for s in basisstates(H) if s ∉ kept)
    states = [outside, kept[2], kept[1]]
    H2 = ConstrainedSpace(H1, states)
    @test basisstates(H2) == [kept[2], kept[1]]
    @test dim(H2) == 2
    @test parent(H2) == parent(H1)
end

@testitem "FilterConstraint agrees with NumberConservation" begin
    using FermionicHilbertSpaces: particle_number
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
    filter_constraint = FilterConstraint(in(-(1:2)) ∘ only ∘ diff ∘ collect; maps=particle_number, spaces=[f[1], f[2]])
    sector_constraint = SectorConstraint(ns -> begin
        n1, n2 = collect(ns)
        n1 - n2 in 1:2 ? n1 - n2 : missing
    end; maps=[particle_number, particle_number],
        spaces=[f[1], f[2]])
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

@testitem "Constrained space" begin
    import FermionicHilbertSpaces: constrain_space, CombineFockNumbersProcessor, subregion
    N = 5
    @fermions f
    H = hilbert_space(f, 1:N)
    sym = sum(f[k]' * f[k] for k in 1:N)
    m = representation(sym, H)
    @test dim(H) == 2^N
    @test size(m) == (2^N, 2^N)

    #total number of particles is 1
    Hc = constrain_space(H, NumberConservation(1))
    Hc2 = tensor_product(H.modes; constraint=NumberConservation(1))
    @test Set(basisstates(Hc)) == Set(basisstates(Hc2))
    @test dim(Hc) == N
    mc = representation(sym, Hc)
    @test size(mc) == (N, N)

    @test m == numberoperator(H)

    H2 = hilbert_space(f, 1:N, NumberConservation(1))
    @test numberoperator(H2) == representation(sym, H2)

    Nsub = 3
    Hsub = subregion(H.modes[1:Nsub], H2)
    @test dim(Hsub) == Nsub + 1
end

@testitem "state_index of constrained and sector spaces" begin
    import FermionicHilbertSpaces: state_index, basisstate, statetype, add_tag, TransposedSpace, BosonicState
    @fermions f
    @boson b
    @spin s 1 // 2
    Hsector = tensor_product(hilbert_space(f, 1:3, NumberConservation(1)), hilbert_space(f, 4:6, NumberConservation(2)))
    Hprod = tensor_product(hilbert_space(f, 1:2), hilbert_space(b, 3), hilbert_space(s))
    Hconstrained = constrain_space(Hprod, collect(basisstates(Hprod))[1:3:end])
    Hprod_sector = tensor_product(hilbert_space(f, 1:3, NumberConservation(1)), hilbert_space(b, 4, NumberConservation(0:2)))
    T = statetype(hilbert_space(f, 1:70))
    Hlarge = constrain_space(hilbert_space(f, 1:70), [T(0), T(1), T(UInt128(1) << 69)])
    Hint = hilbert_space(f, 1:4, [FockNumber(6), FockNumber(1), FockNumber(2)])
    Hsparse = hilbert_space(f, 1:20, NumberConservation(1))
    for H in (Hsector, Hconstrained, Hprod_sector, Hlarge, Hint, Hsparse, TransposedSpace(Hsector), add_tag(Hsector, :x), add_tag(Hconstrained, :x))
        @test all(state_index(basisstate(i, H), H) == i for i in 1:dim(H))
    end
    @test state_index(statetype(Hsector)(0), Hsector) == 0
    @test state_index(statetype(Hsector)(1 << 10), Hsector) == 0
    @test state_index(basisstate(2, Hprod), Hconstrained) == 0
    @test state_index(T(2), Hlarge) == 0
    @test state_index(FockNumber(3), Hint) == 0
    @test state_index(statetype(Hsparse)(3), Hsparse) == 0
    @test state_index(statetype(hilbert_space(f, 1:4))(1), Hint) == 2
    @test_throws MethodError state_index(BosonicState(0), Hsector)
    @test_throws MethodError state_index(FockNumber(0), Hconstrained)

    Ha, Hb = hilbert_space(f, 1:3), hilbert_space(f, 4:6)
    Hsector2 = constrain_space(tensor_product(Ha, Hb), NumberConservation(1, [Ha]) * NumberConservation(2, [Hb]))
    @test Hsector2 == Hsector && hash(Hsector2) == hash(Hsector)
end

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
function localize_constraint(c::SectorConstraint{<:FilterConstraint{<:Any,Missing,Missing}}, H::SectorHilbertSpace)
    # Bind the free-standing reducer to H as its single subspace.
    # sector_function on the larger space will split the state, extract the H-substate,
    # apply the original reducer to it, and return the label via `only`.
    SectorConstraint(only, (c.filter.reducer,), (H,))
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