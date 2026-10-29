struct ConstrainedSpace{B,H,S,C,I} <: AbstractSubspace{B,H}
    parent::H
    states::S
    constraint::C
    index::I
    ConstrainedSpace{B,H,S,C,I}(parent, states, constraint, index) where {B,H,S,C,I} = new{B,H,S,C,I}(parent, states, constraint, index)
end

# `constraint` is what selected `states`; `NoSymmetry()` means they were listed explicitly.
function ConstrainedSpace(space, states; constraint=NoSymmetry())
    parent, members = unnest(space, states)
    stored = _stored_states(members)
    selected_by = constraint isa NoSymmetry ? constraint : index_constraint(coalesce(combine_constraints(own_constraint(space), constraint), NoSymmetry()), parent)
    ConstrainedSpace(parent, stored, selected_by, build_subspace_index(selected_by, parent, stored))
end
ConstrainedSpace(space::H, states::S, constraint, index) where {H,S} = ConstrainedSpace{eltype(states),H,S,typeof(constraint),typeof(index)}(space, states, constraint, index)
ConstrainedSpace(::AbstractSubspace, states, constraint, index) = throw(nested_subspace_error())

Base.parent(H::ConstrainedSpace) = H.parent
subspace_states(H::ConstrainedSpace) = H.states
subspace_index(H::ConstrainedSpace) = H.index
rewrap(H::ConstrainedSpace, parent) = ConstrainedSpace(parent, H.states, H.constraint, H.index)

constrain_space(space::AbstractHilbertSpace, ::NoSymmetry) = space
constrain_space(space::AbstractHilbertSpace, states::Union{AbstractVector,BasisStates}) = constrain_space(space, NoSymmetry(), states)

constrain_space(space, ::NoSymmetry, states) = ConstrainedSpace(space, states)
function constrain_space(space, constraint::AbstractConstraint, states=basisstates(space))
    if supports_sector_grouping(constraint)
        f = sector_function(constraint, space)
        return sector_space(space, states, f, constraint)
    elseif supports_filtering(constraint)
        f = filter_function(constraint, space)
        filtered_states = collect(Iterators.filter(f, states))
        return ConstrainedSpace(space, filtered_states; constraint)
    else
        throw(ArgumentError("Cannot constrain a space with an unsupported constraint, got `$constraint`."))
    end
end

"""
    build_subspace_index(constraint, parent, states)

`default_subspace_index(constraint, parent, states)`, after checking that it gives the first,
middle and last of `states` their positions. Throws an `ArgumentError` otherwise. `constraint`
is in the form given by `index_constraint`.
"""
function build_subspace_index(constraint, parent, states)
    index = default_subspace_index(constraint, parent, states)
    isempty(states) && return index
    n = length(states)
    for i in unique!([1, cld(n, 2), n])
        state = states[firstindex(states)+i-1]
        position = state_position(index, state, parent)
        position == i || throw(ArgumentError("The subspace index $(typeof(index)) gives position $position to the state `$state` at position $i."))
    end
    return index
end

"""
    index_constraint(constraint, parent)

The form of `constraint` that `default_subspace_index` receives. A product is flattened into a
`ProductConstraint` of a tuple, or into its only factor, and `NoSymmetry` factors are dropped.
The `spaces` of an unweighted `NumberConservation` and of a `ParityConservation` become the
atoms of `parent` they act on, in the order of `parent`, also when `spaces` was `missing`.
"""
index_constraint(constraint, parent) = constraint
function index_constraint(constraint::ProductConstraint, parent)
    parts = Tuple(Iterators.flatten(map(c -> _product_parts(index_constraint(c, parent)), constraint.constraints)))
    isempty(parts) && return NoSymmetry()
    length(parts) == 1 ? only(parts) : ProductConstraint(parts)
end
_product_parts(c) = (c,)
_product_parts(c::ProductConstraint) = c.constraints
_product_parts(::NoSymmetry) = ()
index_constraint(c::NumberConservation{<:Any,<:Any,Missing}, parent) = NumberConservation(c.total, parent_atoms(c.spaces, parent), missing)
index_constraint(c::ParityConservation, parent) = ParityConservation(c.allowed_parities, parent_atoms(c.spaces, parent))
parent_atoms(::Missing, parent) = Tuple(atomic_factors(parent))
function parent_atoms(spaces, parent)
    positions = [atom_position(atom, parent) for space in spaces for atom in (space isa AbstractHilbertSpace ? atomic_factors(space) : (space,))]
    any(iszero, positions) && return spaces
    Tuple(atomic_factor(parent, p) for p in sort!(unique!(positions)))
end

# What selected the states of a space, relative to its parent: `NoSymmetry()` for all of them, `missing` if unknown.
own_constraint(H) = isconstrained(H) ? missing : NoSymmetry()
own_constraint(H::ConstrainedSpace) = H.constraint isa NoSymmetry ? missing : H.constraint
own_constraint(H::SectorHilbertSpace) = isnothing(H.constraint) ? missing : H.constraint
combine_constraints(a, b) = (ismissing(a) || ismissing(b)) ? missing : a isa NoSymmetry ? b : b isa NoSymmetry ? a : a * b

# What selected the states of the factor `H` of a product, acting on `H` only, or `missing` if unknown.
factor_constraint(H) = localize_constraint(own_constraint(H), H)
combined_factor_constraint(spaces) = reduce(combine_constraints, map(factor_constraint, spaces); init=NoSymmetry())

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
    index_table(space, states)

An `IndexTable` of the positions of `states`, or `nothing` when no table is used.
Throws an `ArgumentError` if a state appears more than once.
"""
function index_table(space, states)
    has_arithmetic_state_index(space) || return nothing
    B = statetype(space)
    (eltype(states) == B || all(state -> same_state_kind(state, B), states)) || return nothing
    d = dim(space)
    (d isa Int && d <= min(INDEX_TABLE_RATIO * length(states), INDEX_TABLE_MAX_LENGTH)) || return nothing
    table = zeros(Int32, d)
    for (i, state) in enumerate(states)
        k = state_index(state, space)
        iszero(k) && return nothing
        iszero(table[k]) || throw(_duplicate_state_error(state))
        table[k] = i
    end
    return IndexTable(table)
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
    for H in (Hsector, Hconstrained, Hprod_sector, Hlarge, Hint, TransposedSpace(Hsector), add_tag(Hsector, :x), add_tag(Hconstrained, :x))
        @test all(state_index(basisstate(i, H), H) == i for i in 1:dim(H))
    end
    @test state_index(statetype(Hsector)(0), Hsector) == 0
    @test state_index(statetype(Hsector)(1 << 10), Hsector) == 0
    @test state_index(basisstate(2, Hprod), Hconstrained) == 0
    @test state_index(T(2), Hlarge) == 0
    @test state_index(FockNumber(3), Hint) == 0
    @test state_index(statetype(hilbert_space(f, 1:4))(1), Hint) == 2
    @test_throws MethodError state_index(BosonicState(0), Hsector)
    @test_throws MethodError state_index(FockNumber(0), Hconstrained)

    Ha, Hb = hilbert_space(f, 1:3), hilbert_space(f, 4:6)
    Hsector2 = constrain_space(tensor_product(Ha, Hb), NumberConservation(1, [Ha]) * NumberConservation(2, [Hb]))
    @test Hsector2 == Hsector && hash(Hsector2) == hash(Hsector)
end

"""
    localize_constraint(c, H)

Return a version of constraint `c` that is bound to `H` as its subspace, so that
when `sector_function` is later called on a larger combined space it knows to extract
the contribution from `H` only. Already-localized (non-`Missing` subspace) constraints are
returned unchanged. A constraint that cannot be bound gives `missing`, or itself when `H` is a
`SectorHilbertSpace`.
"""
localize_constraint(c, H) = H isa SectorHilbertSpace ? c : missing
localize_constraint(c::NoSymmetry, H) = c
localize_constraint(c::NumberConservation, H) = ismissing(c.spaces) ? NumberConservation(_local_values(c, H), H, c.weights) : c
_local_values(c, H) = c.total
_local_values(c, H::SectorHilbertSpace) = collect(keys(H.qn_to_states))
localize_constraint(c::ParityConservation, H) = ismissing(c.spaces) ? ParityConservation(c.allowed_parities, H) : c
function localize_constraint(c::ProductConstraint, H)
    parts = map(sub -> localize_constraint(sub, H), c.constraints)
    any(ismissing, parts) ? missing : ProductConstraint(parts)
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
# The substate in `parent(H)` is that in `H`, and binding to the parent keeps `H` and its index out of the constraint's type.
localize_constraint(c::SectorConstraint{<:FilterConstraint{<:Any,Missing,Missing}}, H) = SectorConstraint(only, (c.filter.reducer,), (H isa AbstractSubspace ? parent(H) : H,))

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