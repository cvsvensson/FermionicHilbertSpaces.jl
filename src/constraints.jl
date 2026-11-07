
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
    sums = SumConstraint[c for c in prunable if c isa SumConstraint]
    branches = AbstractConstraint[branch_constraint(c, space) for c in prunable if !(c isa SumConstraint)]
    isempty(sums) || push!(branches, branch_constraint(sums, space))
    return ProductConstraint(map(identity, branches))
end
supports_branch_pruning(c::ProductConstraint) = any(supports_branch_pruning, c.constraints)
supports_filtering(c::ProductConstraint) = any(supports_filtering, c.constraints)
# A product with a `quantity => spec` factor also groups when only some factors do, and the others then filter.
supports_sector_grouping(c::ProductConstraint) = all(supports_sector_grouping, c.constraints) || (_spec_form(c) && any(supports_sector_grouping, c.constraints))

supply_missing_constraint_info(constraint::ProductConstraint, space, spaces) = ProductConstraint(map(cons -> supply_missing_constraint_info(cons, space, spaces), constraint.constraints))

# A constraint on a function of the basis state: `reducer(state)`, or with `maps` and `spaces`, `reducer` of the values
# of the maps on the states of the spaces. `rule` turns that value into the sector label, `nothing` to keep the state
# without a label, or `missing` to discard it.
struct StateConstraint{R, FS, H, U} <: AbstractConstraint
    reducer::R
    maps::FS
    spaces::H
    rule::U
    function StateConstraint(reducer::R, maps::FS, spaces::H, rule::U) where {R, FS, H, U}
        _check_maps_length(maps, spaces)
        return new{R, FS, H, U}(reducer, maps, spaces, rule)
    end
end
_keep_true(keep) = keep ? nothing : missing
_label(rule::Function, value) = rule(value)
_labels(::typeof(_keep_true)) = false
_labels(::typeof(identity)) = true

"""
    FilterConstraint(reducer; maps=missing, spaces=missing)
    FilterConstraint(reducer, maps, spaces)

Constraint that keeps complete basis states for which `reducer` returns `true`.

With only `reducer`, the reducer is called directly as `reducer(state)`. When
`maps` and `spaces` are provided, the state is split into the selected
spaces, each map is applied to its corresponding subspace, and the
reducer is called on the resulting values. A single function is applied to all
spaces; a collection of maps supplies one function per subspace.
"""
const FilterConstraint{R, FS, H} = StateConstraint{R, FS, H, typeof(_keep_true)}
FilterConstraint(reducer, maps, spaces) = StateConstraint(reducer, maps, spaces, _keep_true)
FilterConstraint(reducer::Function; maps = missing, spaces = missing) = FilterConstraint(reducer, maps, spaces)

"""
    SectorConstraint(reducer; maps=missing, spaces=missing)
    SectorConstraint(reducer, maps, spaces)

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
const SectorConstraint{R, FS, H} = StateConstraint{R, FS, H, typeof(identity)}
SectorConstraint(reducer, maps, spaces) = StateConstraint(reducer, maps, spaces, identity)
SectorConstraint(reducer; maps = missing, spaces = missing) = SectorConstraint(reducer, maps, spaces)

function _check_maps_length(maps, spaces, name = "maps")
    (maps isa Function || ismissing(maps) || ismissing(spaces)) && return nothing
    length(maps) == length(spaces) || throw(ArgumentError("Got $(length(maps)) $name for $(length(spaces)) spaces. Give one per space, or pass `spaces` explicitly."))
    return nothing
end
supports_branch_pruning(::StateConstraint) = false
# Without `maps`, the reducer acts directly on the full state.
supply_missing_constraint_info(c::StateConstraint, space, spaces) = ismissing(c.maps) || !ismissing(c.spaces) ? c : StateConstraint(c.reducer, c.maps, spaces, c.rule)
quantity_function(c::StateConstraint{<:Any, Missing, Missing}, ::AbstractHilbertSpace) = c.reducer
function quantity_function(c::StateConstraint, space::AbstractHilbertSpace)
    mapper = state_mapper(space, c.spaces)
    maps = c.maps
    return state -> c.reducer(_mapped_values(maps, unique_split_state(state, mapper)))
end
_mapped_values(f::Function, states) = Iterators.map(f, states)
_mapped_values(maps, states) = Iterators.map((s, f) -> f(s), states, maps)
map_spaces(f, c::StateConstraint) = StateConstraint(c.reducer, c.maps, _map_spaces(f, c.spaces), c.rule)

# A constraint on a sum over the states of `constraint_spaces(c)` (`missing` for the spaces
# `tensor_product` combines, or the whole space in `constrain_space`). The internal interface:
# `allowed_values(c)` (`missing` allows every sum), `contribution(c, k, state)` of the state of the
# `k`th space, `modulus(c)` (`nothing`, or `m` to sum modulo `m`), `label(c, sum)` for the sector
# label, and `with_spaces(c, spaces)`.
abstract type AbstractAdditiveConstraint <: AbstractConstraint end
constraint_spaces(c::AbstractAdditiveConstraint) = c.spaces
modulus(::AbstractAdditiveConstraint) = nothing
label(::AbstractAdditiveConstraint, value) = value

"""
    AdditiveConstraint(allowed_values; spaces=missing, maps=missing)
    AdditiveConstraint(allowed_values, maps)

Constraint enforcing that the sum of per-subspace contributions lies in
`allowed_values`. The contribution of each subspace in `spaces` is the value of
its map applied to the subspace's state. A single function in `maps` is applied
to all spaces; a collection of maps supplies one function per subspace.

If `spaces` is missing, `tensor_product` uses the spaces it combines, and
`constrain_space` treats the whole space as a single subspace.
"""
struct AdditiveConstraint{T, H, F} <: AbstractAdditiveConstraint
    allowed_values::T
    spaces::H
    maps::F
    function AdditiveConstraint(allowed_values, spaces, maps)
        _check_maps_length(maps, spaces)
        new{typeof(allowed_values),typeof(spaces),typeof(maps)}(allowed_values, spaces, maps)
    end
end
AdditiveConstraint(allowed_values; spaces=missing, maps=missing) = AdditiveConstraint(allowed_values, spaces, maps)
AdditiveConstraint(allowed_values, maps) = AdditiveConstraint(allowed_values; maps)
allowed_values(c::AdditiveConstraint) = c.allowed_values
contribution(c::AdditiveConstraint{<:Any, <:Any, <:Function}, k, state) = c.maps(state)
contribution(c::AdditiveConstraint, k, state) = c.maps[k](state)
with_spaces(c::AdditiveConstraint, spaces) = AdditiveConstraint(c.allowed_values, spaces, c.maps)
# Without spaces, the maps are only applied to the factors of a `tensor_product`.
supports_filtering(::AdditiveConstraint{<:Any, Missing}) = false
supports_sector_grouping(::AdditiveConstraint{<:Any, Missing}) = false

"""
    particle_number(state)

Return the number of particles in the basis state `state`: the number of occupied modes for
fermions, the occupation for a boson, and the total over the factors of a product state. It is not
defined for spins. This is the quantity [`NumberConservation`](@ref) conserves.
"""
function particle_number end

"""
    NumberConservation(total=missing; spaces=missing, weights=missing)
    NumberConservation(total, spaces; weights=missing)
    NumberConservation(total, spaces, weights)
    NumberConservation(space)

Constraint enforcing conservation of a (possibly weighted) particle number, optionally on selected spaces.
`total` can be a single value or collection of allowed values; `missing` allows any value.
`spaces` restricts the count to the given subspaces (default: all of them). `weights` gives one
weight per subspace in `spaces` (default: all `1`), and the conserved quantity is the weighted sum of their particle numbers.
"""
struct NumberConservation{T, H, W} <: AbstractAdditiveConstraint
    total::T
    spaces::H
    weights::W
    function NumberConservation(_total, _spaces, _weights)
        total = _normalize_constraint_values(_total)
        subspace = _normalize_constraint_subspace(_spaces)
        weights = _normalize_constraint_values(_weights)
        _check_maps_length(weights, subspace, "weights")
        new{typeof(total),typeof(subspace),typeof(weights)}(total, subspace, weights)
    end
end
_normalize_constraint_subspace(subspace::AbstractHilbertSpace) = (subspace,)
_normalize_constraint_subspace(spaces) = spaces
_normalize_constraint_subspace(spaces::Missing) = spaces
allowed_values(c::NumberConservation) = c.total
contribution(::NumberConservation{<:Any, <:Any, Missing}, k, state) = particle_number(state)
contribution(c::NumberConservation, k, state) = c.weights[k] * particle_number(state)
with_spaces(c::NumberConservation, spaces) = NumberConservation(c.total, spaces, c.weights)
NumberConservation(total=missing; weights=missing, spaces=missing) = NumberConservation(total, spaces, weights)
NumberConservation(total, spaces; weights=missing) = NumberConservation(total; spaces, weights)
NumberConservation(space::AbstractHilbertSpace) = NumberConservation(missing; spaces=(space,), weights=missing)

"""
    ParityConservation(parities=[-1, 1]; spaces=missing)
    ParityConservation(parities, spaces)
    ParityConservation(space)

Constraint enforcing allowed fermion parities, optionally on selected spaces.
`parities` is a parity or a collection of parities, each `1` (even) or `-1` (odd).
Sectors are ordered by increasing quantum number, so the odd sector comes first.
"""
struct ParityConservation{H} <: AbstractAdditiveConstraint
    allowed_parities::Vector{Int}
    spaces::H
    function ParityConservation(_allowed, _spaces)
        allowed = _normalize_constraint_values(_allowed)
        allowed in Set([[-1, 1], [1], [-1]]) || throw(ArgumentError("The allowed parities must be a subset of `[-1, 1]`, got $allowed."))
        subspace = _normalize_constraint_subspace(_spaces)
        new{typeof(subspace)}(allowed, subspace)
    end
end
ParityConservation(parities=[-1, 1]; spaces=missing) = ParityConservation(parities, spaces)
ParityConservation(space::AbstractHilbertSpace) = ParityConservation(; spaces=(space,))
allowed_values(c::ParityConservation) = c.allowed_parities
contribution(::ParityConservation, k, state) = particle_number(state)
modulus(::ParityConservation) = 2
label(::ParityConservation, n) = iseven(n) ? 1 : -1
with_spaces(c::ParityConservation, spaces) = ParityConservation(c.allowed_parities, spaces)

# A constraint that allows every value restricts nothing, so pruning by it would only cost time.
function supports_branch_pruning(c::AbstractAdditiveConstraint)
    allowed, m = allowed_values(c), modulus(c)
    ismissing(allowed) && return false
    return isnothing(m) || !all(r -> label(c, r) in allowed, 0:(m - 1))
end
supports_filtering(::AbstractAdditiveConstraint) = true
supports_sector_grouping(::AbstractAdditiveConstraint) = true
supply_missing_constraint_info(c::AbstractAdditiveConstraint, space, spaces) = ismissing(constraint_spaces(c)) ? with_spaces(c, spaces) : c
map_spaces(f, c::AbstractAdditiveConstraint) = with_spaces(c, _map_spaces(f, constraint_spaces(c)))

unique_split_state(state, mapper) = only(first(split_state(state, mapper)))

function sector_function(c::AbstractAdditiveConstraint, space::AbstractHilbertSpace)
    spaces = coalesce(constraint_spaces(c), (space,))
    c isa NumberConservation && _check_maps_length(c.weights, spaces, "weights")
    mapper = state_mapper(space, spaces)
    allowed = allowed_values(c)
    S = _sum_type(allowed)
    return function sector(state)
        substates = unique_split_state(state, mapper)
        value = label(c, _residue(modulus(c), _additive_sum(c, substates, S)))
        return ismissing(allowed) || value in allowed ? value : missing
    end
end
_additive_sum(c, substates, ::Type{S}) where {S} = sum(((k, s),) -> _summand(S, contribution(c, k, s)), enumerate(substates); init = zero(S))
_residue(::Nothing, sum) = sum
_residue(m, sum) = mod(sum, m)
# The sector label lists the labels of the grouping factors, or is the label of the only one; the other factors filter.
function sector_function(constraint::ProductConstraint, space::AbstractHilbertSpace)
    grouping = filter(supports_sector_grouping, constraint.constraints)
    keep = filter_function(ProductConstraint(filter(c -> !supports_sector_grouping(c) && supports_filtering(c), constraint.constraints)), space)
    if length(grouping) == 1
        f = sector_function(only(grouping), space)
        return state -> keep(state) ? f(state) : missing
    end
    subspace_functions = map(cons -> sector_function(cons, space), grouping)
    V = mapreduce(f -> nonmissingtype(Base.promote_op(f, statetype(space))), promote_type, subspace_functions; init = Union{})
    function sector(state)
        keep(state) || return missing
        sectors = Vector{V}(undef, length(subspace_functions))
        for (i, f) in enumerate(subspace_functions)
            label = f(state)
            ismissing(label) && return missing
            sectors[i] = label
        end
        return isconcretetype(V) ? sectors : map(identity, sectors) # narrow the element type
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

sectors(::AbstractConstraint) = nothing

map_spaces(f, c) = c
map_spaces(f, c::ProductConstraint) = ProductConstraint(map(ci -> map_spaces(f, ci), c.constraints))
_map_spaces(f, spaces) = map(f, spaces)
_map_spaces(f, ::Missing) = missing


"""
    constrain_space(space, constraint[, states])
    constrain_space(space, states)

Build a constrained Hilbert space from `space`.

With a `constraint`, keep the basis states of `space` that satisfy it. If `states` is
given, only those states are candidates. Constraints that group states into sectors,
such as `NumberConservation`, return a space with sectors, with states ordered by sector.
`constrain_space(space, NoSymmetry())` returns `space` unchanged.
A constraint can also be `quantity => spec` with a diagonal operator `quantity`, such as
`constrain_space(H, N => 2)` with `N = sum(f[i]'f[i] for i in 1:4)`; see [Constraints and sectors](@ref) for what `spec` can be.
A `quantity => spec` constraint on a space with sectors keeps its labels, and its own labels follow them.

With only `states`, return the subspace spanned by exactly those basis states, in the
given order; this is equivalent to `constrain_space(space, NoSymmetry(), states)`.
"""
constrain_space

@testitem "ProductSymmetry" begin
    using FermionicHilbertSpaces: particle_number
    labels = 1:4
    qn = NumberConservation() * ParityConservation()
    @fermions f
    H = hilbert_space(f, labels, qn)
    @test collect(quantumnumbers(H)) == [[n, (-1)^n] for n in 0:4]
    @test all(q -> q isa Vector{Int}, quantumnumbers(H))
    # Sector labels of mixed types
    qn_mixed = NumberConservation() * FermionicHilbertSpaces.SectorConstraint(s -> particle_number(s) >= 2 ? :hi : :lo)
    H_mixed = hilbert_space(f, labels, qn_mixed)
    @test dim(H_mixed) == 2^4
    @test Set(quantumnumbers(H_mixed)) == Set(Any[n, n >= 2 ? :hi : :lo] for n in 0:4)
    # Numeric labels are promoted to a common type
    qn_numeric = NumberConservation() * FermionicHilbertSpaces.SectorConstraint(s -> isodd(particle_number(s)) ? 0.5 : 1.5)
    q_numeric = collect(quantumnumbers(hilbert_space(f, 1:2, qn_numeric)))
    @test [0, 1.5] in q_numeric
    @test all(q -> q isa Vector{Float64}, q_numeric)
    qn = prod(NumberConservation(missing, hilbert_space(f[l])) for l in labels)
    H = hilbert_space(f, labels, qn)
    @test dim(H) == 2^4
    @test all(isone ∘ dim, sectors(H))
end

@testitem "ProductConstraint filtering" begin
    using FermionicHilbertSpaces: particle_number
    @fermions f
    H = hilbert_space(f, 1:2)
    constraint = NumberConservation(1) * FilterConstraint(state -> state != basisstate(2, H))

    Hconstrained = constrain_space(H, constraint)

    @test dim(Hconstrained) == 1
    @test all(particle_number(state) == 1 for state in basisstates(Hconstrained))
end

@testitem "Constraint maps must match spaces" begin
    using FermionicHilbertSpaces: AdditiveConstraint, particle_number
    @fermions f
    Hs = [hilbert_space(f[i]) for i in 1:3]
    @test_throws ArgumentError tensor_product(Hs...; constraint=AdditiveConstraint(1; maps=(particle_number, particle_number)))
    @test_throws ArgumentError tensor_product(Hs...; constraint=FilterConstraint(all; maps=ntuple(_ -> isone ∘ particle_number, 4)))
    @test_throws ArgumentError AdditiveConstraint(1; spaces=Hs[1:2], maps=(particle_number,))
end
