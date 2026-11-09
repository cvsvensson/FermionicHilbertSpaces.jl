struct ConstrainedSpace{B,H,S,I} <: AbstractSubspace{B,H}
    parent::H
    states::S
    index::I
    ConstrainedSpace{B,H,S,I}(parent, states, index) where {B,H,S,I} = new{B,H,S,I}(parent, states, index)
end

function ConstrainedSpace(space, states; constraint=NoSymmetry())
    parent, members = unnest(space, states)
    stored = _stored_states(members)
    ConstrainedSpace(parent, stored, build_subspace_index(constraint, parent, stored))
end
ConstrainedSpace(space::H, states::S, index) where {H,S} = ConstrainedSpace{eltype(states),H,S,typeof(index)}(space, states, index)
ConstrainedSpace(::AbstractSubspace, states, index) = throw(nested_subspace_error())

Base.parent(H::ConstrainedSpace) = H.parent
subspace_states(H::ConstrainedSpace) = H.states
subspace_index(H::ConstrainedSpace) = H.index
rewrap(H::ConstrainedSpace, parent) = ConstrainedSpace(parent, H.states, H.index)

constrain_space(space::AbstractHilbertSpace, ::NoSymmetry) = space
constrain_space(space::AbstractHilbertSpace, states::Union{AbstractVector, BasisStates}) = _holds_constraints(states) ? constrain_space(space, as_constraint(states)) : constrain_space(space, NoSymmetry(), states)

constrain_space(space, ::NoSymmetry, states) = ConstrainedSpace(space, _states_of(space, states))
function constrain_space(space, constraint::AbstractConstraint)
    split = pruning_split(space, constraint)
    isnothing(split) && return constrain_space(space, constraint, basisstates(space))
    # The search varies its last space fastest and `basisstates(space)` its first atom, so the reversed split keeps that order.
    states = generate_states(reverse(split), supply_missing_constraint_info(constraint, space, split), space)
    # Sectors are labelled by `constraint` as given, as splitting each state over `split` again would be slower.
    return constrain_states(space, constraint, states, unpruned_part(constraint))
end
function constrain_space(space, constraint::AbstractConstraint, labels)
    supports_filtering(constraint) || throw(ArgumentError("Cannot constrain a space with an unsupported constraint, got `$constraint`."))
    return constrain_states(space, constraint, _states_of(space, labels))
end
# The subspace of the candidate `states` that satisfy `constraint`, grouped into sectors when it supports it.
# Only `unchecked`, the part of `constraint` that `states` may violate, is filtered by when not grouping.
# A constraint that cannot filter is taken to hold for all of `states`.
function constrain_states(space, constraint, states, unchecked = nothing)
    constraint = resolve(constraint, space, states)
    unchecked = isnothing(unchecked) ? constraint : resolve(unchecked, space, states)
    if supports_sector_grouping(constraint) && supports_filtering(constraint)
        grouping = _with_sector_labels(space, constraint)
        return sector_space(space, states, sector_function(grouping, space), grouping)
    elseif supports_filtering(unchecked)
        return _restricted_space(space, collect(Iterators.filter(filter_function(unchecked, space), states)), constraint)
    end
    return ConstrainedSpace(space, states; constraint)
end
# A `quantity => spec` constraint on a space with sectors keeps them, and its own labels follow theirs.
_keeps_sectors(space, constraint) = false
_keeps_sectors(space::SectorHilbertSpace, constraint) = !isnothing(space.constraint) && _spec_form(constraint)
_with_sector_labels(space, constraint) = _keeps_sectors(space, constraint) ? space.constraint * constraint : constraint
function _restricted_space(space, states, constraint)
    _keeps_sectors(space, constraint) || return ConstrainedSpace(space, states; constraint)
    return sector_space(space, states, sector_function(space.constraint, parent(space)), space.constraint)
end
# The factors of `constraint` that the branch pruning of `generate_states` does not enforce.
unpruned_part(constraint) = ProductConstraint(filter(!supports_branch_pruning, collect(constraint_factors(constraint))))

# Spaces that the branch constraint of `c` reads: `missing` for the whole state, which any split fits,
# and `nothing` when no split is known to fit, such as for a user's `BranchConstraint`.
branch_spaces(c) = nothing
branch_spaces(c::AbstractAdditiveConstraint) = constraint_spaces(c)
# Without spaces, per-space maps or weights act on the factors of a `tensor_product`, which no split is known to match.
branch_spaces(::AdditiveConstraint{<:Any, Missing}) = nothing
branch_spaces(c::NumberConservation{<:Any, Missing}) = ismissing(c.weights) ? missing : nothing
constraint_factors(c::ProductConstraint) = c.constraints
constraint_factors(c) = (c,)

# Spaces to generate the states of `H` over with branch pruning by `constraint`: the spaces of its prunable
# factors and the remaining atoms of `H`, in the order of the atoms. Returns `nothing` when the spaces are not
# consecutive atoms of `H`, which the basis order needs, or overlap without being equal, which the branch
# constraints cannot locate in the split. It also returns `nothing` when all prunable factors are residues of sums
# over the whole state, such as a parity, which only prune at the last atom.
function pruning_split(H, constraint)
    isconstrained(H) && return nothing # the parent may have many more states than `H`
    prunable = filter(supports_branch_pruning, collect(constraint_factors(constraint)))
    spaces = map(branch_spaces, prunable)
    (isempty(prunable) || any(isnothing, spaces)) && return nothing
    all(c -> ismissing(branch_spaces(c)) && !isnothing(modulus(c)), prunable) && return nothing
    atoms = collect(atomic_factors(H))
    owner = zeros(Int, length(atoms))
    blocks = []
    for space in Iterators.flatten(skipmissing(spaces))
        range = atom_range(space, H, atoms)
        isnothing(range) && return nothing
        k = owner[first(range)]
        !iszero(k) && blocks[k][1] == range && continue
        all(iszero, view(owner, range)) || return nothing
        push!(blocks, (range, space))
        owner[range] .= length(blocks)
    end
    return [iszero(k) ? atoms[i] : blocks[k][2] for (i, k) in enumerate(owner) if iszero(k) || first(blocks[k][1]) == i]
end
# The positions of the atoms of `space` in `atoms`, the atoms of `H`, if they are consecutive and equal, otherwise `nothing`.
function atom_range(space, H, atoms)
    space isa AbstractHilbertSpace && !isconstrained(space) || return nothing
    space_atoms = collect(atomic_factors(space))
    start = atom_position(first(space_atoms), H)
    range = start:(start + length(space_atoms) - 1)
    !iszero(start) && last(range) <= length(atoms) && atoms[range] == space_atoms || return nothing
    return range
end
# Labels become states as in `Ket` and `state_index`, but a basis state or a value of the state type is kept as is,
# so that a space whose states are strings is not parsed. States outside a subspace are dropped by `unnest`.
_states_of(space, states::BasisStates) = states
function _states_of(space, labels)
    return map(labels) do label
        # A subspace can store basis states of another type than its parent, which the parent cannot look up.
        label isa AbstractBasisState && !(label isa statetype(space)) && return label
        state = label isa statetype(space) ? label : interpret_state(label, space)
        space isa AbstractSubspace || !iszero(state_index(state, space)) || throw(ArgumentError("The state `$label` is not a basis state of the space."))
        state
    end
end

"""
    build_subspace_index(constraint, parent, states)

`default_subspace_index(constraint, parent, states)`, after checking that it gives the first,
middle and last of `states` their positions. Throws an `ArgumentError` otherwise.
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
`RootIndexDict`, or `nothing` when `space` has no arithmetic `state_index` with `Int` values.
Throws an `ArgumentError` if a state appears more than once.
"""
function root_keyed_index(space, states)
    has_arithmetic_state_index(space) || return nothing
    B = statetype(space)
    (eltype(states) == B || all(state -> same_state_kind(state, B), states)) || return nothing
    d = dim(space)
    # Wide integer keys measured slower than keying by state.
    d isa Int || return nothing
    if d <= min(INDEX_TABLE_RATIO * length(states), INDEX_TABLE_MAX_LENGTH)
        table = fill_positions!(zeros(Int32, d), space, states)
        return isnothing(table) ? nothing : IndexTable(table)
    end
    positions = fill_positions!(sizehint!(Dict{Int, Int}(), length(states)), space, states)
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
struct IndexTable{V}
    positions::V
end
function state_position(table::IndexTable, state, parent)
    k = state_index(state, parent)
    iszero(k) ? 0 : Int(table.positions[k])
end
"""
    RootIndexDict(positions)

A subspace index whose `positions[state_index(state, parent)]` is the position of `state`. It
is keyed by the integer index rather than by the state.
"""
struct RootIndexDict
    positions::Dict{Int, Int}
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

@testitem "constrain_space on subspaces matches filtering all basis states" begin
    using FermionicHilbertSpaces: AdditiveConstraint, particle_number
    @fermions f
    @spins s 1 // 2
    N = 8
    Hs = [hilbert_space(f[i]) for i in 1:N]
    H = tensor_product(Hs)
    dipole = AdditiveConstraint([12, 13]; spaces = Hs, maps = [s -> i * particle_number(s) for i in 1:N])
    Hm = tensor_product(hilbert_space(f, 1:3), hilbert_space(s[1]), hilbert_space(s[2]))
    mixed = NumberConservation(1, hilbert_space(f, 1:2)) * AdditiveConstraint([0]; spaces = map(hilbert_space, (s[1], s[2])), maps = s -> s.m)
    @bosons b
    Hbs = [hilbert_space(b[i], 3) for i in 1:4]
    @majoranas γ
    Hγ = hilbert_space(γ, 1:8)
    for (space, constraint) in [
            (tensor_product(Hbs), NumberConservation(1:2, Hbs[1:3]) * NumberConservation(2, Hbs[3:4])),
            (Hγ, NumberConservation(1, hilbert_space(γ, 1:4))),
            (H, NumberConservation(0:2, Hs[3:7])),
            (H, NumberConservation(N ÷ 2, Hs) * dipole),
            (H, NumberConservation(3, Hs[1:6]) * FilterConstraint(s -> isodd(particle_number(s)))),
            (Hm, mixed),
            (H, NumberConservation(1, hilbert_space(f, 1:4)) * NumberConservation(2, Hs[3:6])),
            (hilbert_space(f, 1:N), NumberConservation(2:3)),
            (hilbert_space(f, 1:N), NumberConservation(3) * ParityConservation(1)),
            (tensor_product(Hbs), NumberConservation(2)),
            (Hγ, NumberConservation(1)),
        ]
        @test constrain_space(space, constraint) == constrain_space(space, constraint, basisstates(space))
    end
end

@testitem "constrain_space reads labels unless they are states" begin
    using LinearAlgebra
    struct Level
        n::Int
    end
    @fermions f
    H = hilbert_space(f, 1:3)
    Hs = constrain_space(H, ["100", "010"])
    @test basisstates(Hs) == [Ket("100", H).ket, Ket("010", H).ket]
    @test representation(f[1]' * f[2], Hs, :dense) == [0 1; 0 0]
    @test partial_trace(Diagonal([1, 2]), Hs => hilbert_space(f, 1:1)) == Diagonal([2, 1])

    Hg = GenericHilbertSpace(:a, ["up", "down", "x"])
    @test basisstates(constrain_space(Hg, ["x", "up"])) == ["x", "up"]
    P = tensor_product(Hg, hilbert_space(f, 1:2))
    Pc = constrain_space(P, [("up", "10"), ("x", "01")])
    @test partial_trace(Diagonal([1, 2]), Pc => Hg) == Diagonal([1, 0, 2])

    Hl = GenericHilbertSpace(:l, Level.(1:3))
    @test basisstates(constrain_space(Hl, [Level(3), Level(1)])) == [Level(3), Level(1)]
    @test_throws ArgumentError constrain_space(Hl, [Level(4)])
    @test_throws ArgumentError constrain_space(Hg, FilterConstraint(!=("x")), ["up", "y"])
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

@testitem "Tensor product with a product of sectors keeps their constraints" begin
    @fermions f
    @boson b
    Hb = hilbert_space(b, 3)
    H2 = tensor_product(hilbert_space(f, 1:2, NumberConservation(1)), hilbert_space(f, 3:4, ParityConservation(-1)))
    Hf1, Hf2 = hilbert_space(f, 1:2), hilbert_space(f, 3:4)
    H3 = tensor_product(Hf1, Hf2; constraint = NumberConservation(1, [Hf1]) * NumberConservation(1, [Hf2]))
    for H in (H2, H3)
        H4 = tensor_product(H, Hb)
        @test dim(H4) == dim(H) * dim(Hb)
        @test Set(quantumnumbers(H4)) == Set(quantumnumbers(H))
    end
end

@testitem "Tensor product of a space with labelled and filtered constraints" begin
    @spins S 1 // 2
    @fermions f
    nup(s) = count(x -> x.m > 0, s.states)
    H1 = hilbert_space(S, 1:3, (sum(S[i][:z] for i in 1:3), nup => 1))
    H = tensor_product(H1, hilbert_space(f, 1:2))
    @test dim(H) == 4 * dim(H1)
    @test Set(quantumnumbers(H)) == Set(quantumnumbers(H1))
end

@testitem "Tensor product keeps the sector rule of a factor" begin
    using FermionicHilbertSpaces: particle_number
    @fermions f
    rule = s -> particle_number(s) >= 2 ? :hi : :lo
    H1 = hilbert_space(f, 1:2, NumberConservation() * SectorConstraint(rule))
    H2 = hilbert_space(f, 3:4)
    H = tensor_product(H1, H2)
    @test Set(quantumnumbers(H)) == Set(quantumnumbers(H1))
    for qn in quantumnumbers(H1)
        expected = tensor_product(constrain_space(hilbert_space(f, 1:2), collect(basisstates(sector(qn, H1)))), H2)
        @test Set(basisstates(sector(qn, H))) == Set(basisstates(expected))
    end
end

"""
    localize_constraint(c, H::SectorHilbertSpace)

Return a version of constraint `c` that is bound to `H` as its subspace, so that
when `sector_function` is later called on a larger combined space it knows to extract
the contribution from `H` only.  Already-localized (non-`Missing` subspace) or
unsupported constraints are returned unchanged.
"""
localize_constraint(c::AbstractConstraint, ::SectorHilbertSpace) = c
localize_constraint(c::AbstractAdditiveConstraint, H::SectorHilbertSpace) = supply_missing_constraint_info(c, H, (H,))
# The states of `H` already satisfy the filtering factors, so only the grouping factors are kept.
localize_constraint(c::ProductConstraint, H::SectorHilbertSpace) = ProductConstraint(map(ci -> localize_constraint(ci, H), filter(supports_sector_grouping, c.constraints)))
function localize_constraint(c::StateConstraint{<:Any, Missing, Missing}, H::SectorHilbertSpace)
    # Bind the free-standing reducer to H as its single subspace.
    # sector_function on the larger space will split the state, extract the H-substate,
    # apply the original reducer to it, and return the label via `only`.
    return StateConstraint(only, (c.reducer,), (H,), c.rule)
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