
combine_into_group(::AbstractAtomicHilbertSpace, atoms) = only(atoms)
combine_into_group(group_id, spaces) = length(spaces) == 1 ? only(spaces) : throw(ArgumentError("Don't know how to combine spaces $spaces with atomic group $group_id into a group"))
struct ProductState{B} <: AbstractBasisState
    states::B
end
ProductState{B}(state::ProductState{B}) where B = state
ProductState{NTuple{N,S}}(states::AbstractVector{S}) where {N,S} = ProductState{NTuple{N,S}}(Tuple(states))
substate(n::Integer, state::ProductState) = state.states[n]
atomic_factors(state::ProductState) = state.states
Base.:(==)(s1::ProductState, s2::ProductState) = s1.states == s2.states
Base.hash(s::ProductState, h::UInt) = hash(s.states, h)
Base.isless(s1::ProductState, s2::ProductState) = s1.states < s2.states

symbolic_group(h::AbstractAtomicHilbertSpace) = h
# ProductSpaces consists of a list of atomic spaces and factor spaces
struct ProductSpace{B,C,A,T,LI,CI} <: AbstractProductHilbertSpace{B}
    factors::C
    atoms::Vector{A}
    atom_ordering::Dict{A,Int}
    fast_path::T
    lininds::LI
    cartinds::CI
    function ProductSpace(factors::C, atoms::Vector{A}) where {C,A}
        length(factors) == 0 && throw(ArgumentError("Product space must have at least one factor"))
        B = ProductState{Tuple{map(statetype, factors)...}}
        atom_ordering = Dict{A,Int}(a => i for (i, a) in enumerate(atoms))
        fast_path = all(has_internal_rep(f, Int) for f in factors) ? zero(Int) : missing
        lininds = LinearIndices(map(dim, factors))
        cartinds = CartesianIndices(map(dim, factors))
        LI = typeof(lininds)
        CI = typeof(cartinds)
        new{B,C,A,typeof(fast_path),LI,CI}(factors, atoms, atom_ordering, fast_path, lininds, cartinds)
    end
end
fast_path(space::ProductSpace) = space.fast_path
isconstrained(H::ProductSpace) = false
Base.:(==)(H1::ProductSpace, H2::ProductSpace) = H1.factors == H2.factors && H1.atoms == H2.atoms
Base.hash(H::ProductSpace, h::UInt) = hash(H.factors, hash(H.atoms, h))
atomic_factors(H::ProductSpace) = H.atoms
factors(H::ProductSpace) = H.factors
groups(H::ProductSpace) = H.factors
dim(H::ProductSpace) = prod(dim, factors(H); init=1)
atomic_id(H::ProductSpace) = H.atom_ordering
group_id(H::ProductSpace) = H.atom_ordering

basisstates(H::ProductSpace) = collect(Iterators.map(s -> ProductState(s), Iterators.product(map(basisstates, H.factors)...)))
function basisstate(n::Integer, H::ProductSpace{B}) where B
    inds = Tuple(H.cartinds[n])
    ProductState(map(basisstate, inds, H.factors))
end
function state_index(state::B, H::ProductSpace{B}) where B
    cartesian_index = CartesianIndex(Tuple(map(state_index, state.states, H.factors)))
    H.lininds[cartesian_index]
end
_find_atom_position(Hsub::AbstractAtomicHilbertSpace, H::ProductSpace) = get(H.atom_ordering, Hsub, 0)
function _find_position(Hsub, H::ProductSpace)
    pos = findfirst(==(Hsub), H.factors)
    isnothing(pos) && return 0
    return pos
end


maximum_particles(H::ProductSpace) = sum(maximum_particles, factors(H))
particle_number(state::ProductState) = sum(particle_number, atomic_factors(state))
parity(state::ProductState) = prod(parity, atomic_factors(state))

function atomic_substate(n, f::ProductState, space::ProductSpace)
    count = 0
    for (k, s) in enumerate(factors(space))
        add = length(atomic_factors(s))
        if count < n <= count + add
            return substate(n - count, substate(k, f))
        end
        count += add
    end
    throw(ArgumentError("Invalid substate index"))
end


@testitem "ProductSpace" begin
    import FermionicHilbertSpaces: state_index, basisstate, state_mapper, ProductState, substate, complementary_subsystem, atomic_factors, split_state
    @fermions a b
    @fermions c
    Ha = hilbert_space(a[1])
    Hb = hilbert_space(b[1])
    Hc = hilbert_space(c[1])
    H = tensor_product((Ha, Hb, Hc))
    @test length(H.factors) == 2
    @test length(H.atoms) == 3
    @test dim(H) == dim(Ha) * dim(Hb) * dim(Hc)
    @test H.atom_ordering == Dict(Ha => 1, Hb => 2, Hc => 3)
    @test_throws ArgumentError tensor_product([Ha, Ha])

    Hab = H.factors[1]
    @test atomic_factors(Hab) == [Ha, Hb]
    @test dim(Hab) == dim(Ha) * dim(Hb)

    @boson boson
    Hboson = hilbert_space(boson, 2)
    H2 = tensor_product([H, Hboson])
    @test dim(H2) == dim(H) * dim(Hboson)
    @test length(H2.factors) == 3

    @test basisstates(H2) == [ProductState((s_f.states..., s_b)) for s_f in basisstates(H), s_b in basisstates(Hboson)]
    for (i, state) in enumerate(basisstates(H))
        @test state_index(state, H) == i
        @test basisstate(i, H) == state
    end

    mapper = state_mapper(H, (Ha, Hb))
    for state in basisstates(H)
        @test only(first(split_state(state, mapper))) == (substate(FockNumber(1), state.states[1]), substate(FockNumber(2), state.states[1]))
    end
    mapper = state_mapper(H, (Ha, Hc))
    for state in basisstates(H)
        @test only(first(split_state(state, mapper))) == (substate(FockNumber(1), state.states[1]), state.states[2])
    end


    # ══════════════════════════════════════════════════════════════
    # state_mapper tests
    # ══════════════════════════════════════════════════════════════
    import FermionicHilbertSpaces: combine_states

    # Test 1: Trivial partition (whole space as one group)
    p_trivial = state_mapper(H, (H,))
    for state in basisstates(H)
        split = only(first(split_state(state, p_trivial)))
        @test only(split) == state
        @test only(first(combine_states(split, p_trivial))) == state
    end

    # Test 2: Binary partition with whole-group passthrough
    p_binary = state_mapper(H, (Hab, Hc))
    for state in basisstates(H)
        substates = only(first(split_state(state, p_binary)))
        @test length(substates) == 2
        @test only(first(combine_states(substates, p_binary))) == state
    end

    # Test 3: Partition that fractures a fermionic group
    # Split the (Ha, Hb) group by grouping [Ha] and [Hb, Hc]
    p_split = state_mapper(H, [Ha, tensor_product((Hb, Hc))])
    for state in basisstates(H)
        substates = only(first(split_state(state, p_split)))
        @test length(substates) == 2
        @test only(first(combine_states(substates, p_split))) == state
    end

    # Test 4: Three-way partition
    p_three = state_mapper(H, [Ha, Hb, Hc])
    for state in basisstates(H)
        substates = only(first(split_state(state, p_three)))
        @test length(substates) == 3
        @test only(first(combine_states(substates, p_three))) == state
    end

    # Test 5: Partition with bosonic space
    p_mixed = state_mapper(H2, [Hab, tensor_product((Hc, Hboson))])
    for state in basisstates(H2)
        substates = only(first(split_state(state, p_mixed)))
        @test only(first(combine_states(substates, p_mixed))) == state
    end

    # Test 6: Validation errors
    @test_throws ArgumentError state_mapper(H, [Ha, Ha])  # Duplicate atom
    @test_throws ArgumentError state_mapper(H, [Ha, Hb, Hc, Hboson])  # Atom not in parent

    # Test splitting to subsystems
    p_fermion_incomplete = state_mapper(H, [Ha])
    @test Set(basisstates(Ha)) == Set(map(basisstates(H)) do state
        only(only(first(split_state(state, p_fermion_incomplete)))) # only(first()) to get the single outcome, only() to get the substate for Ha
    end)
    p_fermion_incomplete = state_mapper(H, [Hab])
    @test Set(basisstates(Hab)) == Set(map(basisstates(H)) do state
        only(only(first(split_state(state, p_fermion_incomplete))))
    end)

    # test state_mapper for FermionicSpace
    p_fermion = state_mapper(Hab, [Ha, Hb])
    for state in basisstates(Hab)
        substates = only(first(split_state(state, p_fermion)))
        @test only(first(combine_states(substates, p_fermion))) == state
    end

    p_fermion_incomplete = state_mapper(Hab, [Ha])
    @test Set(basisstates(Ha)) == Set(map(basisstates(Hab)) do state
        only(first(first(split_state(state, p_fermion_incomplete)))) # only(first()) to get the single outcome, first() to get the substate for Ha
    end)

    # Complement
    @test complementary_subsystem(H, Ha) == tensor_product((Hb, Hc))
    @test complementary_subsystem(H, Hb) == tensor_product([Ha, Hc])
    @test complementary_subsystem(H, Hc) == tensor_product([Ha, Hb])
    @test complementary_subsystem(H, tensor_product([Ha, Hb])) == Hc
    @test complementary_subsystem(H, tensor_product([Hb, Hc])) == Ha
    @test complementary_subsystem(H, tensor_product([Ha, Hc])) == Hb

    @test complementary_subsystem(H2, Hab) == tensor_product([Hc, Hboson])
    @test complementary_subsystem(H2, tensor_product([Hc, Hboson])) == Hab
    @test complementary_subsystem(H2, tensor_product([Ha, Hc])) == tensor_product([Hb, Hboson])

    # Partial trace and embed
    using LinearAlgebra
    m = rand(dim(Ha), dim(Ha))
    @test embed(m, Ha => H) == kron(I(dim(Hb) * dim(Hc)), m)
    @test embed(m, Ha => Hab) == kron(I(dim(Hb)), m)

    @test partial_trace(embed(m, Ha => Hab), Hab => Ha) == m * dim(Hb)
    @test partial_trace(embed(m, Ha => H), H => Ha) == m * dim(Hb) * dim(Hc)

    # Fermion commutation relations
    amat = matrix_representation(a[1], Ha)
    bmat = matrix_representation(b[1], Hb)
    cmat = matrix_representation(c[1], Hc)
    @test embed(amat, Ha => H) * embed(bmat, Ha => H) ≈ -embed(bmat, Ha => H) * embed(amat, Ha => H)
    @test embed(amat, Ha => H) * embed(cmat, Ha => H) ≈ embed(cmat, Ha => H) * embed(amat, Ha => H)
    @test embed(amat, Ha => H) == matrix_representation(a[1], H)
    @test embed(bmat, Hb => H) == matrix_representation(b[1], H)
    @test embed(cmat, Hc => H) == matrix_representation(c[1], H)

    # Constrained spaces and subregion
    Habcons = constrain_space(Hab, NumberConservation(1))
    @test dim(Habcons) == 2
    @test dim(tensor_product((Habcons, Hc))) == dim(Hc) * dim(Habcons)
    @test dim(tensor_product((Habcons, hilbert_space(a[2])))) == dim(hilbert_space(a[2])) * dim(Habcons)

    spaces = (hilbert_space(a[1]), hilbert_space(c[1]), hilbert_space(a[2]), hilbert_space(b[1]), hilbert_space(c[2]), hilbert_space(b[2]))
    Hprod = tensor_product(spaces)
    m = rand(dim(Hprod), dim(Hprod))
    H1 = subregion(spaces[[1, 2, 4, 5, 6]], Hprod)
    H2 = subregion(spaces[[2, 5, 6]], H1)
    H2direct = subregion(spaces[[2, 5, 6]], Hprod)
    @test Set(basisstates(H2direct)) == Set(basisstates(H2))
    @test partial_trace(m, Hprod => H2) ≈ partial_trace(partial_trace(m, Hprod => H1), H1 => H2)


    Hab = tensor_product(spaces[[1, 3, 4, 6]])
    Hc = tensor_product(spaces[[2, 5]])
    Hprod = tensor_product((constrain_space(Hab, NumberConservation(1:2)), constrain_space(Hc, NumberConservation(1))))
    H1 = subregion(spaces[[1, 2, 4, 5, 6]], Hprod)
    H2 = subregion(spaces[[2, 5, 6]], H1)
    H2direct = subregion(spaces[[2, 5, 6]], Hprod)
    @test Set(basisstates(H2direct)) == Set(basisstates(H2))
    m = rand(dim(Hprod), dim(Hprod))
    partial_trace(m, Hprod => H2)
    @test partial_trace(m, Hprod => H2) ≈ partial_trace(partial_trace(m, Hprod => H1), H1 => H2)


    @test matrix_representation(a[1] + 1, Hprod; projection=true) == embed(matrix_representation(a[1], Ha), Ha => Hprod; skipmissing=true) + I
    @test matrix_representation(c[1], Hprod; projection=true) == embed(matrix_representation(c[1], Hc), Hc => Hprod; skipmissing=true)

    @test matrix_representation(a[1]' * b[1], Hprod) == embed(matrix_representation(a[1]' * b[1], Hab), Hab => Hprod; skipmissing=true)

end

@testitem "Fermion ordering" begin
    import FermionicHilbertSpaces: state_mapper
    @fermions f
    H = hilbert_space(f, 1:3)
    @test_throws ArgumentError state_mapper(H, hilbert_space(f, [2, 1]))
    @test_throws ArgumentError state_mapper(H, hilbert_space(f, [3, 1]))
    @test_throws ArgumentError state_mapper(H, hilbert_space(f, [3, 2]))
end

unique_split(::Any) = false
unique_combine(::Any) = false

_find_position(target::AbstractAtomicHilbertSpace, parent::AbstractAtomicHilbertSpace) = atomic_id(target) == atomic_id(parent) ? 1 : 0
_find_position(target::AbstractGroupedHilbertSpace, parent::AbstractGroupedHilbertSpace) = atomic_id(target) == atomic_id(parent) ? 1 : 0


function partial_trace_phase_factor(state1, state2, space::ProductSpace)
    # product of phase factors from each space and substate
    pf = 1
    for (s1, s2, group) in zip(state1.states, state2.states, space.factors)
        pf *= partial_trace_phase_factor(s1, s2, group)
    end
    return pf
end

struct ProductOperator{C,O,S}
    ops::O
    spaces::S
    inds::Vector{Int} # indices of active spaces in full space
    function ProductOperator{C}(ops::O, spaces::S, inds::Vector{Int}) where {C,O,S}
        new{C,O,S}(ops, spaces, inds)
    end
end
function ProductOperator(ops::O, spaces::S, inds::Vector{Int}) where {O,S}
    length(ops) == length(spaces) || throw(ArgumentError("Length of ops and spaces must match"))
    C = promote_type(Iterators.map(mat_eltype, ops)...)
    ProductOperator{C}(ops, spaces, inds)
end
mat_eltype(::ProductOperator{C}) where {C} = C

function has_internal_rep(space::AbstractHilbertSpace, ::Type{T}) where {T}
    state = basisstate(1, space)
    has_internal_rep(state, space, T)
end
function has_internal_rep(state::ProductState, space::ProductSpace, ::Type{T}) where {T}
    all(Iterators.map((s, f) -> has_internal_rep(s, f, T), state.states, factors(space)))
end

function has_internal_rep(state, space, ::Type{T}) where {T}
    int_rep = try
        _internal_rep(state, space, T)
    catch e
        if isa(e, MethodError) && (e.f == internal_rep || e.f == parent)
            return false
        else
            rethrow()
        end
    end
    newstate = try
        _physical_rep(int_rep, space)
    catch e
        if isa(e, MethodError) && (e.f == physical_rep || e.f == parent)
            return false
        else
            rethrow()
        end
    end
    newstate == state || throw(ArgumentError("internal_rep and physical_rep are inconsistent for space $space: got $newstate from physical_rep(internal_rep($state))"))
    return true
end
function precomputation_before_operator_application(ops::ProductOperator, space::AbstractHilbertSpace)
    return map(precomputation_before_operator_application, ops.ops, ops.spaces)
end

internal_rep(state, space::ProductSpace, ::Type{T}) where T<:Integer = T(state_index(state, space))
physical_rep(state::T, space::ProductSpace) where T<:Integer = basisstate(state, space)

function _apply_local_operators(ops::ProductOperator, state::ProductState{B}, space::ProductSpace, precomps) where B
    if !ismissing(fast_path(space))
        T = typeof(fast_path(space))
        internal_reps = map((s, f) -> _internal_rep(s, f, T), state.states, factors(space))
        newrep, amp = _apply_local_operators_fast(ops, internal_reps, space, precomps)
        return ProductState{B}(newrep), amp
    else
        newstate, amp = _apply_local_operators_slow(ops, state, space, precomps)
        return ProductState{B}(newstate), amp
    end
end
function apply_local_operators(op::NCMul, int_state::InternalRep{T}, space::AbstractHilbertSpace, precomp; kwargs...) where T
    state = _physical_rep(int_state, space)
    newstate, amp = apply_local_operators(op, state, space, precomp; kwargs...)
    return _internal_rep(newstate, space, T), amp
end
function _apply_local_operators_fast(ops::ProductOperator{C}, internal_reps::NTuple{N,InternalRep{T}}, space::ProductSpace, precomps) where {C,T,N}
    # amp::C = one(C)
    # newreps::NTuple{N,InternalRep{T}} = internal_reps
    # n::Int = 0
    # map(ops.ops, ops.spaces, precomps, ops.inds) do op, local_space, precomp, n
    #     internal_rep::InternalRep{T} = internal_reps[n]
    #     new_local_rep::InternalRep{T}, local_amp::C = _apply_local_operators(op, internal_rep, local_space, precomp)
    #     amp *= local_amp
    #     newreps = Base.setindex(newreps, new_local_rep, n)
    #     return nothing
    # end
    amp, newreps = foldl(
        zip(ops.inds, ops.ops, ops.spaces, precomps);
        init=(one(C), internal_reps)
    ) do (amp, newreps), (n, op, local_space, precomp)
        ismissing(op) && return (amp, newreps)
        new_local_rep::InternalRep{T}, local_amp::C =
            _apply_local_operators(op, internal_reps[n], local_space, precomp)
        return (amp * local_amp, Base.setindex(newreps, new_local_rep, n))
    end
    return newreps, amp
end
function _apply_local_operators_slow(ops::ProductOperator{C}, state::ProductState{B}, space::ProductSpace, precomps) where {C,B<:Tuple}
    # amp::C = one(C)
    # newstates::B = state.states
    # n::Int = 0
    # println(typeof(ops.ops), typeof(ops.spaces), typeof(precomps), typeof(ops.inds))
    # for (op, local_space, precomp, n) in zip(ops.ops, ops.spaces, precomps, ops.inds)
    #     # n += 1
    #     ismissing(op) && return newstates
    #     subst = state.states[n]
    #     new_local_state, local_amp::C = _apply_local_operators(op, subst, local_space, precomp)
    #     amp *= local_amp
    #     newstates = Base.setindex(newstates, new_local_state, n)
    # end

    amp, newstates = foldl(
        zip(ops.inds, ops.ops, ops.spaces, precomps);
        init=(one(C), state.states)
    ) do (amp, newstates), (n, op, local_space, precomp)
        ismissing(op) && return (amp, newstates)
        new_local_state, local_amp::C =
            _apply_local_operators(op, state.states[n], local_space, precomp)
        return (amp * local_amp, Base.setindex(newstates, new_local_state, n))
    end

    return ProductState{B}(newstates), amp
end

add_tag(H::ProductSpace, tag) = ProductSpace(map(f -> add_tag(f, tag), H.factors), map(a -> add_tag(a, tag), H.atoms))


##
###############################################################################
# ProductSpaceMapper
#
# Maps between a source ProductSpace and a collection of target spaces.
# Built on a single concept: the *piece table*.
#
# A `piece` is a block of the common refinement of the source factor
# partition and the target partition. Each piece has four coordinates:
#
#     (source factor index, slot among that factor's pieces,
#      target index,        slot among that target's groups)
#
# Split and combine are opposite traversals of this table:
#
#   split_state:     split each source factor into its pieces,
#                    then assemble each target from its pieces.
#   combine_states:  take each target's pieces apart (trivial substate
#                    extraction), then assemble each source factor
#                    from its pieces.
#
# Type stability: all routing indices live in the *type domain* via
# `PieceIndex{block,slot}`. Indexing a heterogeneous tuple with these
# compile-time constants infers concretely, so the hot path is type stable
# and allocation free even for heterogeneous states. Construction, by
# contrast, is deliberately plain and dynamic (Dicts, Vectors, sorting) —
# it runs once, so clarity wins there.
###############################################################################

# ─── Compile-time routing index ──────────────────────────────────────────────

"""
    PieceIndex{block, slot}

A routing index carried in the type domain. `getpiece(blocks, ref)` extracts
`blocks[block][slot]` with compile-time-known indices, so heterogeneous
tuples can be indexed without type instability.
"""
struct PieceIndex{block,slot} end
PieceIndex(block::Int, slot::Int) = PieceIndex{block,slot}()
@inline getpiece(blocks, ::PieceIndex{b,s}) where {b,s} = blocks[b][s]

# ─── Leaves: what happens to each source factor ──────────────────────────────

"""A source factor not covered by any target. It is dropped on split and
cannot be reconstructed on combine."""
struct Uncovered end

"""A source factor that goes unchanged into exactly one target slot.
The trivial inner mapper is kept only for phase-factor computation;
split/combine bypass it entirely."""
struct Passthrough{M}
    mapper::M
end

"""A source factor that is split into several pieces (or reshaped into one
non-identical piece) by an inner leaf mapper."""
struct Fractured{M}
    mapper::M
end

# Split one source factor's state into its tuple of pieces.
split_pieces(::Uncovered, state) = ()
split_pieces(::Passthrough, state) = (state,)
split_pieces(leaf::Fractured, state) = only(first(split_state(state, leaf.mapper)))

# Reassemble one source factor's state from its gathered pieces.
combine_pieces(::Uncovered, pieces) =
    throw(ArgumentError("Cannot reconstruct the full state: the targets do not cover every source factor"))
combine_pieces(::Passthrough, pieces) = only(pieces)
combine_pieces(leaf::Fractured, pieces) = only(first(combine_states(pieces, leaf.mapper)))

# The pieces of a target's state are its substates (one piece per group).
substate_pieces(state::ProductState) = state.states
substate_pieces(state) = (state,)


# ─── The mapper ──────────────────────────────────────────────────────────────

"""
    ProductSpaceMapper(source::ProductSpace, targets)

Mapper between `source` and the partition (or sub-collection) `targets`.

Fields:
- `leaves`        : per source factor, an `Uncovered`, `Passthrough` or
                    `Fractured` leaf describing how that factor fractures.
- `factor_routes` : per source factor, one `PieceIndex{target, group_slot}`
                    per piece (in the factor's piece order). Drives combine.
- `target_routes` : per target, one `PieceIndex{factor, piece_slot}` per
                    group (in the target's group order). Drives split.
- `target_spaces` : the targets themselves.

`factor_routes` and `target_routes` are the two directional views of the
same piece table; they are exact inverses of each other by construction.
"""
struct ProductSpaceMapper{L,FR,TR,TS} <: AbstractStateMapper
    leaves::L
    factor_routes::FR
    target_routes::TR
    target_spaces::TS
end

unique_split(::ProductSpaceMapper) = true
unique_combine(::ProductSpaceMapper) = true

state_mapper(source::ProductSpace, targets) = ProductSpaceMapper(source, targets)

# ─── Construction (cold path: clarity over speed) ────────────────────────────

function ProductSpaceMapper(source::ProductSpace, targets)
    targets = Tuple(targets)
    sfactors = factors(source)

    # Atom bookkeeping: which factor owns each atom, and where inside it.
    A = eltype(source.atoms)
    atom_factor = Dict{A,Int}()
    atom_pos = Dict{A,Int}()
    for (ci, f) in enumerate(sfactors)
        for (k, a) in enumerate(atomic_factors(f))
            atom_factor[a] = ci
            atom_pos[a] = k
        end
    end

    # Validate: every target atom exists in the source, no duplicates.
    seen = Set{A}()
    for t in targets, a in atomic_factors(t)
        haskey(atom_factor, a) || throw(ArgumentError("Atom $a not in source space"))
        a ∈ seen && throw(ArgumentError("Atom $a duplicated across targets"))
        push!(seen, a)
    end

    # Build the piece table: one piece per group of each target.
    # Each group must lie inside a single source factor.
    pieces = [
        begin
            owners = unique(atom_factor[a] for a in atomic_factors(g))
            length(owners) == 1 ||
                throw(ArgumentError("Target group $g spans several source factors; this is not supported"))
            (factor=only(owners), target=ti, group_slot=si, space=g,
                pos=minimum(atom_pos[a] for a in atomic_factors(g)))
        end
        for (ti, t) in enumerate(targets) for (si, g) in enumerate(groups(t))
    ]

    # Per-factor view: build leaves and factor routes.
    # A factor's pieces are ordered by where their atoms sit inside the
    # factor, which is also the order the inner leaf mapper produces them in.
    leaves = Any[]
    factor_routes = Any[]
    piece_slot = Dict{Tuple{Int,Int},Int}()  # (target, group_slot) -> slot within factor
    for (ci, f) in enumerate(sfactors)
        ps = sort([p for p in pieces if p.factor == ci], by=p -> p.pos)
        if isempty(ps)
            push!(leaves, Uncovered())
            push!(factor_routes, ())
            continue
        end
        for (k, p) in enumerate(ps)
            piece_slot[(p.target, p.group_slot)] = k
        end
        inner = state_mapper(f, Tuple(p.space for p in ps))
        leaf = (length(ps) == 1 && only(ps).space == f) ? Passthrough(inner) : Fractured(inner)
        push!(leaves, leaf)
        push!(factor_routes, Tuple(PieceIndex(p.target, p.group_slot) for p in ps))
    end

    # Per-target view: the inverse routing, one entry per group slot.
    target_routes = Tuple(
        Tuple(
            begin
                p = only(q for q in pieces if q.target == ti && q.group_slot == si)
                PieceIndex(p.factor, piece_slot[(ti, si)])
            end
            for si in eachindex(groups(t)))
        for (ti, t) in enumerate(targets))

    ProductSpaceMapper(Tuple(leaves), Tuple(factor_routes), target_routes, targets)
end

# ─── Hot path: split / combine ───────────────────────────────────────────────

function split_state(state::ProductState, sp::ProductSpaceMapper)
    # 1. Fracture each source factor into its pieces.
    pieces = map(split_pieces, sp.leaves, state.states)
    # 2. Assemble each target from its pieces.
    outstates = map(sp.target_routes, sp.target_spaces) do route, tspace
        gathered = map(ref -> getpiece(pieces, ref), route)
        only(first(combine_states(gathered, tspace)))
    end
    return (outstates,), (1,)
end

function combine_states(substates::Tuple, sp::ProductSpaceMapper)
    # 1. Take each target's state apart into its pieces (one per group).
    tpieces = map(substate_pieces, substates)
    # 2. Assemble each source factor from its pieces.
    states = map(sp.leaves, sp.factor_routes) do leaf, route
        combine_pieces(leaf, map(ref -> getpiece(tpieces, ref), route))
    end
    return (ProductState(states),), (1,)
end
combine_states(substates, sp::ProductSpaceMapper) = combine_states(Tuple(substates), sp)
combine_states(states::Tuple, ::ProductSpace{ProductState{B}}) where B = (ProductState{B}(B(states)),), (1,)

# ─── Phase factors ───────────────────────────────────────────────────────────

leaf_phase_factor(::Uncovered) = (s1, s2) -> 1
leaf_phase_factor(leaf::Union{Passthrough,Fractured}) = kron_phase_factor(leaf.mapper)

function kron_phase_factor(sp::ProductSpaceMapper)
    length(sp.target_spaces) == 2 ||
        throw(ArgumentError("Phase factors currently only implemented for binary splits"))
    phase_fns = map(leaf_phase_factor, sp.leaves)
    function phase_factor(fullstate1, fullstate2)
        prod(map((pf, s1, s2) -> pf(s1, s2), phase_fns, fullstate1.states, fullstate2.states))
    end
end
function phase_factor_u(sp::ProductSpaceMapper)
    # Build per-leaf phase-factor functions (Uncovered → nothing, others → recurse)
    leaf_pfus = map(sp.leaves) do leaf
        leaf isa Uncovered ? nothing : phase_factor_u(leaf.mapper)
    end

    function phase_factor(state)
        pf = 1
        for (s, leaf, pfu) in zip(state.states, sp.leaves, leaf_pfus)
            leaf isa Uncovered && continue
            pf *= pfu(s)
        end
        return pf
    end
end
