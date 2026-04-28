
combine_into_group(::AbstractAtomicHilbertSpace, atoms) = only(atoms)
combine_into_group(group_id, spaces) = length(spaces) == 1 ? only(spaces) : throw(ArgumentError("Don't know how to combine spaces $spaces with atomic group $group_id into a group"))
struct ProductState{B} <: AbstractBasisState
    states::B
end
substate(n::Int, state::ProductState) = state.states[n]
atomic_factors(state::ProductState) = state.states
Base.:(==)(s1::ProductState, s2::ProductState) = s1.states == s2.states
Base.hash(s::ProductState, h::UInt) = hash(s.states, h)
Base.isless(s1::ProductState, s2::ProductState) = s1.states < s2.states

symbolic_group(h::AbstractAtomicHilbertSpace) = h
# ProductSpaces consists of a list of atomic spaces and factor spaces
struct ProductSpace{B,C,A} <: AbstractProductHilbertSpace{B}
    factors::C
    atoms::Vector{A}
    atom_ordering::Dict{A,Int}
    function ProductSpace(factors::C, atoms::Vector{A}) where {C,A}
        length(factors) == 0 && throw(ArgumentError("Product space must have at least one factor"))
        B = ProductState{Tuple{map(statetype, factors)...}}
        atom_ordering = Dict{A,Int}(a => i for (i, a) in enumerate(atoms))
        new{B,C,A}(factors, atoms, atom_ordering)
    end
end

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
function basisstate(n::Int, H::ProductSpace{B}) where B
    inds = Tuple(CartesianIndices(map(dim, H.factors))[n])
    ProductState(map(basisstate, inds, H.factors))
end
function state_index(state::ProductState, H::ProductSpace)
    cartesian_index = CartesianIndex(Tuple(map(state_index, state.states, H.factors)))
    LinearIndices(map(dim, H.factors))[cartesian_index]
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
##
struct ProductSpaceMapper{CS,TP,CP,TS} <: AbstractStateMapper
    # For each source factor: mapper into per-target pieces, or nothing if uncovered
    factor_mappers::CS

    # For each target j: (source_group_idx, piece_idx) pairs sorted by position in target j.
    # Invariant: gathered[k] corresponds to target_spaces[j].factors[k].
    target_piece_sources::TP

    # For each source factor i: (target_idx, sub_idx_in_target) per piece, in piece-output order
    factor_piece_targets::CP

    target_spaces::TS
end
unique_split(::Any) = false
unique_combine(::Any) = false
unique_split(::ProductSpaceMapper) = true
unique_combine(::ProductSpaceMapper) = true

function state_mapper(source::ProductSpace, targets)
    targets = Tuple(targets)

    atom_to_target = Dict{eltype(source.atoms),Int}()
    for (ti, target) in enumerate(targets)
        for a in atomic_factors(target)
            a ∈ Set(source.atoms) || throw(ArgumentError("Atom $a not in source space"))
            haskey(atom_to_target, a) && throw(ArgumentError("Atom $a duplicated across targets"))
            atom_to_target[a] = ti
        end
    end

    factor_mappers = []
    factor_piece_targets = []  # (ti, sub_idx) per piece, in piece-output order
    pending_pieces = [Tuple{Int,Int,Int}[] for _ in targets]
    for (ci, factor) in enumerate(factors(source))
        catoms = atomic_factors(factor)
        covered_targets = Tuple(unique(atom_to_target[a] for a in catoms if haskey(atom_to_target, a)))

        if isempty(covered_targets)
            push!(factor_mappers, nothing)
            push!(factor_piece_targets, ())
            continue
        end
        atomic_id_set = Set(map(atomic_id, catoms))
        piece_destinations = map(covered_targets) do ti
            (ti, findfirst(factor -> all(atom -> in(atomic_id(atom), atomic_id_set), atomic_factors(factor)), groups(targets[ti])))
        end

        subspaces = [groups(targets[ti])[dest] for (ti, dest) in piece_destinations]
        push!(factor_mappers, state_mapper(factor, subspaces))

        # Store where each piece goes: (ti, sub_idx_in_target)
        push!(factor_piece_targets, piece_destinations)

        # Accumulate for sorting
        for (pi, (ti, tsub)) in enumerate(piece_destinations)
            push!(pending_pieces[ti], (ci, pi, tsub))
        end
    end

    # Sort each target's pieces by their sub-index
    target_piece_sources = Tuple(
        Tuple((p[1], p[2]) for p in sort(pieces, by=p -> p[3]))
        for pieces in pending_pieces
    )

    ProductSpaceMapper(
        Tuple(factor_mappers),
        target_piece_sources,
        Tuple(factor_piece_targets),
        targets,)
end

# ─── helpers ───────────────────────────────────────────────────────────────────

_find_position(target::AbstractAtomicHilbertSpace, parent::AbstractAtomicHilbertSpace) = atomic_id(target) == atomic_id(parent) ? 1 : 0
_find_position(target::AbstractGroupedHilbertSpace, parent::AbstractGroupedHilbertSpace) = atomic_id(target) == atomic_id(parent) ? 1 : 0

# Extract the k-th sub-state (for ProductState) or the state itself (for atomic/group)
extract_substate(state::ProductState, k) = state.states[k]
extract_substate(state, k) = state

# ─── split / combine ───────────────────────────────────────────────────────────

function split_state(state::ProductState, sp::ProductSpaceMapper)
    # Split each source factor into its pieces
    factor_pieces = map(sp.factor_mappers, state.states) do mapper, substate
        isnothing(mapper) ? () :
        only(first(split_state(substate, mapper))) #TODO: handle multiple outcomes from split_state. The use of only(first()) assumes that each factor mapper produces exactly one piece per target
    end
    outstates = map(sp.target_piece_sources, sp.target_spaces) do sources, target_space
        gathered = map(sources) do source
            factor_pieces[source[1]][source[2]]
        end
        only(first(combine_states(gathered, target_space)))
    end
    (outstates,), (1,)
end

function combine_states(substates, sp::ProductSpaceMapper)
    outstate = ProductState(map(sp.factor_mappers, sp.factor_piece_targets) do mapper, piece_destinations
        isnothing(mapper) && error("Cannot reconstruct state: piece_destinations = $piece_destinations, substates = $substates")
        gathered = map(piece_destinations) do dest
            extract_substate(substates[dest[1]], dest[2])
        end
        only(first(combine_states(gathered, mapper))) #TODO: handle multiple outcomes from combine_states
    end)
    (outstate,), (1,)
end


extract_piece(state, ::Int) = state
extract_piece(state::ProductState, idx::Int) = state.states[idx]

combine_states(states::Tuple, ::ProductSpace{ProductState{B}}) where B = (ProductState{B}(B(states)),), (1,)


function kron_phase_factor(state_mapper::ProductSpaceMapper)
    length(state_mapper.target_spaces) == 2 || throw(ArgumentError("Phase factors currently only implemented for binary splits"))
    mappers = state_mapper.factor_mappers
    phase_factor_maps = map(kron_phase_factor, mappers)
    function phase_factor(fullstate1, fullstate2)
        pf = 1
        for (s1, s2, mapper, pfh) in zip(fullstate1.states, fullstate2.states, mappers, phase_factor_maps)
            isnothing(mapper) && continue
            pf *= pfh(s1, s2)
        end
        return pf
    end
end
function partial_trace_phase_factor(state1, state2, space::ProductSpace)
    # product of phase factors from each space and substate
    pf = 1
    for (s1, s2, group) in zip(state1.states, state2.states, space.factors)
        pf *= partial_trace_phase_factor(s1, s2, group)
    end
    return pf
end

struct OperatorSequence{O}
    ops::O
end
function _precomputation_before_operator_application(ops::OperatorSequence, space::ProductSpace)
    map((subops, space) -> _precomputation_before_operator_application(subops, space), ops.ops, factors(space))
end
function _apply_local_operators(ops::OperatorSequence, state::ProductState{B}, space::ProductSpace, precomps) where B<:Tuple
    amp = 1
    spaces = factors(space)
    newstates = state.states
    for n in eachindex(state.states)
        subst = state.states[n]
        space = spaces[n]
        op = ops.ops[n]
        precomp = precomps[n]
        new_local_state, local_amp = _apply_local_operators(op, subst, space, precomp)
        amp *= local_amp
        newstates = Base.setindex(newstates, new_local_state, n)
    end
    return ProductState{B}(newstates), amp
end

add_tag(H::ProductSpace, tag) = ProductSpace(map(f -> add_tag(f, tag), H.factors), map(a -> add_tag(a, tag), H.atoms))