
combine_into_cluster(::AbstractAtomicHilbertSpace, atoms) = only(atoms)
combine_into_cluster(group_id, spaces) = length(spaces) == 1 ? only(spaces) : throw(ArgumentError("Don't know how to combine spaces $spaces with atomic group $group_id into a cluster"))
struct ProductState{B} <: AbstractBasisState
    states::B
end
substate(n::Int, state::ProductState) = state.states[n]
atomic_factors(state::ProductState) = state.states
Base.:(==)(s1::ProductState, s2::ProductState) = s1.states == s2.states
Base.hash(s::ProductState, h::UInt) = hash(s.states, h)

#ClusterSpace consists of atoms. A cluster compresses states and has nontrivial phase factors
atomic_group(h::AbstractAtomicHilbertSpace) = h
# ProductSpaces consists of a list of atomic spaces and clusters
struct ProductSpace{B,C,A} <: AbstractProductHilbertSpace{B}
    clusters::C
    atoms::Vector{A}
    atom_ordering::Dict{A,Int}
    function ProductSpace(clusters::C, atoms::Vector{A}) where {C,A}
        B = ProductState{Tuple{map(statetype, clusters)...}}
        atom_ordering = Dict{A,Int}(a => i for (i, a) in enumerate(atoms))
        new{B,C,A}(clusters, atoms, atom_ordering)
    end
end
function ProductSpace(clusters::C) where {C}
    atoms = mapreduce(atomic_factors, vcat, clusters)
    ProductSpace(clusters, atoms)
end
tensor_product(H::AbstractHilbertSpace) = H
"""
    tensor_product(spaces...)

Construct the composite Hilbert space from the spaces in `spaces`.
"""
tensor_product(spaces...) = tensor_product(spaces)
"""
    tensor_product(spaces)

Construct the composite Hilbert space from the spaces in `spaces`.
"""
function tensor_product(spaces)
    if length(spaces) == 1
        return only(spaces)
    end
    atoms = (Iterators.flatten(Iterators.map(atomic_factors, spaces)))
    _groups = group(atomic_group, atoms)
    groups = (map(g -> map(identity, g), _groups)) #This can convert the groups into vectors of concrete types
    clusters = map(typegroup -> combine_into_cluster(typegroup...), pairs(groups))
    full_space = ProductSpace(Tuple(clusters), collect(atoms))
    if any(isconstrained, spaces)
        splitter = state_splitter(full_space, spaces)
        states = [combine_states(states, splitter) for states in Iterators.product(map(basisstates, spaces)...)]
        if length(full_space.clusters) == 1
            return constrain_space(only(full_space.clusters), vec(map(s -> only(s.states), states)))
        end
        full_space = constrain_space(full_space, states)
    elseif length(clusters) == 1
        return only(clusters)
    end
    return full_space
end
isconstrained(H::AbstractAtomicHilbertSpace) = false
isconstrained(H::ProductSpace) = false
Base.:(==)(H1::ProductSpace, H2::ProductSpace) = H1.clusters == H2.clusters && H1.atoms == H2.atoms
Base.hash(H::ProductSpace, h::UInt) = hash(H.clusters, hash(H.atoms, h))
atomic_factors(H::ProductSpace) = H.atoms
factors(H::ProductSpace) = H.clusters
dim(H::ProductSpace) = prod(dim, H.clusters)

basisstates(H::ProductSpace) = collect(Iterators.map(s -> ProductState(s), Iterators.product(map(basisstates, H.clusters)...)))
function basisstate(n::Int, H::ProductSpace{B}) where B
    inds = Tuple(CartesianIndices(map(dim, H.clusters))[n])
    ProductState(map(basisstate, inds, H.clusters))
end
function state_index(state::ProductState, H::ProductSpace)
    cartesian_index = CartesianIndex(Tuple(map(state_index, state.states, H.clusters)))
    LinearIndices(map(dim, H.clusters))[cartesian_index]
end
_find_atom_position(Hsub::AbstractAtomicHilbertSpace, H::ProductSpace) = get(H.atom_ordering, Hsub, 0)
function _find_position(Hsub, H::ProductSpace)
    pos = findfirst(==(Hsub), H.clusters)
    isnothing(pos) && return 0
    return pos
end


function Base.show(io::IO, H::ProductSpace)
    println(io, "$(dim(H))-dimensional ProductSpace:")
    print(io, "$(length(H.clusters)) clusters")
end

function complementary_subsystem(H::AbstractHilbertSpace, Hsub)
    sub_atoms = Set(atomic_factors(Hsub))

    # Verify Hsub is actually a subsystem of H
    inparent = in(atomic_factors(H))
    for a in atomic_factors(Hsub)
        inparent(a) || throw(ArgumentError("Atom $a in subsystem not found in parent space"))
    end

    # Check for duplicates in Hsub
    length(atomic_factors(Hsub)) == length(sub_atoms) || throw(ArgumentError("Duplicate atoms in subsystem"))

    # Filter atoms preserving original order
    remaining = filter(a -> !(a in sub_atoms), atomic_factors(H))
    isempty(remaining) && throw(ArgumentError("Complementary subsystem is empty"))

    Hcomp = tensor_product(remaining)
    if isconstrained(H)
        #restrict states in Hcomp to those compatible with states in Hsub
        splitter = state_splitter(H, (Hsub, Hcomp))
        states = unique!([compstate for (substate, compstate) in Iterators.map(Base.Fix2(split_state, splitter), basisstates(H)) if !ismissing(state_index(substate, Hsub))])
        return constrain_space(Hcomp, states)
    end
    return Hcomp
    # #  Hsub = tensor_product(Hs)
    # # isconstrained(H) || return Hsub
    # # splitter = state_splitter(H, (Hsub,))
    # # function se(state)
    # #     only(split_state(state, splitter))
    # # end
    # # states = unique!(vec(map(se, basisstates(H))))
    # # ConstrainedSpace(Hsub, states)

    # # Hcomp = subregion(remaining, H)
    # if isconstrained(H)
    #     #restrict states in Hcomp to those compatible with states in Hsub
    #     splitter = state_splitter(H, (Hsub, Hcomp))
    #     states = filter(basisstates(H)) do state
    #         substate, compstate = split_state(state, splitter)
    #         !ismissing(state_index(substate, Hsub)) && !ismissing(state_index(compstate, Hcomp))
    #     end
    #     return ConstrainedSpace(Hcomp, states)
    # end
    # return Hcomp
end


struct AtomicStateSplitter end
function state_splitter(H::AbstractAtomicHilbertSpace, Hs)
    only(Hs) == H || throw(ArgumentError("For atomic subspaces, the only valid partition is the whole space"))
    AtomicStateSplitter()
end
function split_state(state, ::AtomicStateSplitter)
    (state,)
end
function combine_states(states, ::AtomicStateSplitter)
    only(states)
end
kron_phase_factor(::AtomicStateSplitter) = (f1, f2) -> 1


@testitem "ProductSpace" begin
    import FermionicHilbertSpaces: state_index, basisstate, TruncatedBosonicHilbertSpace, state_splitter, ProductState, substate, complementary_subsystem, atomic_factors, split_state, tensor_product
    @fermions a b
    @fermions c
    Ha = hilbert_space(a[1])
    Hb = hilbert_space(b[1])
    Hc = hilbert_space(c[1])
    H = tensor_product((Ha, Hb, Hc))
    @test length(H.clusters) == 2
    @test length(H.atoms) == 3
    @test dim(H) == dim(Ha) * dim(Hb) * dim(Hc)
    @test H.atom_ordering == Dict(Ha => 1, Hb => 2, Hc => 3)
    @test_throws ArgumentError tensor_product([Ha, Ha])

    Hab = H.clusters[1]
    @test atomic_factors(Hab) == [Ha, Hb]
    @test dim(Hab) == dim(Ha) * dim(Hb)

    Hboson = TruncatedBosonicHilbertSpace(2)
    H2 = tensor_product([H, Hboson])
    @test dim(H2) == dim(H) * dim(Hboson)
    @test length(H2.clusters) == 3

    @test basisstates(H2) == [ProductState((s_f.states..., s_b)) for s_f in basisstates(H), s_b in basisstates(Hboson)]
    for (i, state) in enumerate(basisstates(H))
        @test state_index(state, H) == i
        @test basisstate(i, H) == state
    end

    splitter = state_splitter(H, (Ha, Hb))
    for state in basisstates(H)
        @test split_state(state, splitter) == (substate(1, state.states[1]), substate(2, state.states[1]))
    end
    splitter = state_splitter(H, (Ha, Hc))
    for state in basisstates(H)
        @test split_state(state, splitter) == (substate(1, state.states[1]), state.states[2])
    end


    # ══════════════════════════════════════════════════════════════
    # state_splitter tests
    # ══════════════════════════════════════════════════════════════
    import FermionicHilbertSpaces: combine_states

    # Test 1: Trivial partition (whole space as one group)
    p_trivial = state_splitter(H, (H,))
    for state in basisstates(H)
        split = split_state(state, p_trivial)
        @test only(split) == state
        @test combine_states(split, p_trivial) == state
    end

    # Test 2: Binary partition with whole-cluster passthrough
    p_binary = state_splitter(H, (Hab, Hc))
    for state in basisstates(H)
        substates = split_state(state, p_binary)
        @test length(substates) == 2
        @test combine_states(substates, p_binary) == state
    end

    # Test 3: Partition that fractures a fermionic cluster
    # Split the (Ha, Hb) cluster by grouping [Ha] and [Hb, Hc]
    p_split = state_splitter(H, [Ha, ProductSpace([Hb, Hc])])
    for state in basisstates(H)
        substates = split_state(state, p_split)
        @test length(substates) == 2
        @test combine_states(substates, p_split) == state
    end

    # Test 4: Three-way partition
    p_three = state_splitter(H, [Ha, Hb, Hc])
    for state in basisstates(H)
        substates = split_state(state, p_three)
        @test length(substates) == 3
        @test combine_states(substates, p_three) == state
    end

    # Test 5: Partition with bosonic space
    p_mixed = state_splitter(H2, [Hab, ProductSpace([Hc, Hboson])])
    for state in basisstates(H2)
        substates = split_state(state, p_mixed)
        @test combine_states(substates, p_mixed) == state
    end

    # Test 6: Validation errors
    @test_throws ArgumentError state_splitter(H, [Ha, Ha])  # Duplicate atom
    @test_throws ArgumentError state_splitter(H, [Ha, Hb, Hc, Hboson])  # Atom not in parent

    # Test splitting to subsystems
    p_fermion_incomplete = state_splitter(H, [Ha])
    @test Set(basisstates(Ha)) == Set(map(basisstates(H)) do state
        only(split_state(state, p_fermion_incomplete))
    end)
    p_fermion_incomplete = state_splitter(H, [Hab])
    @test Set(basisstates(Hab)) == Set(map(basisstates(H)) do state
        only(split_state(state, p_fermion_incomplete))
    end)

    # test state_splitter for FermionCluster
    p_fermion = state_splitter(Hab, [Ha, Hb])
    for state in basisstates(Hab)
        substates = split_state(state, p_fermion)
        @test combine_states(substates, p_fermion) == state
    end

    p_fermion_incomplete = state_splitter(Hab, [Ha])
    @test Set(basisstates(Ha)) == Set(map(basisstates(Hab)) do state
        only(split_state(state, p_fermion_incomplete))
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
    import FermionicHilbertSpaces: constrain_space
    Habcons = constrain_space(Hab, NumberConservation(1))
    @test dim(Habcons) == 2
    @test dim(tensor_product((Habcons, Hc))) == dim(Hc) * dim(Habcons)
    @test dim(tensor_product((Habcons, hilbert_space(a[2])))) == dim(hilbert_space(a[2])) * dim(Habcons)

    spaces = (hilbert_space(a[1]), hilbert_space(c[1]), hilbert_space(a[2]), hilbert_space(b[1]), hilbert_space(c[2]), hilbert_space(b[2]))
    Hprod = tensor_product(spaces)
    m = rand(dim(Hprod), dim(Hprod))
    H1 = subregion(spaces[[1, 2, 4, 5, 6]], Hprod)
    H2 = subregion(spaces[[2, 5, 6]], H1)
    @test subregion(spaces[[2, 5, 6]], Hprod) == H2
    @test partial_trace(m, Hprod => H2) ≈ partial_trace(partial_trace(m, Hprod => H1), H1 => H2)


    Hab = tensor_product(spaces[[1, 3, 4, 6]])
    Hc = tensor_product(spaces[[2, 5]])
    Hprod = tensor_product((constrain_space(Hab, NumberConservation(1:2)), constrain_space(Hc, NumberConservation(1))))
    H1 = subregion(spaces[[1, 2, 4, 5, 6]], Hprod)
    H2 = subregion(spaces[[2, 5, 6]], H1)
    @test subregion(spaces[[2, 5, 6]], Hprod) == H2
    m = rand(dim(Hprod), dim(Hprod))
    partial_trace(m, Hprod => H2)
    @test partial_trace(m, Hprod => H2) ≈ partial_trace(partial_trace(m, Hprod => H1), H1 => H2)


    @test matrix_representation(a[1] + 1, Hprod; projection=true) == embed(matrix_representation(a[1], Ha), Ha => Hprod; skipmissing=true) + I
    @test matrix_representation(c[1], Hprod; projection=true) == embed(matrix_representation(c[1], Hc), Hc => Hprod; skipmissing=true)

    @test matrix_representation(a[1]' * b[1], Hprod) == embed(matrix_representation(a[1]' * b[1], Hab), Hab => Hprod; skipmissing=true)

end

##
struct ProductSpaceSplitter{CS,TP,CP,TS}
    # For each source cluster: splitter into per-target pieces, or nothing if uncovered
    cluster_splitters::CS

    # For each target j: (source_cluster_idx, piece_idx) pairs sorted by position in target j.
    # Invariant: gathered[k] corresponds to target_spaces[j].clusters[k].
    target_piece_sources::TP

    # For each source cluster i: (target_idx, sub_idx_in_target) per piece, in piece-output order
    cluster_piece_targets::CP

    target_spaces::TS
end

function state_splitter(source::ProductSpace, targets)
    targets = Tuple(targets)

    atom_to_target = Dict{eltype(source.atoms),Int}()
    for (ti, target) in enumerate(targets)
        for a in atomic_factors(target)
            a ∈ Set(source.atoms) || throw(ArgumentError("Atom $a not in source space"))
            haskey(atom_to_target, a) && throw(ArgumentError("Atom $a duplicated across targets"))
            atom_to_target[a] = ti
        end
    end

    cluster_splitters = []
    cluster_piece_targets = []  # (ti, sub_idx) per piece, in piece-output order
    pending_pieces = [Tuple{Int,Int,Int}[] for _ in targets]

    for (ci, cluster) in enumerate(source.clusters)
        catoms = atomic_factors(cluster)
        covered_targets = unique(atom_to_target[a] for a in catoms if haskey(atom_to_target, a))

        if isempty(covered_targets)
            push!(cluster_splitters, nothing)
            push!(cluster_piece_targets, ())
            continue
        end

        subspaces = [cluster_target_subspace(targets[ti], catoms, atom_to_target, ti)
                     for ti in covered_targets]
        push!(cluster_splitters, state_splitter(cluster, subspaces))

        # Store where each piece goes: (ti, sub_idx_in_target)
        piece_destinations = Tuple((ti, cluster_target_sub_idx(targets[ti], catoms, atom_to_target, ti))
                                   for ti in covered_targets)
        push!(cluster_piece_targets, piece_destinations)

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

    ProductSpaceSplitter(
        Tuple(cluster_splitters),
        target_piece_sources,
        Tuple(cluster_piece_targets),
        targets,
    )
end

# ─── helpers ───────────────────────────────────────────────────────────────────

# Sub-space of `target` corresponding to the atoms of catoms that belong to target ti
cluster_target_subspace(target::AbstractAtomicHilbertSpace, catoms, a2t, ti) = target
cluster_target_subspace(target::AbstractClusterHilbertSpace, catoms, a2t, ti) = target
function cluster_target_subspace(target::ProductSpace, catoms, a2t, ti)
    overlap = [a for a in catoms if get(a2t, a, nothing) == ti]
    inoverlap = in(Set(overlap))
    for sub in factors(target)
        atoms = atomic_factors(sub)
        all(inoverlap, atoms) && length(overlap) == length(atoms) && return sub
    end
    throw(ArgumentError("No sub-space with atoms $overlap in target"))
end

# Index of that sub-space within target (1 for atomic/cluster targets)
# cluster_target_sub_idx(::Any, catoms, a2t, ti) = 1
cluster_target_sub_idx(target, catoms, a2t, ti) = _find_position(cluster_target_subspace(target, catoms, a2t, ti), target)
_find_position(target::AbstractAtomicHilbertSpace, parent::AbstractAtomicHilbertSpace) = target == parent ? 1 : 0
_find_position(target::AbstractClusterHilbertSpace, parent::AbstractClusterHilbertSpace) = target == parent ? 1 : 0
# cluster_target_sub_idx(target::AbstractAtomicHilbertSpace, catoms, a2t, ti) = 1
# cluster_target_sub_idx(target::AbstractClusterHilbertSpace, catoms, a2t, ti) = 1
# function cluster_target_sub_idx(target::ProductSpace, catoms, a2t, ti)
#     overlap = [a for a in catoms if get(a2t, a, nothing) == ti]
#     for (ki, sub) in enumerate(factors(target))
#         atomic_factors(sub) == overlap && return ki
#     end
#     throw(ArgumentError("No sub-space with atoms $overlap in target"))
# end

# Extract the k-th sub-state (for ProductState) or the state itself (for atomic/cluster)
extract_substate(state::ProductState, k) = state.states[k]
extract_substate(state, k) = state

# ─── split / combine ───────────────────────────────────────────────────────────

function split_state(state::ProductState, sp::ProductSpaceSplitter)
    # Split each source cluster into its pieces
    cluster_pieces = ntuple(length(sp.cluster_splitters)) do i
        isnothing(sp.cluster_splitters[i]) ? () :
        split_state(state.states[i], sp.cluster_splitters[i])
    end
    # For each target, collect pieces already in target-cluster order (guaranteed by sort above)
    ntuple(length(sp.target_spaces)) do j
        sources = sp.target_piece_sources[j]
        gathered = ntuple(k -> cluster_pieces[sources[k][1]][sources[k][2]], length(sources))
        combine_states(gathered, sp.target_spaces[j])
    end
end

function combine_states(substates::Tuple, sp::ProductSpaceSplitter)
    ProductState(ntuple(length(sp.cluster_splitters)) do i
        isnothing(sp.cluster_splitters[i]) &&
            error("Cannot reconstruct state: cluster $i has no atoms in any target")

        piece_destinations = sp.cluster_piece_targets[i]
        gathered = ntuple(k -> extract_substate(substates[piece_destinations[k][1]],
                piece_destinations[k][2]),
            length(piece_destinations))
        combine_states(gathered, sp.cluster_splitters[i])
    end)
end


extract_piece(state, ::Int) = state
extract_piece(state::ProductState, idx::Int) = state.states[idx]

combine_states(states::Tuple, ::ProductSpace{ProductState{B}}) where B = ProductState{B}(B(states))


function kron_phase_factor(state_splitter::ProductSpaceSplitter)
    length(state_splitter.target_spaces) == 2 || throw(ArgumentError("Phase factors currently only implemented for binary splits"))
    splitters = state_splitter.cluster_splitters
    phase_factor_maps = map(kron_phase_factor, splitters)
    function phase_factor(fullstate1, fullstate2)
        pf = 1
        for (s1, s2, splitter, pfh) in zip(fullstate1.states, fullstate2.states, splitters, phase_factor_maps)
            isnothing(splitter) && continue
            pf *= pfh(s1, s2)
        end
        return pf
    end
end
function partial_trace_phase_factor(state1, state2, space::ProductSpace)
    # product of phase factors from each space and substate
    pf = 1
    for (s1, s2, cluster) in zip(state1.states, state2.states, space.clusters)
        pf *= partial_trace_phase_factor(s1, s2, cluster)
    end
    return pf
end


"""
    subregion(Hs, H::AbstractHilbertSpace)

Return the subsystem of `H` spanned by the factors `Hs`.

This is primarily used for product spaces where the subsystem is specified by a list/tuple of factor spaces.

# Examples
```julia
H1 = hilbert_space(1:1)
H2 = hilbert_space(2:2)
H3 = hilbert_space(3:3)
H = tensor_product((H1, H2, H3))
Hsub = subregion((H1, H3), H)
```
"""
function subregion(Hs, H::AbstractHilbertSpace)
    Hsub = tensor_product(Hs)
    isconstrained(H) || return Hsub
    splitter = state_splitter(H, (Hsub,))
    function se(state)
        only(split_state(state, splitter))
    end
    states = unique!(vec(map(se, basisstates(H))))
    constrain_space(Hsub, states)
end


# function ispartition(Hs, H::AbstractProductHilbertSpace)
#     all_atoms = atomic_factors(H)
#     iters = [map(a -> atom_position(a, H), (atomic_factors(Hsub))) for Hsub in Hs]
#     ispartition(iters, all_atoms)
# end
# function isorderedpartition(Hs, H::AbstractProductHilbertSpace)
#     iters = [atomic_factors(Hsub) for Hsub in Hs]
#     isorderedpartition(iters, H.atom_ordering)
# end
# function isorderedsubsystem(Hsub, H::AbstractProductHilbertSpace)
#     consistent_ordering(Hsub, H) || return false
#     issubsystem(Hsub, H) || return false
#     return true
# end

# function consistent_ordering(subsystem, jw::Dict)::Bool
#     lastpos = 0
#     for label in subsystem
#         haskey(jw, label) || return false
#         newpos = jw[label]
#         newpos > lastpos || return false
#         lastpos = newpos
#     end
#     return true
# end

function isorderedpartition(partition, order)
    n = length(order)
    covered = falses(n)
    for subsystem in partition
        lastpos = 0
        for label in subsystem
            pos = _find_position(label, order)
            pos == 0 && return false
            pos > lastpos || return false
            covered[pos] && return false
            covered[pos] = true
            lastpos = pos
        end
    end
    all(covered) || return false
    return true
end
function isorderedpartition(partition)
    n = sum(length, partition)
    covered = falses(n)
    for subsystem in partition
        lastpos = 0
        for pos in subsystem
            pos > lastpos || return false
            covered[pos] && return false
            covered[pos] = true
            lastpos = pos
        end
    end
    all(covered) || return false
    return true
end
atom_position(atom, H::AbstractProductHilbertSpace) = _find_position(atom, atomic_factors(H))

# function SubStateExtender(Hs, H::AbstractProductHilbertSpace)
#     perm = map(space -> findfirst(==(space), Hs), H.spaces)
#     isperm(perm) || throw(ArgumentError("The spaces in Hs must be a permutation of the spaces in H"))
#     return function extender(states)
#         combine_states(map(p -> states[p], perm), H)
#     end
# end
# function StateExtender(Hs, H::AbstractProductHilbertSpace)
#     # ispartition(Hs, H) || throw(ArgumentError("The spaces in Hs must form a partition of the spaces in H"))
#     # println(Hs)
#     # println(H)
#     # println(map(h -> _find_position(h, H), Hs))
#     if all(h -> _find_position(h, H) > 0, Hs)
#         return SubStateExtender(Hs, H)
#     end
#     ## Check
#     return AtomicStateExtender(Hs, H)
# end
# function AtomicStateExtender(Hs, H::AbstractProductHilbertSpace)
#     atoms = atomic_factors(H)
#     subatoms = Iterators.flatten(map(atomic_factors, Hs))
#     grouped_atoms = map(atomic_factors, factors(H))
#     # find position of each atom in each group in grouped_atoms in subtoms
#     positions = map(grouped_atoms) do group
#         map(group) do atom
#             for (Hpos, Hsub) in enumerate(Hs)
#                 pos = atom_position(atom, Hsub)
#                 if pos > 0
#                     return (Hpos, pos)
#                 end
#             end
#             throw(ArgumentError("Each atomic factor in the full space must belong to one of the subspaces"))
#         end
#     end
#     return function extender(substates)
#         ## Split each substate into its atomic factors, then combine the atomic factors according to the grouping in the full space
#         state_atoms = map(atomic_factors, substates, Hs)
#         combined_atoms = map(positions) do group
#             map(pos -> state_atoms[pos[1]][pos[2]], group)
#         end
#         substates = map(combine_atoms, combined_atoms, factors(H))
#         combine_states(substates, H)
#     end

# end
# function complementary_subsystem1(H::AbstractProductHilbertSpace, Hsub::AbstractHilbertSpace)
#     # check if Hsub matches a single space in H
#     subspace_ind = _find_position(Hsub, H)
#     if subspace_ind > 0
#         other_spaces = deleteat!(collect(factors(H)), subspace_ind)
#         return subregion(other_spaces, H)
#     end
#     # otherwise we need to split into atoms and find which atoms belong to the subspace
#     atoms = collect(atomic_factors(H))
#     subatoms = collect(atomic_factors(Hsub))
#     other_atoms = setdiff(atoms, subatoms)
#     subregion(other_atoms, H)
# end

# StateSplitter(H::ConstrainedSpace, Hs) = StateExtractor(H, Hs)