
combine_into_cluster(::AbstractAtomicHilbertSpace, atoms) = only(atoms)
combine_into_cluster(group_id, spaces) = length(spaces) == 1 ? only(spaces) : throw(ArgumentError("Don't know how to combine spaces $spaces with atomic group $group_id into a cluster"))
struct ProductState{B} <: AbstractBasisState
    states::B
end
substate(n::Int, state::ProductState) = state.states[n]
atomic_factors(state::ProductState) = state.states
Base.:(==)(s1::ProductState, s2::ProductState) = s1.states == s2.states
Base.hash(s::ProductState, h::UInt) = hash(s.states, h)
Base.isless(s1::ProductState, s2::ProductState) = s1.states < s2.states
#ClusterSpace consists of atoms. A cluster compresses states and has nontrivial phase factors
symbolic_group(h::AbstractAtomicHilbertSpace) = h
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

isconstrained(H::ProductSpace) = false
Base.:(==)(H1::ProductSpace, H2::ProductSpace) = H1.clusters == H2.clusters && H1.atoms == H2.atoms
Base.hash(H::ProductSpace, h::UInt) = hash(H.clusters, hash(H.atoms, h))
atomic_factors(H::ProductSpace) = H.atoms
factors(H::ProductSpace) = H.clusters
clusters(H::ProductSpace) = H.clusters
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
    if get(io, :compact, false)
        print(io, "ProductSpace($(dim(H))-dim, $(length(H.clusters)) clusters)")
    else
        print(io, "$(dim(H))-dimensional ProductSpace: ")
        dims = map(dim, H.clusters)
        println(io, "(", join(dims, "x"), ")")
        for (i, c) in enumerate(H.clusters)
            i > 1 && print(io, " ⊗ ")
            show(IOContext(io, :compact => true), c)
        end
    end
end
maximum_particles(H::ProductSpace) = sum(maximum_particles, factors(H))
particle_number(state::ProductState) = sum(particle_number, atomic_factors(state))

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
    # isempty(remaining) && throw(ArgumentError("Complementary subsystem is empty"))
    isempty(remaining) && return nothing

    Hcomp = tensor_product(remaining)
    if isconstrained(H)
        #restrict states in Hcomp to those compatible with states in Hsub
        splitter = state_splitter(H, (Hsub, Hcomp))
        states = _find_compatible_complementary_states(H, Hsub, splitter)
        return constrain_space(Hcomp, states)
    end
    return Hcomp
end

function _find_compatible_complementary_states(H, Hsub, splitter)
    split = Base.Fix2(split_state, splitter)
    split_state_iterator = if unique_split(splitter)
        Iterators.map(only ∘ first ∘ split, basisstates(H))
    else
        Iterators.flatten(Iterators.map(first ∘ split, basisstates(H)))
    end
    unique(fbar for (fsub, fbar) in split_state_iterator if !ismissing(state_index(fsub, Hsub)))
end
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
    import FermionicHilbertSpaces: state_index, basisstate, state_splitter, ProductState, substate, complementary_subsystem, atomic_factors, split_state, tensor_product
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

    @boson boson
    Hboson = hilbert_space(boson, 2)
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
        @test only(first(split_state(state, splitter))) == (substate(1, state.states[1]), substate(2, state.states[1]))
    end
    splitter = state_splitter(H, (Ha, Hc))
    for state in basisstates(H)
        @test only(first(split_state(state, splitter))) == (substate(1, state.states[1]), state.states[2])
    end


    # ══════════════════════════════════════════════════════════════
    # state_splitter tests
    # ══════════════════════════════════════════════════════════════
    import FermionicHilbertSpaces: combine_states

    # Test 1: Trivial partition (whole space as one group)
    p_trivial = state_splitter(H, (H,))
    for state in basisstates(H)
        split = only(first(split_state(state, p_trivial)))
        @test only(split) == state
        @test only(first(combine_states(split, p_trivial))) == state
    end

    # Test 2: Binary partition with whole-cluster passthrough
    p_binary = state_splitter(H, (Hab, Hc))
    for state in basisstates(H)
        substates = only(first(split_state(state, p_binary)))
        @test length(substates) == 2
        @test only(first(combine_states(substates, p_binary))) == state
    end

    # Test 3: Partition that fractures a fermionic cluster
    # Split the (Ha, Hb) cluster by grouping [Ha] and [Hb, Hc]
    p_split = state_splitter(H, [Ha, tensor_product((Hb, Hc))])
    for state in basisstates(H)
        substates = only(first(split_state(state, p_split)))
        @test length(substates) == 2
        @test only(first(combine_states(substates, p_split))) == state
    end

    # Test 4: Three-way partition
    p_three = state_splitter(H, [Ha, Hb, Hc])
    for state in basisstates(H)
        substates = only(first(split_state(state, p_three)))
        @test length(substates) == 3
        @test only(first(combine_states(substates, p_three))) == state
    end

    # Test 5: Partition with bosonic space
    p_mixed = state_splitter(H2, [Hab, tensor_product((Hc, Hboson))])
    for state in basisstates(H2)
        substates = only(first(split_state(state, p_mixed)))
        @test only(first(combine_states(substates, p_mixed))) == state
    end

    # Test 6: Validation errors
    @test_throws ArgumentError state_splitter(H, [Ha, Ha])  # Duplicate atom
    @test_throws ArgumentError state_splitter(H, [Ha, Hb, Hc, Hboson])  # Atom not in parent

    # Test splitting to subsystems
    p_fermion_incomplete = state_splitter(H, [Ha])
    @test Set(basisstates(Ha)) == Set(map(basisstates(H)) do state
        only(only(first(split_state(state, p_fermion_incomplete)))) # only(first()) to get the single outcome, only() to get the substate for Ha
    end)
    p_fermion_incomplete = state_splitter(H, [Hab])
    @test Set(basisstates(Hab)) == Set(map(basisstates(H)) do state
        only(only(first(split_state(state, p_fermion_incomplete))))
    end)

    # test state_splitter for FermionCluster
    p_fermion = state_splitter(Hab, [Ha, Hb])
    for state in basisstates(Hab)
        substates = only(first(split_state(state, p_fermion)))
        @test only(first(combine_states(substates, p_fermion))) == state
    end

    p_fermion_incomplete = state_splitter(Hab, [Ha])
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

@testitem "Fermion ordering" begin
    import FermionicHilbertSpaces: state_splitter
    @fermions f
    H = hilbert_space(f, 1:3)
    @test_throws ArgumentError state_splitter(H, hilbert_space(f, [2, 1]))
    @test_throws ArgumentError state_splitter(H, hilbert_space(f, [3, 1]))
    @test_throws ArgumentError state_splitter(H, hilbert_space(f, [3, 2]))
end
##
struct ProductSpaceSplitter{CS,TP,CP,TS} <: AbstractStateSplitter
    # For each source cluster: splitter into per-target pieces, or nothing if uncovered
    cluster_splitters::CS

    # For each target j: (source_cluster_idx, piece_idx) pairs sorted by position in target j.
    # Invariant: gathered[k] corresponds to target_spaces[j].clusters[k].
    target_piece_sources::TP

    # For each source cluster i: (target_idx, sub_idx_in_target) per piece, in piece-output order
    cluster_piece_targets::CP

    target_spaces::TS
end
unique_split(::Any) = false
unique_combine(::Any) = false
unique_split(::ProductSpaceSplitter) = true
unique_combine(::ProductSpaceSplitter) = true

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
        covered_targets = Tuple(unique(atom_to_target[a] for a in catoms if haskey(atom_to_target, a)))

        if isempty(covered_targets)
            push!(cluster_splitters, nothing)
            push!(cluster_piece_targets, ())
            continue
        end

        piece_destinations = map(covered_targets) do ti
            (ti, findfirst(cluster -> all(in(catoms), atomic_factors(cluster)), clusters(targets[ti])))
        end
        subspaces = [clusters(targets[ti])[dest] for (ti, dest) in piece_destinations]
        push!(cluster_splitters, state_splitter(cluster, subspaces))

        # Store where each piece goes: (ti, sub_idx_in_target)
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
        targets,)
end

# ─── helpers ───────────────────────────────────────────────────────────────────

_find_position(target::AbstractAtomicHilbertSpace, parent::AbstractAtomicHilbertSpace) = target == parent ? 1 : 0
_find_position(target::AbstractClusterHilbertSpace, parent::AbstractClusterHilbertSpace) = target == parent ? 1 : 0

# Extract the k-th sub-state (for ProductState) or the state itself (for atomic/cluster)
extract_substate(state::ProductState, k) = state.states[k]
extract_substate(state, k) = state

# ─── split / combine ───────────────────────────────────────────────────────────

function split_state(state::ProductState, sp::ProductSpaceSplitter)
    # Split each source cluster into its pieces
    cluster_pieces = map(sp.cluster_splitters, state.states) do splitter, substate
        isnothing(splitter) ? () :
        only(first(split_state(substate, splitter))) #TODO: handle multiple outcomes from split_state. The use of only(first()) assumes that each cluster splitter produces exactly one piece per target
    end
    outstates = map(sp.target_piece_sources, sp.target_spaces) do sources, target_space
        gathered = map(sources) do source
            cluster_pieces[source[1]][source[2]]
        end
        only(first(combine_states(gathered, target_space)))
    end
    (outstates,), (1,)
end

function combine_states(substates, sp::ProductSpaceSplitter)
    outstate = ProductState(map(sp.cluster_splitters, sp.cluster_piece_targets) do splitter, piece_destinations
        isnothing(splitter) && error("Cannot reconstruct state: cluster $i has no atoms in any target")
        gathered = map(piece_destinations) do dest
            extract_substate(substates[dest[1]], dest[2])
        end
        only(first(combine_states(gathered, splitter))) #TODO: handle multiple outcomes from combine_states
    end)
    (outstate,), (1,)
end


extract_piece(state, ::Int) = state
extract_piece(state::ProductState, idx::Int) = state.states[idx]

combine_states(states::Tuple, ::ProductSpace{ProductState{B}}) where B = ((ProductState{B}(B(states)), 1),)


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
    issubsystem(Hsub, H) || throw(ArgumentError("The spaces in Hs must be a subsystem of H"))
    splitter = state_splitter(H, (Hsub,))
    states = _find_subregion_states(H, splitter)
    constrain_space(Hsub, states)
end
function _find_subregion_states(H, splitter)
    split = Base.Fix2(split_state, splitter)
    split_state_iterator = if unique_split(splitter)
        Iterators.map(only ∘ only ∘ first ∘ split, basisstates(H))
    else
        Iterators.map(only, Iterators.flatten(Iterators.map(first ∘ split, basisstates(H))))
    end
    unique(split_state_iterator)
end


_find_position(n, v::AbstractVector) = (pos = findfirst(==(n), v); isnothing(pos) ? 0 : pos)

_find_atom_position(atom, H::AbstractClusterHilbertSpace) = _find_position(atom, H)
_find_atom_position(atom, H::AbstractHilbertSpace) = _find_position(atom, atomic_factors(H))

function isorderedsubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace)
    positions = [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)]
    all(pos -> pos > 0, positions) || return false
    issorted(positions) || return false
    return true
end
function isorderedpartition(Hsubs, H::AbstractHilbertSpace)
    positions = map(Hsub -> [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)], Hsubs)
    isorderedpartition(positions, length(atomic_factors(H)))
end
function issubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace)
    positions = [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)]
    all(pos -> pos > 0, positions)
end
function ispartition(partition, H::AbstractHilbertSpace)
    partition_inds = [_find_atom_position(atom, H) for part in partition for atom in atomic_factors(part)]
    ispartition(partition_inds, length(atomic_factors(H)))
end
function ispartition(partition, N::Int)
    covered = falses(N)
    for subsystem in partition
        for pos in subsystem
            pos == 0 && return false
            covered[pos] && return false
            covered[pos] = true
        end
    end
    return all(covered)
end
function ispartition(partition, labels)
    n = length(labels)
    covered = falses(n)
    for subsystem in partition
        for label in subsystem
            pos = _find_position(label, labels)
            pos == 0 && return false
            covered[pos] && return false
            covered[pos] = true
        end
    end
    return all(covered)
end

@testitem "Partition and ordered partition checks" begin
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    order = 1:3
    ispart = Base.Fix2(ispartition, order)
    @test ispart([[1], [2], [3]])
    @test !ispart([[1], [2]])
    @test !ispart([[1, 1, 1]])
    @test !ispart([[1], [1], [2]])
    @test ispart([[1], [2, 3]])
    @test !ispart([[1], [2, 3, 4]])
    @test ispart([[1, 2, 3]])
    @test !ispart([[1, 2]])
    @test ispart([[2], [1], [3]])
    @test ispart([[2], [3], [1]])
    @test ispart([[1, 3], [2]])
    @test ispart([[3, 1], [2]])
    @test !ispart([[3, 1], [2, 4]])
    @test ispart([[2], [1, 3]])
    @test !ispart([[2], [2, 3]])
    @test ispart([[], [1, 2, 3]])
    @test !ispart([[1], [1, 2, 3]])

    ## same for ispartvec
    ispartvec = Base.Fix2(ispartition, order)
    @test ispartvec([[1], [2], [3]])
    @test !ispartvec([[1], [2]])
    @test !ispartvec([[1, 1, 1]])
    @test !ispartvec([[1], [1], [2]])
    @test ispartvec([[1], [2, 3]])
    @test !ispartvec([[1], [2, 3, 4]])
    @test ispartvec([[1, 2, 3]])
    @test !ispartvec([[1, 2]])
    @test ispartvec([[2], [1], [3]])
    @test ispartvec([[2], [3], [1]])
    @test ispartvec([[1, 3], [2]])
    @test ispartvec([[3, 1], [2]])
    @test !ispartvec([[3, 1], [2, 4]])
    @test ispartvec([[2], [1, 3]])
    @test !ispartvec([[2], [2, 3]])
    @test ispartvec([[], [1, 2, 3]])
    @test !ispartvec([[1], [1, 2, 3]])

    ## Ordered partition
    isorderedpart = Base.Fix2(isorderedpartition, order)

    @test isorderedpart([[1], [2], [3]])
    @test isorderedpart([[1], [2, 3]])
    @test isorderedpart([[1, 2, 3]])
    @test isorderedpart([[2], [1], [3]])
    @test isorderedpart([[2], [3], [1]])
    @test isorderedpart([[1, 3], [2]])
    @test !isorderedpart([[3, 1], [2]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[3, 1], [2, 4]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[2], [3, 1]])
    @test !isorderedpart([[1], [3, 2]])
    @test !isorderedpart([[1], [3, 1]])
    @test !isorderedpart([[3], [2, 1]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[2], [2, 3]])
    @test isorderedpart([[], [1, 2, 3]])
    @test !isorderedpart([[1], [1, 2, 3]])
end

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
function isorderedpartition(partition, N::Int)
    covered = falses(N)
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

function _precomputation_before_operator_application(ops::Vector, space::ProductSpace)
    map((subops, space) -> _precomputation_before_operator_application(subops, space), ops, factors(space))
end
function apply_local_operators(ops::Vector{<:NCMul}, state::ProductState{B}, space::ProductSpace, precomps) where B
    amp = 1
    spaces = clusters(space)
    newstates = map(state.states, spaces, ops, precomps) do subst, space, op, precomp
        new_local_states, local_amps = apply_local_operators(op, subst, space, precomp) #TODO: add support for multiple terms here
        amp *= only(local_amps)
        only(new_local_states)
    end
    newstate = ProductState{B}(Tuple(newstates))
    return (newstate,), (amp,)
end
function operator_indices_and_amplitudes_generic!((outinds, ininds, amps), ops::Vector{<:NCMul}, space::AbstractHilbertSpace; projection=false)
    # This is for productspaces, where ops is a list of operators applying to each factor space
    coeff = prod(op.coeff for op in ops)
    precomp = _precomputation_before_operator_application(ops, space)
    for (n, state) in enumerate(basisstates(space))
        newstates, newamps = apply_local_operators(ops, state, space, precomp)
        for (newstate, amp) in zip(newstates, newamps)
            if !iszero(amp)
                outind = state_index(newstate, space)
                if !projection || !ismissing(outind)
                    push!(outinds, outind)
                    push!(amps, amp * coeff)
                    push!(ininds, n)
                end
            end
        end
    end
    return (outinds, ininds, amps)
end
