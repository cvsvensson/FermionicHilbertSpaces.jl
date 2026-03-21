
struct FermionicGroup
    id::UInt64
end
Base.hash(x::FermionicGroup, h::UInt) = hash(x.id, h)
Base.:(==)(a::FermionicGroup, b::FermionicGroup) = a.id == b.id
atomic_group(g::FermionicGroup) = g
Base.isless(g1::FermionicGroup, g2::FermionicGroup) = g1.id < g2.id
struct FermionicMode{L,S} <: AbstractAtomicHilbertSpace{FockNumber{Bool}}
    label::L
    symbolic_basis::S
end
atomic_group(h::FermionicMode) = fermionic_group(h)
fermionic_group(h::FermionicMode) = fermionic_group(h.symbolic_basis)
combine_states(states, ::AbstractAtomicHilbertSpace) = ((only(states), 1),)
modes(H::FermionicMode) = (H.symbolic_basis[H.label],)
Base.:(==)(m1::FermionicMode, m2::FermionicMode) = m1.label == m2.label && m1.symbolic_basis == m2.symbolic_basis
Base.hash(m::FermionicMode, h::UInt) = hash(m.label, hash(m.symbolic_basis, h))
function Base.show(io::IO, c::FermionicMode)
    get(io, :compact, false) || return print(io, "FermionicMode(", c.symbolic_basis.name, "[", c.label, "])")
    print(io, c.symbolic_basis.name, "[", c.label, "]")
end
combine_into_cluster(group::FermionicGroup, fermions) = FermionCluster(fermions, group)
function basisstate(n::Int, H::FermionicMode)
    n == 1 && return FockNumber(false)
    n == 2 && return FockNumber(true)
    throw(ArgumentError("Invalid state index $n for FermionicMode"))
end
basisstates(::FermionicMode) = (FockNumber(false), FockNumber(true))
maximum_particles(::FermionicMode) = 1
state_index(s::FockNumber{Bool}, H::FermionicMode) = s.f + 1
function state_index(s::FockNumber, H::FermionicMode)
    iszero(s.f) && return 1
    isone(s.f) && return 2
    throw(ArgumentError("Invalid state $s for FermionicMode"))
end
atomic_factors(H::FermionicMode) = [H,]
atom_position(atom, H::FermionicMode) = atom == H ? 1 : 0

label(H::FermionicMode) = H.label
fermionic_hilbert_space(label) = FermionicMode(label)
nbr_of_modes(H::FermionicMode) = 1
dim(H::FermionicMode) = 2

struct FermionCluster{F,L} <: AbstractFermionicClusterHilbertSpace{F}
    modes::Vector{L}
    mode_ordering::OrderedDict{L,Int}
    group::FermionicGroup
    function FermionCluster(modes::AbstractVector{L}, group::FermionicGroup, ::Type{F}=FockNumber{default_fock_representation(length(modes))}) where F where L
        mode_ordering = OrderedDict{L,Int}(m => i for (i, m) in enumerate(modes))
        length(mode_ordering) == length(modes) || throw(ArgumentError("Duplicate modes in fermionic group"))
        new{F,L}(modes, mode_ordering, group)
    end
end
Base.:(==)(c1::FermionCluster, c2::FermionCluster) = c1.modes == c2.modes && c1.group == c2.group
Base.hash(c::FermionCluster, h::UInt) = hash(c.modes, hash(c.group, h))
basisstates(H::FermionCluster{F}) where F = Iterators.map(F ∘ FockNumber, UnitRange{UInt64}(0, dim(H) - 1))
basisstate(ind, ::FermionCluster{F}) where F = (F ∘ FockNumber)(ind - 1)
state_index(state::FockNumber, ::FermionCluster) = state.f + 1
dim(H::FermionCluster) = 2^nbr_of_modes(H)
atomic_factors(H::FermionCluster) = H.modes
nbr_of_modes(H::FermionCluster) = length(H.modes)
atomic_group(H::FermionCluster) = H.group
mode_ordering(H::FermionCluster) = H.mode_ordering
modes(H::FermionCluster) = H.modes
_find_position(f::FermionicMode, H::FermionCluster) = get(H.mode_ordering, f, 0)
_find_position(f::FermionicMode, H::FermionicMode) = f == H ? 1 : 0
_find_position(H::AbstractHilbertSpace, ordering::AbstractDict) = get(ordering, H, 0)
operators(H::FermionCluster) = fermions(H)
operators(H::FermionicMode) = fermions(H)

function subregion(Hsub::FermionCluster, H::FermionCluster)
    positions = map(f -> _find_position(f, H), modes(Hsub))
    all(x -> x > 0, positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    issorted(positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    Hsub
end
isconstrained(H::FermionCluster) = false

combine_states(states, H::FermionCluster{F}) where F = ((F(catenate_fock_states(states, H.modes)), 1),)
state_splitter(H::FermionCluster, Hs::AbstractHilbertSpace) = state_splitter(H, (Hs,))
function state_splitter(H::FermionCluster, Hs)
    fermionpositions = [[_find_position(atom, H) for atom in atomic_factors(cluster)] for cluster in Hs]
    all(x -> x > 0, Iterators.flatten(fermionpositions)) || throw(ArgumentError("All subspaces must be part of the cluster"))
    FockMapper(Tuple(fermionpositions))
end

function Base.show(io::IO, c::FermionCluster)
    print(io, "FermionCluster(")
    print(IOContext(io, :compact => true), c.modes)
    print(io, ")")
end

function embedding_unitary(partition, states, H::FermionCluster)
    atoms = atomic_factors(H)
    positions = [[_find_position(atom, atoms) for atom in atomic_factors(cluster)] for cluster in partition]
    embedding_unitary(positions, states)
end
function bipartite_embedding_unitary(X, Xbar, states, H::FermionCluster)
    atoms = atomic_factors(H)
    Xpos = [_find_position(atom, atoms) for atom in atomic_factors(X)]
    Xbarpos = [_find_position(atom, atoms) for atom in atomic_factors(Xbar)]
    bipartite_embedding_unitary(Xpos, Xbarpos, states)
end
embedding_unitary(partition, H::FermionCluster) = embedding_unitary(partition, basisstates(H), H)
bipartite_embedding_unitary(X, Xbar, H::FermionCluster) = bipartite_embedding_unitary(X, Xbar, basisstates(H), H)
# isorderedpartition(partition, H::FermionCluster) = isorderedpartition(map(atomic_factors, partition), mode_ordering(H))
partial_trace_phase_factor(f1, f2, H::FermionCluster) = phase_factor_f(f1, f2, nbr_of_modes(H))

struct NumberConservation{T,H} <: AbstractConstraint
    total::T
    subspaces::H
end
NumberConservation(n) = NumberConservation(n, nothing)
NumberConservation() = NumberConservation(nothing, nothing)

struct ParityConservation{H} <: AbstractConstraint
    allowed_parities::Vector{Int}
    subspaces::H
end
ParityConservation() = ParityConservation([-1, 1], nothing)
ParityConservation(ps::AbstractVector{Int}) = ParityConservation(Vector{Int}(ps), nothing)
ParityConservation(p::Int) = ParityConservation([p], nothing)

function instantiate(constraint::ParityConservation, H::Union{<:FermionCluster,<:FermionicMode})
    possible_numbers = isnothing(constraint.subspaces) ? (0:nbr_of_modes(H)) : (0:sum(nbr_of_modes, constraint.subspaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    constraint = unweighted_number_branch_constraint(allowed_numbers, constraint.subspaces, atomic_factors(H))
end

function instantiate(constraint::NumberConservation{T,H}, space::Union{<:FermionCluster,<:FermionicMode}) where {T,H}
    subspaces = H <: Nothing ? atomic_factors(space) : constraint.subspaces
    total = T <: Nothing ? (0:sum(maximum_particles, subspaces)) : constraint.total
    constraint = unweighted_number_branch_constraint(total, subspaces, atomic_factors(space))
end
sectors(::AbstractConstraint) = nothing
has_sectors(N::NumberConservation) = true
has_sectors(P::ParityConservation) = true
has_sectors(c::ProductConstraint) = any(has_sectors, c.constraints)

function constrain_space(space, constraint::AbstractConstraint)
    # constrained = constrain_space(space, new_constraint; leaf_processor=default_processor(space, constraint, new_constraint), sortby=default_sorter(space, constraint, new_constraint))
    sortby = default_sorter(space, constraint)
    leaf_processor = default_processor(space, constraint)
    # new_constraint = instantiate(constraint, space)
    states = generate_states(space, instantiate(constraint, space); leaf_processor)
    isnothing(sortby) || sort!(states, by=sortby)
    has_sectors(constraint) || return ConstrainedSpace(space, states)
    block_space(space, states, constraint)
end
default_processor(space::Union{<:FermionCluster,<:FermionicMode}, _) = CombineFockNumbersProcessor()
default_processor(space, _, _) = nothing
default_sorter(space, constraint) = nothing
default_sorter(space::Union{<:FermionCluster,<:FermionicMode}, constraint::ParityConservation) = f -> (parity(f), f)
default_sorter(space::Union{<:FermionCluster,<:FermionicMode}, constraint::NumberConservation) = f -> (particle_number(f), f)

sector(f::FockNumber, qn::ParityConservation) = parity(f)
sector(f::FockNumber, qn::NumberConservation) = particle_number(f)
sector(state, c::ProductConstraint) = map(q -> sector(state, q), c.constraints)
@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = NumberConservation() * ParityConservation()
    @fermions f
    H = hilbert_space(f, labels, qn)
    @test keys(H.symmetry.qntofockstates).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(number_conservation(label -> label == l) for l in labels)
    @test all(length.(hilbert_space(labels, qn).symmetry.qntofockstates) .== 1)
end


@testitem "IndexConservation" begin
    import FermionicHilbertSpaces: number_conservation
    labels = 1:4
    qn = number_conservation(==(1))
    qn2 = number_conservation(label -> label in 1:1)
    H = hilbert_space(labels, qn)
    H2 = hilbert_space(labels, qn2)
    @test H == H2

    spatial_labels = 1:1
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = number_conservation(label -> label[2] == :↑) * number_conservation(label -> label[2] == :↓)
    H = hilbert_space(all_labels, qn)
    @test all(length.(H.symmetry.qntofockstates) .== 1)

    spatial_labels = 1:2
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = number_conservation(1, label -> label[2] == :↑)
    H = hilbert_space(all_labels, qn)
    @test length(basisstates(H)) == 2^3
end
