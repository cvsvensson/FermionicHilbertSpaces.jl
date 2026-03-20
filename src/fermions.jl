
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
combine_states(states, ::AbstractAtomicHilbertSpace) = only(states)
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
operators(H::FermionCluster) = fermions(H)
operators(H::FermionicMode) = fermions(H)

function subregion(Hsub::FermionCluster, H::FermionCluster)
    positions = map(f -> _find_position(f, H), modes(Hsub))
    all(x -> x > 0, positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    issorted(positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    Hsub
end
isconstrained(H::FermionCluster) = false

combine_states(states, H::FermionCluster{F}) where F = F(catenate_fock_states(states, H.modes))
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

embedding_unitary(partition, H::FermionCluster) = embedding_unitary(partition, basisstates(H), H.mode_ordering)
bipartite_embedding_unitary(X, Xbar, H::FermionCluster) = bipartite_embedding_unitary(X, Xbar, basisstates(H), H.mode_ordering)
isorderedpartition(partition, H::FermionCluster) = isorderedpartition(map(atomic_factors, partition), mode_ordering(H))
partial_trace_phase_factor(f1, f2, H::FermionCluster) = phase_factor_f(f1, f2, nbr_of_modes(H))


struct NumberConservation{T,H}
    total::T
    subspaces::H
end
NumberConservation(n) = NumberConservation(n, nothing)
NumberConservation() = NumberConservation(nothing, nothing)
function constrain_space(space::Union{<:FermionCluster,<:FermionicMode}, constraint::NumberConservation{T,H}) where {T,H}
    subspaces = H <: Nothing ? atomic_factors(space) : constraint.subspaces
    total = T <: Nothing ? (0:sum(maximum_particles, subspaces)) : constraint.total
    constraint = unweighted_number_branch_constraint(total, subspaces, atomic_factors(space))
    constrain_space(space, constraint; leaf_processor=CombineFockNumbersProcessor(), sortby=particle_number)
end

struct ParityConservation{H}
    allowed_parities::Vector{Int}
    subspaces::H
end
ParityConservation() = ParityConservation([-1, 1], nothing)
ParityConservation(ps::AbstractVector{Int}) = ParityConservation(Vector{Int}(ps), nothing)
ParityConservation(p::Int) = ParityConservation([p], nothing)
function constrain_space(H::FermionCluster, constraint::ParityConservation)
    possible_numbers = isnothing(constraint.subspaces) ? (0:nbr_of_modes(H)) : (0:sum(nbr_of_modes, constraint.subspaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    constraint = unweighted_number_branch_constraint(allowed_numbers, constraint.subspaces, H.modes)
    constrain_space(H, constraint; leaf_processor=CombineFockNumbersProcessor(), sortby=parity)
end


# combine_atoms(f::Tuple{FockNumber{Bool}}, ::FermionicMode{Int64}) = f[1]

# function split_and_combine_atoms(state::FockNumber, positions)
#     subbits = Iterators.map(i -> _bit(state, i), positions)
#     return focknbr_from_bits(subbits)
# end
