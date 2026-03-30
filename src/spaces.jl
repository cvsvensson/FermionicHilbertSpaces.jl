## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractAtomicHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractProductHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractClusterHilbertSpace{B} <: AbstractProductHilbertSpace{B} end

"""
	basisstates(H)

Return an iterable of basis states for the Hilbert space `H`, in the order used by
matrix representations and indexing utilities.
"""
basisstates

"""
	hilbert_space(args...)

Construct a Hilbert space from symbolic degrees of freedom and labels, optionally
with additional arguments such as constraints.
"""
hilbert_space

factors(H::AbstractAtomicHilbertSpace) = (H,)
clusters(H::AbstractAtomicHilbertSpace) = (H,)
atomic_factors(H::AbstractAtomicHilbertSpace) = (H,)
factors(H::AbstractClusterHilbertSpace) = atomic_factors(H)
clusters(H::AbstractClusterHilbertSpace) = (H,)
atomic_substate(n, f, ::AbstractClusterHilbertSpace) = substate(n, f)
isconstrained(H::AbstractAtomicHilbertSpace) = false

partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1


abstract type AbstractStateMapper end
"""
    state_mapper(H, Hs)

Create a mapper object for decomposing states in `H` into the target subsystems `Hs`.
The returned object must subtype `AbstractStateMapper`.
"""
state_mapper(H, Hs) = throw(MethodError(state_mapper, (H, Hs)))

"""
    split_state(state, mapper)

Split a state using `mapper`.

Contract: returns a tuple with one entry per target subsystem. Each entry is a weighted
collection `((substate, weight), ...)` for that target.
"""
split_state(state, mapper::AbstractStateMapper) = throw(MethodError(split_state, (state, mapper)))

"""
    combine_states(substates, mapper)

Combine subsystem states using `mapper`.

Contract: returns a states and weights `states, weights`.
"""
combine_states(substates, mapper::AbstractStateMapper) = throw(MethodError(combine_states, (substates, mapper)))

"""
    kron_phase_factor(mapper)

Return phase-factor function associated with `mapper` for fermionic tensor products.
"""
kron_phase_factor(mapper::AbstractStateMapper) = throw(MethodError(kron_phase_factor, (mapper,)))

struct AtomicStateMapper <: AbstractStateMapper end
function state_mapper(H::AbstractAtomicHilbertSpace, Hs)
    only(Hs) == H || throw(ArgumentError("For atomic subspaces, the only valid partition is the whole space"))
    AtomicStateMapper()
end
function split_state(state, ::AtomicStateMapper)
    # the return state here is a tuple of tuples because each state is a state in the tuple of subsystem and complement, but the complement is empty
    ((state,),), (1,)
end
function combine_states(states, ::AtomicStateMapper)
    (only(states),), (1,)
end
kron_phase_factor(::AtomicStateMapper) = (f1, f2) -> 1

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
    Hs_items = Hs isa Tuple || Hs isa AbstractVector ? Hs : (Hs,)
    input_ids = map(atomic_id, Iterators.flatten(Iterators.map(atomic_factors, Hs_items)))
    isempty(input_ids) && throw(ArgumentError("Hs must contain at least one space or symbolic operator"))
    Hatoms = collect(atomic_factors(H))
    Hatom_ids = map(atomic_id, Hatoms)
    positions = map(id -> findfirst(==(id), Hatom_ids), input_ids)
    all(!isnothing, positions) || throw(ArgumentError("The spaces/operators in Hs must match atomic factors in H, but the following were not found: $(input_ids[findall(isnothing, positions)])."))
    length(unique(positions)) == length(positions) || throw(ArgumentError("Hs contains duplicate atomic factors"))

    Hsub = tensor_product(Hatoms[positions])
    issubsystem(Hsub, H) || throw(ArgumentError("The spaces in Hs must be a subsystem of H"))
    mapper = state_mapper(H, (Hsub,)) #TODO: this is before the next line, only because it checks that the fermions are ordered correctly. We should probably split that check out into a separate function.
    !isconstrained(H) && return Hsub
    states = _find_subregion_states(H, mapper)
    ConstrainedSpace(Hsub, states)
end

@testitem "Subregion: matching spaces with symbols" begin
    @fermions f
    @bosons b
    Hf = hilbert_space(f, 1:3)
    Hb = hilbert_space(b, 1:3, 2)
    H = tensor_product(Hf, Hb)
    Hfsub = subregion([f[1]], H)
    Hbsub = subregion([b[2]], H)
    Hsub = subregion([f[1], b[2]], H)
    @test Hsub == tensor_product(Hfsub, Hbsub)
    @test_throws ArgumentError subregion([f[2], f[1]], H)
    @test_throws ArgumentError subregion([f[4]], H)
    @test subregion(Hf, H) == Hf
    @test subregion(Hb, H) == Hb

    @test subregion([f[1]], Hf) == hilbert_space(f, 1:1)
    @test subregion([b[1]], Hb) == hilbert_space(b, 1:1, 2)

    H = tensor_product(Hf, Hb; constraint=NumberConservation(1))
    Hfsub = subregion([f[1], f[2]], H)
    Hbsub = subregion([b[2], b[3]], H)
    @test dim(Hfsub) == 3
    @test dim(Hbsub) == 3

    using FermionicHilbertSpaces: complementary_subsystem
    @test complementary_subsystem(H, subregion([f[3], b[2]], H)) ==
          subregion([f[1], f[2], b[1], b[3]], H)

end

function _find_subregion_states(H, mapper)
    split = Base.Fix2(split_state, mapper)
    split_state_iterator = if unique_split(mapper)
        Iterators.map(only ∘ only ∘ first ∘ split, basisstates(H))
    else
        Iterators.map(only, Iterators.flatten(Iterators.map(first ∘ split, basisstates(H))))
    end
    unique(split_state_iterator)
end

function _find_combined_states(space, spaces, mapper=state_mapper(space, spaces))
    sub_state_iter = Iterators.product(map(basisstates, spaces)...)
    combine = Base.Fix2(combine_states, mapper)
    state_iterator = if unique_split(mapper)
        Iterators.map(only ∘ first ∘ combine, sub_state_iter)
    else
        Iterators.flatten(Iterators.map(first ∘ combine, sub_state_iter))
    end
    unique(state_iterator)
end

function _find_compatible_complementary_states(H, Hsub, mapper)
    split = Base.Fix2(split_state, mapper)
    split_state_iterator = if unique_split(mapper)
        Iterators.map(only ∘ first ∘ split, basisstates(H))
    else
        Iterators.flatten(Iterators.map(first ∘ split, basisstates(H)))
    end
    unique(fbar for (fsub, fbar) in split_state_iterator if !ismissing(state_index(fsub, Hsub)))
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
    remaining = collect(Iterators.filter(a -> !(a in sub_atoms), atomic_factors(H)))
    # isempty(remaining) && throw(ArgumentError("Complementary subsystem is empty"))
    isempty(remaining) && return nothing
    Hcomp = tensor_product(remaining)
    if isconstrained(H)
        #restrict states in Hcomp to those compatible with states in Hsub
        mapper = state_mapper(H, (Hsub, Hcomp))
        states = _find_compatible_complementary_states(H, Hsub, mapper)
        return constrain_space(Hcomp, states)
    end
    return Hcomp
end

_find_position(n, v::Union{<:AbstractVector,<:Base.Generator}) = (pos = findfirst(==(n), v); isnothing(pos) ? 0 : pos)
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
