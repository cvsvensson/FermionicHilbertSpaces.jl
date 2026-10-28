## Some types
abstract type AbstractBasisState end
abstract type AbstractFockState <: AbstractBasisState end
abstract type AbstractHilbertSpace{S} end
abstract type AbstractConstraint end
abstract type AbstractAtomicHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractProductHilbertSpace{B} <: AbstractHilbertSpace{B} end
abstract type AbstractGroupedHilbertSpace{B} <: AbstractProductHilbertSpace{B} end

const PairWithHilbertSpaces = Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}

"""
	basisstates(H)

Return the basis states of the Hilbert space `H`, in the order used by matrix
representations. The result has `length(basisstates(H)) == dim(H)` and can be iterated and
indexed, with `basisstates(H)[i] == basisstate(i, H)`. Use `collect` to get a `Vector`.
"""
basisstates

struct BasisStates{B,H,I}
    space::H
    iter::I
end
BasisStates(space::AbstractHilbertSpace{B}, iter) where B = BasisStates{B,typeof(space),typeof(iter)}(space, iter)
Base.eltype(::Type{<:BasisStates{B}}) where B = B
Base.length(bs::BasisStates) = dim(bs.space)
Base.iterate(bs::BasisStates) = iterate(bs.iter)
Base.iterate(bs::BasisStates, state) = iterate(bs.iter, state)
Base.keys(bs::BasisStates) = Base.OneTo(length(bs))
Base.firstindex(bs::BasisStates) = 1
Base.lastindex(bs::BasisStates) = length(bs)
Base.@propagate_inbounds Base.getindex(bs::BasisStates, i::Integer) = basisstate(i, bs.space)
Base.getindex(bs::BasisStates, I::AbstractVector{<:Integer}) = [bs[i] for i in I]
Base.:(==)(a::BasisStates, b::BasisStates) = _equal_states(a, b)
Base.:(==)(a::BasisStates, b::AbstractVector) = _equal_states(a, b)
Base.:(==)(a::AbstractVector, b::BasisStates) = _equal_states(a, b)
_equal_states(a, b) = length(a) == length(b) && all(splat(==), zip(a, b))
Base.hash(bs::BasisStates, h::UInt) = hash(collect(bs), h)
# Spaces store a plain vector of their states, not a vector tied to another space
_stored_states(states) = states
_stored_states(bs::BasisStates) = collect(bs)

"""
    basisstate(i, H)

Return the `i`-th basis state of the Hilbert space `H`, in the order of `basisstates(H)`.
Throws a `BoundsError` if `i` is not in `1:dim(H)`. A string or character that spells a
state of `H` is accepted in place of an index, and an `ArgumentError` is thrown if it is not a state of `H`.
"""
basisstate

"""
    state_index(state, H)

Return the index of the basis state `state` in the Hilbert space `H`, the inverse of
`basisstate`. Returns `0` if `state` is not a state of `H`; no exception is thrown in that case.
"""
state_index

"""
	hilbert_space(args...)

Construct a Hilbert space from symbolic degrees of freedom and labels, optionally
with additional arguments such as constraints.
"""
hilbert_space

## Decomposition of spaces
# Every space is a two-level tree: atoms, collected into groups, and a product space is the
# ordinary tensor product of its groups. Wrapper spaces (AbstractSubspace, TransposedSpace)
# forward all of the functions below to their parent (TransposedSpace also wraps the results
# in TransposedSpace).
#
# Identity of atoms is kept separate from their basis:
# - Lookups only use names. `atomic_id` names an atom (or a symbol naming one) and ignores
#   truncation, transposition and constraints; composite spaces are named by `atom_ids`, and
#   positions are found with `atom_position`. This is what subregion, the state mappers,
#   SymbolicState equality and constraint subspaces use.
# - `match_atoms` (issubsystem, complementary_subsystem, ispartition) also matches atoms by
#   name only. A subsystem atom may have a different basis than the atom of the same name in
#   the parent (truncation, spin, transposition); such a difference shows up as missing states,
#   which each operation (partial trace, embed, ...) handles. Complements are built from the
#   parent's atoms.
# - Operators match at the group level, `symbolic_group(op) == group_id(group)`.

"""
    atomic_factors(H)

Return the atoms of `H`: the indivisible spaces it is built from. Atoms are fixed points,
`atomic_factors(a) == (a,)`. For product spaces they are ordered group-major, i.e.
`atomic_factors(H)` is the concatenation of `atomic_factors.(groups(H))`.
"""
atomic_factors(H::AbstractAtomicHilbertSpace) = (H,)
# The i:th atom of `H`. Overload to avoid building all atoms.
atomic_factor(H, i) = atomic_factors(H)[i]

"""
    groups(H)

Return the groups of `H`, which line up with the slots of its basis states: for a product
space, `groups(H)[i]` is the space of `state.states[i]`; otherwise `groups(H) == (H,)`.
Groups are fixed points, `groups(g) == (g,)`, and they are the factors of the ordinary
tensor product, so `tensor_product(groups(H))` recovers `H` (without its constraints).
"""
groups(H::AbstractAtomicHilbertSpace) = (H,)
groups(H::AbstractGroupedHilbertSpace) = (H,)

"""
    factors(H)

Split `H` one level down: a product space into its groups, a grouped space (such as a
fermionic space) into its atoms, and an atomic space into itself. Use this when building
partitions, permutations and constraints. Internal code that needs the state layout
should use `groups` instead.
"""
factors(H::AbstractAtomicHilbertSpace) = (H,)
factors(H::AbstractGroupedHilbertSpace) = atomic_factors(H)

"""
    group_id(atom)

Key used by `tensor_product` to merge atoms into one group (see `combine_into_group`).
It equals `symbolic_group(op)` for every symbolic operator `op` acting on the atom.
"""
group_id(H::AbstractAtomicHilbertSpace) = atomic_id(H)

"""
    atomic_id(atom)

Name of an atom: the key that identifies its degrees of freedom. It is built only from
symbolic identity (basis, label and tags), never from truncation, spin value, transposition
or constraints, so a symbol naming a whole atom has the same id as the atom. Only atoms and
symbols have an `atomic_id`; composite spaces are named by `atom_ids`.
"""
atomic_id(f::AbstractSym) = symbolic_group(f) # generic fallback

"""
    atom_ids(H)

Names of the atoms of `H`, `map(atomic_id, atomic_factors(H))`. Since atoms are ordered
group-major, this is canonical: two spaces over the same degrees of freedom, in the same
order within each group, have equal `atom_ids` whatever their truncation, transposition or
constraints.
"""
atom_ids(H) = map(atomic_id, atomic_factors(H))

# Number of atoms of `H`, without building them
nbr_of_atoms(H) = length(atom_ids(H))

"""
    atom_position(x, H)

Position among `atomic_factors(H)` of the atom named by `x` (an atom or a symbol naming one),
or `0` if `H` has no such atom. The lookup is by `atomic_id` only, so the basis of `x` is
not compared; see `match_atoms` for that.
"""
atom_position(x, H::AbstractHilbertSpace) = something(findfirst(==(atomic_id(x)), atom_ids(H)), 0)
atom_position(x, H::AbstractAtomicHilbertSpace) = atomic_id(x) == atomic_id(H) ? 1 : 0

isconstrained(H::AbstractAtomicHilbertSpace) = false
atomic_factors(f::AbstractSym) = (f,)

partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1

maximum_particles(H::AbstractHilbertSpace) = maximum(particle_number, basisstates(H))

struct InternalRep{T}
    data::T
end
_internal_rep(state, space, ::Type{T}) where T = InternalRep{T}(internal_rep(state, space, T))
_physical_rep(state::InternalRep, space_or_type) = physical_rep(state.data, space_or_type)
Base.convert(::Type{B}, state::InternalRep{T}) where {B<:AbstractBasisState,T} = _physical_rep(state, B)
physical_rep(state, space::AbstractHilbertSpace) = physical_rep(state, statetype(space))
# The following should be overloaded for custom basis states
internal_rep(state, space, ::Type{T}) where T = internal_rep(state, parent(space), T)
# physical_rep(state, type::B<:AbstractBasisState)

# First dispatch on NC operators with precomputation_before_operator_application
# then one can dispatch on abstract syms, spaces or statetypes
function precomputation_before_operator_application(op::NCMul, space::AbstractHilbertSpace)
    map(op -> _precomputation_before_operator_application(op, space), op.factors)
end
precomputation_before_operator_application(::NCAdd, ::AbstractHilbertSpace) = nothing
function precomputation_before_operator_application(op::AbstractSym, space::AbstractHilbertSpace)
    _precomputation_before_operator_application(op, space)
end
_precomputation_before_operator_application(factors, space) = nothing

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

Returns a pair `(outcomes, weights)`. Each element of `outcomes` is a tuple with one
substate per target subsystem, and `weights` holds the coefficient of the
corresponding outcome. A state that splits uniquely gives a single outcome with
weight `1`, for example `((substate1, substate2),), (1,)`.
"""
split_state(state, mapper::AbstractStateMapper) = throw(MethodError(split_state, (state, mapper)))

"""
    combine_states(substates, mapper)

Combine subsystem states using `mapper`.

Returns a pair `(states, weights)`, where `states` holds the resulting states of the
combined space and `weights` the coefficient of each. A state that combines uniquely
gives `(state,), (1,)`.
"""
combine_states(substates, mapper::AbstractStateMapper) = throw(MethodError(combine_states, (substates, mapper)))

"""
    kron_phase_factor(mapper)

Return phase-factor function associated with `mapper` for fermionic tensor products.
"""
kron_phase_factor(mapper::AbstractStateMapper) = throw(MethodError(kron_phase_factor, (mapper,)))

struct AtomicStateMapper <: AbstractStateMapper end
function state_mapper(H::AbstractAtomicHilbertSpace, Hs)
    atom_ids(only(Hs)) == atom_ids(H) || throw(ArgumentError("For atomic subspaces, the only valid partition is the whole space, got `Hs` = $Hs"))
    AtomicStateMapper()
end
function split_state(state, ::AtomicStateMapper)
    # the return state here is a tuple of tuples because each state is a state in the tuple of subsystem and complement, but the complement is empty
    ((state,),), (1,)
end
function combine_states(states, ::AtomicStateMapper)
    (only(states),), (1,)
end
function combine_states(states, ::AbstractAtomicHilbertSpace)
    (only(states),), (1,)
end
kron_phase_factor(::AtomicStateMapper) = (f1, f2) -> 1

"""
    subregion(Hs, H::AbstractHilbertSpace)

Return the subsystem of `H` spanned by the factors `Hs`, given as a space, a symbolic
operator, or a list/tuple of spaces or symbolic operators (e.g. `[f[1], b[2]]`).

If `Hs` contains only symbolic operators, their order does not matter: the modes are ordered
as in `H`. If any element of `Hs` is a space, the modes must keep their order within each
group of `H` (e.g. the Jordan-Wigner order of fermions), otherwise an `ArgumentError` is thrown.

# Examples
```julia
@fermions f
H1 = hilbert_space(f, 1:1)
H2 = hilbert_space(f, 2:2)
H3 = hilbert_space(f, 3:3)
H = tensor_product((H1, H2, H3))
Hsub = subregion((H1, H3), H)
Hsub == subregion([f[3], f[1]], H)
```
"""
function subregion(Hs, H::AbstractHilbertSpace)
    Hs_items = Hs isa Tuple || Hs isa AbstractVector ? Hs : (Hs,)
    names = collect(Iterators.flatten(Iterators.map(atomic_factors, Hs_items)))
    isempty(names) && throw(ArgumentError("`Hs` must contain at least one space or symbolic operator."))
    positions = map(x -> atom_position(x, H), names)
    all(>(0), positions) || throw(ArgumentError("The spaces/operators in `Hs` must match atomic factors of `H`, got the following unmatched ones: $(names[iszero.(positions)])."))
    allunique(positions) || throw(ArgumentError("`Hs` must not contain duplicate atomic factors, got $names."))
    any(x -> x isa AbstractHilbertSpace, Hs_items) || sort!(positions)

    # Resolve the names to H's own atoms, so the basis of the subregion is that of H
    Hsub = tensor_product(collect(atomic_factors(H))[positions])
    match_atoms(Hsub, H) # checks the ordering within groups
    !isconstrained(H) && return Hsub
    states = _find_subregion_states(H, state_mapper(H, (Hsub,)))
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
    @test subregion([b[2], f[1]], H) == subregion([f[1], b[2]], H)
    @test_throws ArgumentError subregion(hilbert_space(f, [2, 1]), H)
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

@testitem "Subregion: symbols interleaving the factors of a product of sectors" begin
    @fermions f
    up = [(i, :↑) for i in 1:4]
    dn = [(i, :↓) for i in 1:4]
    H = tensor_product(hilbert_space(f, up, NumberConservation(2)), hilbert_space(f, dn, NumberConservation(2)))
    @test subregion([f[(i, σ)] for i in 1:2 for σ in (:↑, :↓)], H) ==
          subregion(hilbert_space(f, [up[1:2]; dn[1:2]]), H)
end

@testitem "Matching syms and spaces with mutable labels" begin
    # Labels are compared by value: separately allocated but equal vectors must give the same
    # syms, spaces and atomic ids, otherwise subregion, partial_trace and states break.
    using FermionicHilbertSpaces: atomic_id, SymbolicState, basisstate
    using LinearAlgebra
    @fermions f
    @spins s 1 // 2
    @bosons b
    l1() = [1, 2] # fresh vector on each call
    l2() = [3, 4]
    for (x, space) in [(f, l -> hilbert_space(f[l])), (s, l -> hilbert_space(s[l])), (b, l -> hilbert_space(b[l], 2))]
        @test x[l1()] == x[l1()]
        @test hash(x[l1()]) == hash(x[l1()])
        @test atomic_id(x[l1()]) == atomic_id(space(l1()))
        @test atomic_id(x[l1()]) != atomic_id(space(l2()))
    end

    # Product space matched against freshly built subspaces and states
    Hf() = tensor_product(hilbert_space(f, [l1(), l2()]), hilbert_space(s, [l1(), l2()]), hilbert_space(b, [l1(), l2()], 2))
    H = Hf()
    @test H == Hf()
    @test hash(H) == hash(Hf())
    Hsub = tensor_product(hilbert_space(f[l2()]), hilbert_space(s[l1()]), hilbert_space(b[l2()], 2))
    @test subregion([f[l2()], s[l1()], b[l2()]], H) == Hsub
    @test partial_trace(1.0*I(dim(H)), H => Hsub) == 8 * I(dim(Hsub))
    @test SymbolicState(Hf(), basisstate(2, H))' * SymbolicState(Hf(), basisstate(2, H)) == 1
end

@testitem "Space decomposition contract" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: atomic_factors, groups, factors, group_id, group_ids, atomic_id,
        atom_ids, atom_position, issubsystem, ispartition, complementary_subsystem, TransposedSpace, GenericHilbertSpace,
        AbstractSubspace, ProductState
    @fermions f
    @fermions g
    @bosons b
    @spins s 1 // 2
    @majoranas γ
    Hf = hilbert_space(f, 1:3)
    Hfb = tensor_product(hilbert_space(f, 1:2), hilbert_space(b[1], 2))
    zoo = [
        Hf,
        hilbert_space(f[1]),
        hilbert_space(b[1], 3),
        hilbert_space(s[1]),
        Hfb,
        tensor_product(hilbert_space(f[1]), hilbert_space(b[1], 2), hilbert_space(f[2])),
        tensor_product(hilbert_space(f, 1:2), hilbert_space(g, 1:2)),
        hilbert_space(γ, 1:4),
        hilbert_space(f, 1:3, NumberConservation(1)),
        tensor_product(hilbert_space(f, 1:2), hilbert_space(b[1], 2); constraint=NumberConservation(1)),
        constrain_space(Hfb, collect(basisstates(Hfb))[1:5]),
        TransposedSpace(hilbert_space(f, 1:2)),
        TransposedSpace(Hfb),
        tensor_product(GenericHilbertSpace(:A, [:a, :b]), GenericHilbertSpace(:B, [:c, :d, :e])),
    ]
    unwrap(H) = H isa AbstractSubspace ? parent(H) : H
    flat_atoms(spaces) = reduce(vcat, map(collect ∘ atomic_factors, spaces))
    for H in zoo
        atoms = collect(atomic_factors(H))
        grps = collect(groups(H))
        # atoms and groups are fixed points
        @test all(a -> collect(atomic_factors(a)) == [a], atoms)
        @test all(gr -> collect(groups(gr)) == [gr], grps)
        # canonical group-major atom order, and factors refine to the same atoms
        @test atoms == flat_atoms(grps)
        @test atoms == flat_atoms(factors(H))
        # round trips through tensor_product
        @test tensor_product(atoms) == unwrap(H)
        @test tensor_product(grps) == unwrap(H)
        @test tensor_product(collect(factors(H))) == unwrap(H)
        # state layout: one slot per group
        for state in basisstates(H)
            slots = state isa ProductState ? state.states : (state,)
            @test length(slots) == length(grps)
            @test all(state_index(x, gr) > 0 for (x, gr) in zip(slots, grps))
        end
        @test collect(group_ids(H)) == map(group_id, grps)
        @test allunique(group_ids(H))
        @test allunique(atom_ids(H))
        @test [atom_position(a, H) for a in atoms] == eachindex(atoms)
        # subsystems
        @test ispartition(atoms, H)
        @test ispartition(grps, H)
        @test collect(atomic_factors(subregion(atoms, H))) == atoms
        ρ = rand(ComplexF64, dim(H), dim(H))
        ρ = ρ + ρ'
        for a in atoms
            @test issubsystem(a, H)
            @test collect(atomic_factors(subregion([a], H))) == [a]
            comp = complementary_subsystem(H, a)
            @test (isnothing(comp) ? [] : collect(atomic_factors(comp))) == filter(!=(a), atoms)
            @test tr(partial_trace(ρ, H => a)) ≈ tr(ρ)
        end
    end

    # wrappers forward the decomposition to their parent
    for H in (Hf, Hfb)
        for W in (constrain_space(H, NumberConservation(1)), constrain_space(H, collect(basisstates(H))[1:2]))
            @test atomic_factors(W) == atomic_factors(H)
            @test groups(W) == groups(H)
            @test factors(W) == factors(H)
            @test atom_ids(W) == atom_ids(H)
        end
        Ht = TransposedSpace(H)
        @test collect(atomic_factors(Ht)) == map(TransposedSpace, collect(atomic_factors(H)))
        @test collect(groups(Ht)) == map(TransposedSpace, collect(groups(H)))
        @test collect(factors(Ht)) == map(TransposedSpace, collect(factors(H)))
        @test atom_ids(Ht) == atom_ids(H)
    end
end

@testitem "Atom identity and ordering" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: atomic_factors, atomic_id, atom_ids, group_id, symbolic_group, issubsystem,
        ispartition, complementary_subsystem, TransposedSpace
    @fermions f
    @bosons b
    @spins s 1 // 2
    # a symbol naming a whole atom has the atom's id, and operators act on the atom's group
    for (sym, op, H) in ((f[1], f[1]', hilbert_space(f[1])),
        (b[1], b[1]', hilbert_space(b[1], 3)),
        (s[1], s[1][:z], hilbert_space(s[1])))
        @test atomic_id(sym) == atomic_id(H)
        @test group_id(H) == symbolic_group(op)
    end

    # lookup by name resolves to the parent's atom, and the subsystem predicates also match by
    # name, so a different truncation is accepted
    Hb3 = hilbert_space(b[1], 3)
    Hb5 = hilbert_space(b[1], 5)
    Hbig = tensor_product(Hb3, hilbert_space(b[2], 3))
    @test atomic_id(Hb3) == atomic_id(Hb5)
    @test Hb3 != Hb5
    @test subregion([Hb5], Hbig) == Hb3
    @test subregion([b[1]], Hbig) == Hb3
    @test issubsystem(Hb3, Hbig)
    @test issubsystem(Hb5, Hbig)
    @test ispartition([Hb5, hilbert_space(b[2], 3)], Hbig)
    @test complementary_subsystem(Hbig, Hb5) == hilbert_space(b[2], 3)
    @test partial_trace(Matrix(I, dim(Hbig), dim(Hbig)), Hbig => Hb5) == Diagonal([3, 3, 3, 0, 0])

    # composite spaces are named by their atoms, ignoring truncation and constraints
    @test atom_ids(Hbig) == atom_ids(tensor_product(Hb5, hilbert_space(b[2], 2)))
    @test atom_ids(hilbert_space(f, 1:2)) == atom_ids(hilbert_space(f, 1:2, NumberConservation(1)))
    @test atom_ids(hilbert_space(f, 1:2)) != atom_ids(hilbert_space(f, [2, 1]))
    @test_throws ArgumentError atomic_id(hilbert_space(f, 1:2))

    # transposition is not part of the name, so it is not checked by the predicates
    Hf12 = hilbert_space(f, 1:2)
    Hf1t = TransposedSpace(hilbert_space(f[1]))
    @test atomic_id(Hf1t) == atomic_id(f[1])
    @test issubsystem(Hf1t, Hf12)
    @test issubsystem(hilbert_space(f[1]), TransposedSpace(Hf12))
    @test issubsystem(Hf1t, TransposedSpace(Hf12))
    @test complementary_subsystem(Hf12, Hf1t) == hilbert_space(f[2])
    @test complementary_subsystem(TransposedSpace(Hf12), Hf1t) == TransposedSpace(hilbert_space(f[2]))

    # atom order is canonical (group-major), so interleaving doesn't matter
    Hf1, Hf2, Hb = hilbert_space(f[1]), hilbert_space(f[2]), hilbert_space(b[1], 2)
    A = tensor_product(Hf1, Hb, Hf2)
    B = tensor_product(Hf1, Hf2, Hb)
    @test A == B
    @test hash(A) == hash(B)
    @test collect(atomic_factors(A)) == [Hf1, Hf2, Hb]

    # ordering within a fermionic group is checked regardless of wrappers
    Hf = hilbert_space(f, 1:3)
    for H in (Hf, hilbert_space(f, 1:3, NumberConservation(1)), constrain_space(Hf, collect(basisstates(Hf))[1:4]), tensor_product(Hf, Hb))
        @test issubsystem(hilbert_space(f, [1, 3]), H)
        @test !issubsystem(hilbert_space(f, [3, 1]), H)
        @test !issubsystem(hilbert_space(f, [4]), H)
    end
    # but not between groups
    @test issubsystem(tensor_product(Hb, Hf1), tensor_product(Hf, Hb))

    # Majorana spaces recognise their own atoms
    @majoranas γ
    Hm = hilbert_space(γ, 1:4)
    pair = first(atomic_factors(Hm))
    @test issubsystem(pair, Hm)
    @test subregion([pair], Hm) == pair
    # and their fast atom_ids, atom_position and atomic_factor agree with the generic definitions
    using FermionicHilbertSpaces: atom_position, atomic_factor, nbr_of_atoms
    # including spaces with several Majorana names from one block
    @majoranas ma mc
    Hac = tensor_product(hilbert_space(ma, 1:2), hilbert_space(mc, 1:2))
    for H in (Hm, hilbert_space(γ, [1, :a, 2, 3]), hilbert_space(γ, 1:4, ParityConservation(1)), Hac, tensor_product(Hac, Hb))
        ids = map(atomic_id, atomic_factors(H))
        @test atom_ids(H) == ids
        @test eltype(atom_ids(H)) == eltype(ids)
        @test nbr_of_atoms(H) == length(atomic_factors(H))
        @test [atomic_factor(H, i) for i in 1:nbr_of_atoms(H)] == collect(atomic_factors(H))
        others = [γ[1], f[1], first(atomic_factors(hilbert_space(γ, [2, 1]))), first(atomic_factors(hilbert_space(γ, [1, 3]))), atomic_factors(Hac)...]
        for x in vcat(collect(atomic_factors(Hm)), others)
            @test atom_position(x, H) == something(findfirst(==(atomic_id(x)), ids), 0)
        end
    end
    # symbolic states on equal but distinct spaces are equal and hash alike
    k1 = Ket("01", hilbert_space(γ, 1:4))
    k2 = Ket("01", hilbert_space(γ, 1:4))
    @test k1.space !== k2.space
    @test k1 == k2
    @test hash(k1) == hash(k2)
    @test k1 != Ket("01", hilbert_space(γ, [3, 4, 1, 2]))

    # symbolic states on different spaces of the same group can't be multiplied
    @test_throws ArgumentError Kets(hilbert_space(f, 1:2))("10") * Kets(hilbert_space(f, 2:3))("01")

    # transposed spaces support partial traces, consistent with transposition
    Hsub = hilbert_space(f, [1, 3])
    m = rand(ComplexF64, dim(Hf), dim(Hf))
    @test partial_trace(transpose(m), TransposedSpace(Hf) => TransposedSpace(Hsub)) ≈ transpose(partial_trace(m, Hf => Hsub))
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
    unique(fbar for (fsub, fbar) in split_state_iterator if !iszero(state_index(fsub, Hsub)))
end


"""
    match_atoms(Hsub, H)

Return the positions of the atoms of `Hsub` among `atomic_factors(H)`, after checking that
`Hsub` is a subsystem of `H`. Each atom of `Hsub` is looked up by name (`atom_position`) and
must then
- occur in `H`, and only once in `Hsub`,
- keep the relative order it has in `H` among the atoms of its group (e.g. the
  Jordan-Wigner order of fermions). The order between groups does not matter.

Throws an `ArgumentError` describing the first violation.
"""
function match_atoms(Hsub, H::AbstractHilbertSpace)
    positions, problem = _match_atoms(Hsub, H)
    isnothing(problem) || throw(ArgumentError(_match_atoms_message(problem, Hsub, H)))
    return positions
end

# Positions of the atoms of Hsub in H, and why Hsub is not a subsystem of H (or nothing).
# The reason is a tuple (kind, data...), only formatted by `match_atoms` when it throws.
function _match_atoms(Hsub, H)
    sub_atoms = collect(atomic_factors(Hsub))
    positions = map(a -> atom_position(a, H), sub_atoms)
    for (a, pos) in zip(sub_atoms, positions)
        iszero(pos) && return positions, (:missing, a)
    end
    allunique(positions) || return positions, (:duplicate,)
    sub_groups = map(group_id, sub_atoms)
    for id in unique(sub_groups)
        issorted(positions[findall(==(id), sub_groups)]) || return positions, (:order,)
    end
    return positions, nothing
end
function _match_atoms_message(problem, Hsub, H)
    kind = first(problem)
    kind == :missing && return "The atoms of `Hsub` must be atoms of `H`, got `$(problem[2])` which is not in `$H`."
    kind == :duplicate && return "The atoms of `Hsub` must be distinct, got duplicates in `$Hsub`."
    return "The atoms of `Hsub` must keep their order within each group of `H`, got `$Hsub` and `$H`."
end

"""
    match_same_atoms(H1, H2)

Check that `H1` and `H2` have the same atoms, in the same order within each group (see
`match_atoms`). The order between groups may differ. Throws an `ArgumentError` otherwise.
"""
function match_same_atoms(H1, H2)
    match_atoms(H1, H2)
    nbr_of_atoms(H1) == nbr_of_atoms(H2) || throw(ArgumentError("`H1` and `H2` must have the same atoms, but `$H2` has atoms that are not in `$H1`."))
    return nothing
end

"""
    issubsystem(Hsub, H)

Check that `Hsub` is a subsystem of `H`, see `match_atoms`.
"""
issubsystem(Hsub, H::AbstractHilbertSpace) = isnothing(last(_match_atoms(Hsub, H)))

"""
    isorderedsubsystem(Hsub, H)

Check that `Hsub` is a subsystem of `H` whose atoms are in the same order as in `H`, also
between groups.
"""
function isorderedsubsystem(Hsub, H::AbstractHilbertSpace)
    positions, problem = _match_atoms(Hsub, H)
    isnothing(problem) && issorted(positions)
end

"""
    ispartition(Hs, H::AbstractHilbertSpace)

Check that the spaces `Hs` are subsystems of `H` (see `match_atoms`) that together contain
each atom of `H` exactly once.
"""
function ispartition(partition, H::AbstractHilbertSpace)
    matches = map(Hsub -> _match_atoms(Hsub, H), partition)
    all(isnothing ∘ last, matches) && ispartition(map(first, matches), nbr_of_atoms(H))
end

"""
    isorderedpartition(Hs, H::AbstractHilbertSpace)

Like `ispartition`, but the atoms of each subsystem must also be in the same order as in `H`.
"""
function isorderedpartition(partition, H::AbstractHilbertSpace)
    matches = map(Hsub -> _match_atoms(Hsub, H), partition)
    all(isnothing ∘ last, matches) && isorderedpartition(map(first, matches), nbr_of_atoms(H))
end

"""
    complementary_subsystem(H, Hsub)

Return the subsystem of `H` made of the atoms not in `Hsub`, or `nothing` if there are none.
`Hsub` must be a subsystem of `H`, see `match_atoms`. If `H` is constrained, the complement
only keeps the states that are compatible with some state of `Hsub`.
"""
function complementary_subsystem(H::AbstractHilbertSpace, Hsub)
    positions = match_atoms(Hsub, H)
    remaining = [a for (n, a) in enumerate(atomic_factors(H)) if n ∉ positions]
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

# Position of `x` in the collection `v`, or 0
_position_in(x, v) = something(findfirst(==(x), v), 0)

# Partitions of positions 1:N, or of the elements of `labels`
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
            pos = _position_in(label, labels)
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
            pos = _position_in(label, order)
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

@testitem "basisstate and state_index failure model" begin
    using FermionicHilbertSpaces: basisstate, state_index
    @fermions f
    @spin s
    @boson b
    Hf = hilbert_space(f, 1:2)
    Hs = hilbert_space(s, 1)
    Hb = hilbert_space(b, 3)
    spaces = (
        Hf, Hs, Hb,
        hilbert_space(f, 1:2, NumberConservation()),
        hilbert_space(f, 1:2, NumberConservation(1)),
        tensor_product(Hf, Hs),
    )
    for H in spaces
        for i in (0, -1, dim(H) + 1)
            @test_throws BoundsError basisstate(i, H)
        end
        @test basisstate(1, H) == first(basisstates(H))
        @test basisstate(dim(H), H) == last(collect(basisstates(H)))
        @test all(i -> state_index(basisstate(i, H), H) == i, 1:dim(H))
    end
    @test state_index(FermionicHilbertSpaces.BosonicState(3), Hb) == 0
end

@testitem "Basis states collection" begin
    @fermions f
    H = hilbert_space(f, 1:70)
    bs = basisstates(H)
    @test length(bs) == dim(H)
    @test first(bs) == basisstate(1, H)
    @test last(bs) == bs[end] == basisstate(dim(H), H)
    i = dim(H) ÷ 3
    @test state_index(bs[i], H) == i
    @test bs[1:3] == collect(Iterators.take(bs, 3))
    @test_throws BoundsError bs[dim(H) + 1]
end
