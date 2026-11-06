
combine_into_group(::AbstractAtomicHilbertSpace, atoms) = only(atoms)
combine_into_group(group_id, spaces) = length(spaces) == 1 ? only(spaces) : throw(ArgumentError("The group `$group_id` requires a single space, got $spaces."))
# Base's tuple `map` and `setindex`, and `LinearIndices`, are not inferred past 32 entries.
@generated _tuple_map(f::F, a::NTuple{N, Any}, b::NTuple{N, Any}) where {F, N} = Expr(:tuple, (:(f(a[$i], b[$i])) for i in 1:N)...)
function _linear_index(inds::NTuple{N, Int}, axes) where {N}
    k, stride = 1, 1
    for n in 1:N
        k += (inds[n] - 1) * stride
        stride *= length(axes[n])
    end
    return k
end

"""
    ProductState(states::Tuple)

A basis state of a tensor product space, holding one basis state per factor, e.g.
`ProductState((SpinState(1//2), BosonicState(0)))`.
"""
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
# ProductSpaces consists of a list of groups (stored as `factors`) and their atoms, in group-major order
struct ProductSpace{B,C,A,T,LI,CI} <: AbstractProductHilbertSpace{B}
    factors::C
    atoms::Vector{A}
    atom_index::Dict{Any,Int} # atomic_id => position in atoms. Only used for lookups, so not concretely typed
    fast_path::T
    lininds::LI
    cartinds::CI
    function ProductSpace(factors::C) where {C}
        length(factors) == 0 && throw(ArgumentError("A product space must have at least one factor."))
        atoms = collect(Iterators.flatten(map(atomic_factors, factors)))
        A = eltype(atoms)
        B = ProductState{Tuple{map(statetype, factors)...}}
        atom_index = Dict{Any,Int}(atomic_id(a) => i for (i, a) in enumerate(atoms))
        length(atom_index) == length(atoms) || throw(ArgumentError("The atoms of a product space must be distinct, got duplicates in $(map(atomic_id, atoms))."))
        fast_path = all(has_internal_rep(f, Int) for f in factors) ? zero(Int) : missing
        lininds = LinearIndices(map(dim, factors))
        cartinds = CartesianIndices(map(dim, factors))
        LI = typeof(lininds)
        CI = typeof(cartinds)
        new{B,C,A,typeof(fast_path),LI,CI}(factors, atoms, atom_index, fast_path, lininds, cartinds)
    end
end
# The copy of a `ProductSpace` that GPU kernels use, without the atoms
struct DeviceProductSpace{B, C, LI, CI} <: AbstractHilbertSpace{B}
    factors::C
    lininds::LI
    cartinds::CI
end
const _ProductSpace{B} = Union{ProductSpace{B}, DeviceProductSpace{B}}
fast_path(space::ProductSpace) = space.fast_path
fast_path(::DeviceProductSpace) = 0
isconstrained(H::ProductSpace) = false
Base.:(==)(H1::ProductSpace, H2::ProductSpace) = H1.factors == H2.factors && H1.atoms == H2.atoms
Base.hash(H::ProductSpace, h::UInt) = hash(H.factors, hash(H.atoms, h))
atomic_factors(H::ProductSpace) = H.atoms
nbr_of_atoms(H::ProductSpace) = length(H.atoms)
factors(H::ProductSpace) = H.factors
groups(H::_ProductSpace) = H.factors
dim(H::ProductSpace) = mapfoldl(dim, Base.Checked.checked_mul, groups(H); init = 1)
atom_position(x, H::ProductSpace) = get(H.atom_index, atomic_id(x), 0)

# Iterates `cartinds` rather than `Iterators.product`, whose iteration allocates per state from 12 factors on
basisstates(H::ProductSpace) = BasisStates(H, Iterators.map(I -> _product_basisstate(I, H), H.cartinds))
Base.@propagate_inbounds basisstate(n::Integer, H::_ProductSpace) = _product_basisstate(H.cartinds[n], H)
Base.@propagate_inbounds _product_basisstate(I::CartesianIndex, H::_ProductSpace) = ProductState(_tuple_map(basisstate, Tuple(I), H.factors))
function _state_index(state::ProductState, H::_ProductSpace)
    length(state.states) == length(H.factors) || throw(MethodError(state_index, (state, H)))
    inds = _tuple_map(_state_index, state.states, H.factors)
    # `all` on a tuple of one type keeps the tuple in memory, which on a GPU is slow local memory
    mapreduce(>(0), &, inds) || return 0
    return _linear_index(inds, H.lininds.indices)
end
# `lininds` is exact only when the product of the factor dimensions fits in an `Int`
has_arithmetic_state_index(H::ProductSpace) = all(has_arithmetic_state_index, H.factors) && prod(big ∘ dim, H.factors) <= typemax(Int)

maximum_particles(H::ProductSpace) = sum(maximum_particles, groups(H))
particle_number(state::ProductState) = sum(particle_number, atomic_factors(state))
parity(state::ProductState) = prod(parity, atomic_factors(state))


@testitem "ProductSpace" begin
    import FermionicHilbertSpaces: state_index, basisstate, state_mapper, ProductState, substate, complementary_subsystem, atomic_factors, atomic_id, split_state
    @fermions a b
    @fermions c
    Ha = hilbert_space(a[1])
    Hb = hilbert_space(b[1])
    Hc = hilbert_space(c[1])
    H = tensor_product((Ha, Hb, Hc))
    @test length(H.factors) == 2
    @test length(H.atoms) == 3
    @test dim(H) == dim(Ha) * dim(Hb) * dim(Hc)
    @test H.atom_index == Dict(atomic_id(Ha) => 1, atomic_id(Hb) => 2, atomic_id(Hc) => 3)
    @test_throws ArgumentError tensor_product([Ha, Ha])

    Hab = H.factors[1]
    @test atomic_factors(Hab) == [Ha, Hb]
    @test dim(Hab) == dim(Ha) * dim(Hb)

    @boson boson
    Hboson = hilbert_space(boson, 2)
    H2 = tensor_product([H, Hboson])
    @test dim(H2) == dim(H) * dim(Hboson)
    @test length(H2.factors) == 3

    @test collect(basisstates(H2)) == vec([ProductState((s_f.states..., s_b)) for s_f in basisstates(H), s_b in basisstates(Hboson)])
    for (i, state) in enumerate(basisstates(H))
        @test state_index(state, H) == i
        @test basisstate(i, H) == state
    end
    # a state that is missing in any factor is not in H
    @test state_index(ProductState((FockNumber(0), FockNumber(0), FermionicHilbertSpaces.BosonicState(2))), H2) == 0
    @test state_index(ProductState((FockNumber(4), FockNumber(0), FermionicHilbertSpaces.BosonicState(1))), H2) == 0
    @test state_index(ProductState((FockNumber(3), FockNumber(1), FermionicHilbertSpaces.BosonicState(1))), H2) == dim(H2)

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
    amat = representation(a[1], Ha)
    bmat = representation(b[1], Hb)
    cmat = representation(c[1], Hc)
    @test embed(amat, Ha => H) * embed(bmat, Ha => H) ≈ -embed(bmat, Ha => H) * embed(amat, Ha => H)
    @test embed(amat, Ha => H) * embed(cmat, Ha => H) ≈ embed(cmat, Ha => H) * embed(amat, Ha => H)
    @test embed(amat, Ha => H) == representation(a[1], H)
    @test embed(bmat, Hb => H) == representation(b[1], H)
    @test embed(cmat, Hc => H) == representation(c[1], H)

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


    @test representation(a[1] + 1, Hprod; projection = true) == embed(representation(a[1], Ha), Ha => Hprod; projection = true) + I
    @test representation(c[1], Hprod; projection = true) == embed(representation(c[1], Hc), Hc => Hprod; projection = true)

    @test representation(a[1]' * b[1], Hprod) == embed(representation(a[1]' * b[1], Hab), Hab => Hprod; projection = true)

end

@testitem "dim of a ProductSpace throws on overflow" begin
    @spins S 1 // 2
    H = hilbert_space(S, 1:64)
    @test_throws OverflowError dim(H)
    @test S[1][:x] * Ket(fill("↑", 64), H) == Ket(["↓"; fill("↑", 63)], H) / 2
end

@testitem "ProductSpaceMapper combines vectors of substates" begin
    import FermionicHilbertSpaces: state_mapper, combine_states, basisstates
    @fermions f
    @spins S 1 // 2
    Hs = [hilbert_space(S[n]) for n in 1:3]
    H = tensor_product(hilbert_space(f, 1:2), Hs...)
    targets = (hilbert_space(f, 1:2), Hs...)
    mapper = state_mapper(H, targets)
    for substates in Iterators.product(map(basisstates, targets)...)
        @test combine_states(collect(Any, substates), mapper) == combine_states(substates, mapper)
    end
    @test_throws ArgumentError combine_states(Any[first(basisstates(Hs[1]))], mapper)
end

@testitem "split_state and combine_states infer on a fermion, boson and spin product" begin
    import FermionicHilbertSpaces: state_mapper, split_state, combine_states
    @fermions f
    @boson b
    @spin s 1 // 2
    Hb, Hs = hilbert_space(b, 3), hilbert_space(s)
    H = tensor_product(hilbert_space(f, 1:3), Hb, Hs)
    mapper = state_mapper(H, (hilbert_space(f, [1, 3]), tensor_product(hilbert_space(f, 2:2), Hs), Hb))
    for state in basisstates(H)
        substates = only(first(@inferred split_state(state, mapper)))
        @test only(first(@inferred combine_states(substates, mapper))) == state
    end
end

@testitem "Bose-Hubbard matrix on more than 32 sites" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: matrix_representation
    @bosons b
    space(n) = hilbert_space(b, 1:n, 2, NumberConservation(1))
    nb(i) = b[i]' * b[i]
    hamiltonian(n) = -sum(b[i]' * b[i + 1] + hc for i in 1:(n - 1)) + sum(nb(i) * (nb(i) - 1) for i in 1:n)
    function allocations(n)
        H, op = space(n), hamiltonian(n)
        matrix_representation(op, H)
        return @allocations matrix_representation(op, H)
    end
    @test allocations(33) < 2 * allocations(31)

    n = 33
    exact = sort([-2cos(k * pi / (n + 1)) for k in 1:n])
    @test eigvals(Symmetric(Matrix(matrix_representation(hamiltonian(n), space(n))))) ≈ exact
end

@testitem "Matrix on a fermion, boson and spin product of more than 32 groups" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: matrix_representation, BosonicState, SpinState
    @fermions f
    @bosons b
    @spins s 1 // 2
    excitation(state::BosonicState) = state.n // 1
    excitation(state::SpinState) = state.m + 1 // 2
    raise(i) = isodd(i) ? b[i]' : s[i][:+]
    lower(i) = isodd(i) ? b[i] : s[i][:-]
    fermion_spaces = [hilbert_space(f, i:i) for i in 1:2]
    # The fermions form one group, so this chain gives 33 groups
    n = 32
    chain_spaces = Tuple(isodd(i) ? hilbert_space(b[i], 2) : hilbert_space(s[i]) for i in 1:n)
    op = -(f[1]' * f[2] + hc) - sum(raise(i) * lower(i + 1) + hc for i in 1:(n - 1))
    function space(excitations)
        return tensor_product(
            (fermion_spaces..., chain_spaces...);
            constraint = NumberConservation(1, fermion_spaces) * AdditiveConstraint(excitations // 1, chain_spaces, excitation)
        )
    end
    function allocations(excitations)
        H = space(excitations)
        matrix_representation(op, H)
        return @allocations matrix_representation(op, H)
    end
    # A cost per basis state would grow with the sector
    @test allocations(2) < 2 * allocations(1)

    H = space(1)
    exact = sort([-2cos(k * pi / (n + 1)) + d for k in 1:n for d in (-1, 1)])
    @test eigvals(Symmetric(Matrix(matrix_representation(op, H)))) ≈ exact
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

partial_trace_phase_factor_eltype(space::ProductSpace) = promote_type((partial_trace_phase_factor_eltype(H) for H in groups(space))...)
partial_transpose_phase_factor_eltype(space::ProductSpace) = promote_type((partial_transpose_phase_factor_eltype(H) for H in groups(space))...)

function partial_trace_phase_factor(state1, state2, space::ProductSpace)
    prod(map(partial_trace_phase_factor, state1.states, state2.states, space.factors); init=1)
end
function partial_transpose_phase_factor(state1, state2, space::ProductSpace)
    prod(map(partial_transpose_phase_factor, state1.states, state2.states, space.factors); init=1)
end

struct ProductOperator{C, O, S, I}
    ops::O
    spaces::S
    inds::I # indices of active spaces in full space, a `Vector{Int}` except in the copies for GPU kernels
    ProductOperator{C, O, S, I}(ops, spaces, inds) where {C, O, S, I} = new{C, O, S, I}(ops, spaces, inds)
end
ProductOperator{C}(ops, spaces, inds) where {C} = ProductOperator{C, typeof(ops), typeof(spaces), typeof(inds)}(ops, spaces, inds)
function ProductOperator(ops::O, spaces::S, inds::Vector{Int}) where {O,S}
    length(ops) == length(spaces) || throw(ArgumentError("`ops` and `spaces` must have the same length, got $(length(ops)) and $(length(spaces))."))
    C = promote_type(Iterators.map(mat_eltype, ops)...)
    ProductOperator{C}(ops, spaces, inds)
end
mat_eltype(::ProductOperator{C}) where {C} = C

function has_internal_rep(space::AbstractHilbertSpace, ::Type{T}) where {T}
    iszero(dim(space)) && return false
    state = basisstate(1, space)
    has_internal_rep(state, space, T)
end
function has_internal_rep(state::ProductState, space::ProductSpace, ::Type{T}) where {T}
    all(Iterators.map((s, f) -> has_internal_rep(s, f, T), state.states, groups(space)))
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

internal_rep(state::ProductState, space::ProductSpace, ::Type{T}) where T<:Integer = T(state_index(state, space))
physical_rep(state::T, space::ProductSpace) where T<:Integer = basisstate(state, space)

function _apply_local_operators(ops::ProductOperator, state::ProductState{B}, space::_ProductSpace, precomps) where {B}
    if !ismissing(fast_path(space))
        T = typeof(fast_path(space))
        internal_reps = _tuple_map((s, f) -> _internal_rep(s, f, T), state.states, groups(space))
        newrep, amp = _apply_local_operators_fast(ops, internal_reps, precomps)
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
@inline function _apply_local_operators_fast(ops::ProductOperator{C}, internal_reps::Tuple{InternalRep{T}, Vararg{InternalRep{T}}}, precomps) where {C, T}
    return _apply_group_operators(one(C), internal_reps, internal_reps, ops.inds, 1, ops.ops, ops.spaces, precomps)
end
function _apply_local_operators_slow(ops::ProductOperator{C}, state::ProductState{B}, space::ProductSpace, precomps) where {C, B <: Tuple}
    newstates, amp = _apply_group_operators(one(C), state.states, state.states, ops.inds, 1, ops.ops, ops.spaces, precomps)
    return ProductState{B}(newstates), amp
end
# Apply the k-th local operator to the substate `states[inds[k]]` and recurse over the remaining
# operators. Recursing over the tuples `ops`, `spaces` and `precomps` keeps each call concretely typed.
@inline _apply_group_operators(amp, states, newstates, inds, k, ::Tuple{}, ::Tuple{}, ::Tuple{}) = (newstates, amp)
@inline function _apply_group_operators(amp::C, states, newstates, inds, k, ops::Tuple, spaces::Tuple, precomps::Tuple) where {C}
    n = inds[k]
    new_local_state, local_amp = _apply_local_operators(first(ops), states[n], first(spaces), first(precomps))
    return _apply_group_operators(
        amp * convert(C, local_amp), states, ntuple(i -> i == n ? new_local_state : newstates[i], Val(length(newstates))),
        inds, k + 1, Base.tail(ops), Base.tail(spaces), Base.tail(precomps))
end

add_tag(H::ProductSpace, tag) = ProductSpace(map(f -> add_tag(f, tag), H.factors))


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
# compile-time constants infers concretely, so the hot path is type stable.
# Construction, by contrast, is deliberately plain and dynamic (Dicts,
# Vectors, sorting) — it runs once, so clarity wins there.
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

# `map` over tuples, unrolled for any length. Base's `map` stops unrolling beyond 32 elements
# and then dispatches dynamically per element.
@generated function tuplemap(f, ts::Tuple...)
    n = fieldcount(ts[1])
    Expr(:tuple, (:(f($((:(ts[$k][$i]) for k in eachindex(ts))...))) for i in 1:n)...)
end

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

    # Atom bookkeeping, by name: which factor owns each atom, and where inside it.
    # The targets' bases are not checked here, that is done by `match_atoms` in the callers.
    atom_factor = Dict{Any,Int}()
    for (ci, f) in enumerate(sfactors)
        for a in atomic_factors(f)
            atom_factor[atomic_id(a)] = ci
        end
    end

    # Validate: every target atom exists in the source, no duplicates.
    seen = Set()
    for t in targets, a in atomic_factors(t)
        haskey(atom_factor, atomic_id(a)) || throw(ArgumentError("Each target atom must be an atom of the source space, got `$a` which is not."))
        atomic_id(a) ∈ seen && throw(ArgumentError("Each atom may appear in only one target, got `$a` in several."))
        push!(seen, atomic_id(a))
    end

    # Build the piece table: one piece per group of each target.
    # Each group must lie inside a single source factor. The integer coordinates of the
    # pieces are kept apart from their (heterogeneous) spaces, so that grouping and sorting
    # them works on a concretely typed vector.
    pieces = NamedTuple{(:factor, :target, :group_slot),NTuple{3,Int}}[]
    piece_spaces = Any[]
    piece_of = Dict{Tuple{Int,Int},Int}()  # (target, group_slot) -> piece
    for (ti, t) in enumerate(targets), (si, g) in enumerate(groups(t))
        ids = atom_ids(g)
        owners = unique(atom_factor[id] for id in ids)
        length(owners) == 1 ||
            throw(ArgumentError("A target group must lie within a single source factor, got `$g` which spans several."))
        push!(pieces, (factor=only(owners), target=ti, group_slot=si))
        push!(piece_spaces, g)
        piece_of[(ti, si)] = length(pieces)
    end

    # Per-factor view: build leaves and factor routes.
    # A factor's pieces stay in target and slot order (the sort is stable).
    order = sortperm(pieces, by=p -> p.factor)
    leaves = Any[]
    factor_routes = Any[]
    piece_slot = zeros(Int, length(pieces))  # piece -> slot within its factor
    k = 1
    for (ci, f) in enumerate(sfactors)
        first_piece = k
        while k <= length(order) && pieces[order[k]].factor == ci
            piece_slot[order[k]] = k - first_piece + 1
            k += 1
        end
        ps = view(order, first_piece:k-1)
        if isempty(ps)
            push!(leaves, Uncovered())
            push!(factor_routes, ())
            continue
        end
        spaces = Tuple(piece_spaces[n] for n in ps)
        inner = state_mapper(f, spaces)
        leaf = (length(ps) == 1 && only(spaces) == f) ? Passthrough(inner) : Fractured(inner)
        push!(leaves, leaf)
        push!(factor_routes, Tuple(PieceIndex(pieces[n].target, pieces[n].group_slot) for n in ps))
    end

    # Per-target view: the inverse routing, one entry per group slot.
    target_routes = Tuple(
        Tuple(
            begin
                n = piece_of[(ti, si)]
                PieceIndex(pieces[n].factor, piece_slot[n])
            end
            for si in eachindex(groups(t)))
        for (ti, t) in enumerate(targets))

    ProductSpaceMapper(Tuple(leaves), Tuple(factor_routes), target_routes, targets)
end

# ─── Hot path: split / combine ───────────────────────────────────────────────

function split_state(state::ProductState, sp::ProductSpaceMapper)
    # 1. Fracture each source factor into its pieces.
    pieces = tuplemap(split_pieces, sp.leaves, state.states)
    # 2. Assemble each target from its pieces.
    outstates = tuplemap(sp.target_routes, sp.target_spaces) do route, tspace
        gathered = tuplemap(ref -> getpiece(pieces, ref), route)
        only(first(combine_states(gathered, tspace)))
    end
    return (outstates,), (1,)
end

function combine_states(substates::Tuple, sp::ProductSpaceMapper)
    # 1. Take each target's state apart into its pieces (one per group).
    tpieces = tuplemap(substate_pieces, substates)
    # 2. Assemble each source factor from its pieces.
    states = tuplemap(sp.leaves, sp.factor_routes) do leaf, route
        combine_pieces(leaf, tuplemap(ref -> getpiece(tpieces, ref), route))
    end
    return (ProductState(states),), (1,)
end
combine_states(substates, sp::ProductSpaceMapper) = combine_states(Tuple(substates), sp)
# A vector of substates has one entry per target; the mapper type gives that number statically
function combine_states(substates::AbstractVector, sp::ProductSpaceMapper{<:Any,<:Any,<:Any,<:NTuple{N,Any}}) where {N}
    length(substates) == N || throw(ArgumentError("Expected $N substates, got $(length(substates))."))
    combine_states(ntuple(i -> substates[firstindex(substates)+i-1], Val(N)), sp)
end
combine_states(states::Tuple, ::ProductSpace{ProductState{B}}) where B = (ProductState{B}(B(states)),), (1,)

# ─── Phase factors ───────────────────────────────────────────────────────────

leaf_phase_factor(::Uncovered) = (s1, s2) -> 1
leaf_phase_factor(leaf::Union{Passthrough,Fractured}) = kron_phase_factor(leaf.mapper)

function kron_phase_factor(sp::ProductSpaceMapper)
    phase_fns = map(leaf_phase_factor, sp.leaves)
    function phase_factor(fullstate1, fullstate2)
        reduce(*, tuplemap((pf, s1, s2) -> pf(s1, s2), phase_fns, fullstate1.states, fullstate2.states))
    end
end
function phase_factor_u(sp::ProductSpaceMapper)
    # Build per-leaf phase-factor functions (Uncovered → nothing, others → recurse)
    leaf_pfus = map(sp.leaves) do leaf
        leaf isa Uncovered ? nothing : phase_factor_u(leaf.mapper)
    end

    function phase_factor(state)
        reduce(*, tuplemap((s, pfu) -> isnothing(pfu) ? 1 : pfu(s), state.states, leaf_pfus))
    end
end
