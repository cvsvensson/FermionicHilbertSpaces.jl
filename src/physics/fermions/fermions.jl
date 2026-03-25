

struct FermionicMode{L,S} <: AbstractAtomicHilbertSpace{FockNumber{UInt}}
    label::L
    symbolic_basis::S
end
symbolic_group(h::FermionicMode) = fermionic_group(h)
fermionic_group(h::FermionicMode) = fermionic_group(h.symbolic_basis)
modes(H::FermionicMode) = (H.symbolic_basis[H.label],)
Base.:(==)(m1::FermionicMode, m2::FermionicMode) = m1.label == m2.label && m1.symbolic_basis == m2.symbolic_basis
Base.hash(m::FermionicMode, h::UInt) = hash(m.label, hash(m.symbolic_basis, h))
function Base.show(io::IO, c::FermionicMode)
    get(io, :compact, false) || return print(io, "FermionicMode(", c.symbolic_basis.name, "[", c.label, "])")
    print(io, c.symbolic_basis.name, "[", c.label, "]")
end
combine_states(states, ::AbstractAtomicHilbertSpace) = ((only(states), 1),)
combine_into_cluster(group::FermionicGroup, fermions) = all(f -> symbolic_group(f) == group, fermions) ? FermionCluster(fermions, group) : throw(ArgumentError("Not all fermions belong to the same group"))
function basisstate(n::Int, H::FermionicMode)
    n == 1 && return FockNumber(zero(default_fock_representation(1)))
    n == 2 && return FockNumber(one(default_fock_representation(1)))
    throw(ArgumentError("Invalid state index $n for FermionicMode"))
end
basisstates(::FermionicMode) = (FockNumber(zero(default_fock_representation(Val(1)))), FockNumber(one(default_fock_representation(Val(1)))))
maximum_particles(::FermionicMode) = 1
# state_index(s::FockNumber{Bool}, H::FermionicMode) = s.f + 1
function state_index(s::FockNumber, H::FermionicMode)
    iszero(s.f) && return 1
    isone(s.f) && return 2
    throw(ArgumentError("Invalid state $s for FermionicMode"))
end
atomic_factors(H::FermionicMode) = (H,)

label(H::FermionicMode) = H.label
nbr_of_modes(H::FermionicMode) = 1
dim(H::FermionicMode) = 2

abstract type AbstractFermionicClusterHilbertSpace{B} <: AbstractClusterHilbertSpace{B} end
struct FermionCluster{F,L} <: AbstractFermionicClusterHilbertSpace{F}
    modes::Vector{L}
    mode_ordering::OrderedDict{L,Int}
    group::FermionicGroup
    function FermionCluster(modes::Union{<:AbstractVector{L},<:NTuple{N,L}}, group::FermionicGroup, ::Type{F}=FockNumber{default_fock_representation(length(modes))}) where {F,L<:FermionicMode,N}
        length(modes) == 0 && throw(ArgumentError("Cannot create a FermionCluster with no modes"))
        length(modes) == 1 && return only(modes)
        mode_ordering = OrderedDict{L,Int}(m => i for (i, m) in enumerate(modes))
        length(mode_ordering) == length(modes) || throw(ArgumentError("Duplicate modes in fermionic group"))
        new{F,L}(collect(modes), mode_ordering, group)
    end
end
FermionCluster(mode::FermionicMode) = mode
function FermionCluster(modes::Union{<:AbstractVector{F},NTuple{N,F}}) where {N,F<:FermionicMode}

    FermionCluster(modes, only(unique(map(symbolic_group, modes))))
end
maximum_particles(H::FermionCluster) = nbr_of_modes(H)
Base.:(==)(c1::FermionCluster, c2::FermionCluster) = c1.modes == c2.modes && c1.group == c2.group
Base.hash(c::FermionCluster, h::UInt) = hash(c.modes, hash(c.group, h))
basisstates(H::FermionCluster{F}) where F = Iterators.map(F ∘ FockNumber, UnitRange{UInt64}(0, dim(H) - 1))
basisstate(ind, ::FermionCluster{F}) where F = (F ∘ FockNumber)(ind - 1)
state_index(state::FockNumber, ::FermionCluster) = state.f + 1
dim(H::FermionCluster) = 2^nbr_of_modes(H)
atomic_factors(H::FermionCluster) = H.modes
nbr_of_modes(H::FermionCluster) = length(H.modes)
symbolic_group(H::FermionCluster) = H.group
mode_ordering(H::FermionCluster) = H.mode_ordering
modes(H::FermionCluster) = H.modes
_find_position(f::FermionicMode, H::FermionCluster) = get(H.mode_ordering, f, 0)
_find_position(f::FermionicMode, H::FermionicMode) = f == H ? 1 : 0
_find_position(H::AbstractHilbertSpace, ordering::AbstractDict) = get(ordering, H, 0)
operators(H::FermionCluster) = fermions(H)
operators(H::FermionicMode) = fermions(H)

function fermion_submodes(sub::Vector{L}, H::FermionCluster{<:Any,L}) where L
    return sub
end
function fermion_submodes(sub::FermionCluster{<:Any,L}, H::FermionCluster{<:Any,L}) where L
    return sub.modes
end
function fermion_submodes(sub::F, H::FermionCluster{<:Any,F}) where F
    return (sub,)
end
function fermion_submodes(sub::Vector{T}, H::FermionCluster{<:Any,<:FermionicMode{T}}) where T
    bases = unique!(map(m -> m.symbolic_basis, atomic_factors(H)))
    length(bases) == 1 || throw(ArgumentError("Specifying only labels is ambiguous when the cluster contains modes with different symbolic bases"))
    basis = only(bases)
    return map(s -> FermionicMode(basis[s]), sub)
end
function fermion_submodes(sub::Vector{<:FermionSym{T}}, H::FermionCluster{<:Any,<:FermionicMode{T}}) where T
    bases = map(m -> m.symbolic_basis, atomic_factors(H))
    modes = map(FermionicMode, sub)
    for m in modes
        m in H.modes || throw(ArgumentError("Mode $m is not part of the cluster $H"))
    end
    return modes
end

function subregion(Hsub, H::FermionCluster)
    submodes = fermion_submodes(Hsub, H)
    positions = map(f -> _find_position(f, H), submodes)
    all(x -> x > 0, positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    issorted(positions) || throw(ArgumentError("The modes $(modes(Hsub)) are not an ordered subsystem of the Hilbert space $(H)"))
    Hsub
end
isconstrained(H::FermionCluster) = false
combine_states(states, H::FermionCluster{F}) where F = ((catenate_fock_states(states, H.modes, F), 1),)

state_splitter(H::FermionCluster, Hs::AbstractHilbertSpace) = state_splitter(H, (Hs,))
function state_splitter(H::FermionCluster, Hs)
    fermionpositions = [[_find_position(atom, H) for atom in atomic_factors(cluster)] for cluster in Hs]
    all(x -> x > 0, Iterators.flatten(fermionpositions)) || throw(ArgumentError("All subspaces must be part of the cluster"))
    FockMapper(Tuple(fermionpositions))
end

function _compact_fermionic_modes(io::IO, c::FermionCluster)
    isempty(c.modes) && return print(io, "FermionCluster()")
    i = firstindex(c.modes)
    N = lastindex(c.modes)
    print(io, "(")
    while i <= N
        m = c.modes[i]
        name = m.symbolic_basis.name
        print(io, name, "[", m.label)
        i += 1
        while i <= N && name == c.modes[i].symbolic_basis.name
            print(io, ", ", c.modes[i].label)
            i += 1
        end
        print(io, "]")
        i <= N && print(io, ", ")
    end
    print(io, ")")
end

function Base.show(io::IO, c::FermionCluster)
    if get(io, :compact, false)
        _compact_fermionic_modes(io, c)
    else
        print(io, "$(dim(c))-dimensional FermionCluster\n")
        print(io, "Modes: ")
        _compact_fermionic_modes(io, c)
    end
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
partial_trace_phase_factor(f1, f2, H::FermionCluster) = phase_factor_f(f1, f2, nbr_of_modes(H))


function branch_constraint(constraint::ParityConservation, spaces)
    possible_numbers = ismissing(constraint.subspaces) ? (0:sum(maximum_particles, spaces)) : (0:sum(nbr_of_modes, constraint.subspaces))
    allowed_numbers = filter(n -> any(p -> p == (-1)^n, constraint.allowed_parities), possible_numbers)
    unweighted_number_branch_constraint(allowed_numbers, constraint.subspaces, spaces)
end

function branch_constraint(constraint::NumberConservation{T,H,W}, spaces) where {T,H,W}
    subspaces = H === Missing ? spaces : constraint.subspaces
    if W === Missing
        total = T === Missing ? (0:sum(maximum_particles, subspaces)) : constraint.total
        return unweighted_number_branch_constraint(total, subspaces, spaces)
    end
    T === Missing && throw(ArgumentError("Total particle number must be specified when using weighted number branch constraint"))
    weighted_number_branch_constraint(constraint.total, constraint.weights, subspaces, spaces)
end


struct CombineFockNumbersProcessor{T} end
function (processor::CombineFockNumbersProcessor{T})(full_state, spaces) where T
    catenate_fock_states(full_state, spaces, T)
end
_init_results(spaces, ::CombineFockNumbersProcessor{T}) where T = T[]

focknbr_from_site_label(mode::FermionicMode, H::FermionCluster) = focknbr_from_site_index(_find_position(mode, H))
focknbr_from_site_labels(Hsub::FermionCluster, H::FermionCluster) = mapreduce(Base.Fix2(focknbr_from_site_label, H), |, modes(Hsub), init=FockNumber(zero(default_fock_representation(nbr_of_modes(H)))))


# _precomputation_before_operator_application(op::NCMul, space::AbstractHilbertSpace{B}) where {B<:FockNumber} = map(op -> _find_position(op, space), op.factors)
_precomputation_before_operator_application(op::NCMul, space::Union{<:FermionicMode,<:FermionCluster{B}}) where {B<:FockNumber} = map(op -> _find_position(op, space), op.factors)
function apply_local_operators(op::NCMul, state::FockNumber{I}, space::AbstractHilbertSpace, fermionpositions) where I
    factors = op.factors
    newfocknbr = state
    fermionstatistics = op.coeff
    for (op, digitpos) in Iterators.reverse(zip(factors, fermionpositions))
        dagger = op.creation
        op = one(I) << (digitpos - 1)
        occupied = !iszero(op & newfocknbr)
        if dagger == occupied
            return ((newfocknbr, 0),)
        end
        fermionstatistics *= jwstring(digitpos, newfocknbr)
        newfocknbr = op ⊻ newfocknbr
    end
    return ((newfocknbr, fermionstatistics),)
end


"""
    fermion_sparse_matrix(fermion_number, H::AbstractHilbertSpace)

Constructs a sparse matrix of size representing a fermionic annihilation operator at bit position `fermion_number` on the Hilbert space H. 
"""
function fermion_sparse_matrix(fermion_number, H::AbstractHilbertSpace{<:AbstractFockState})
    sparse_fockoperator(Base.Fix1(removefermion, fermion_number), H)
end


function sparse_fockoperator(op, H::AbstractHilbertSpace{<:AbstractFockState})
    fs = basisstates(H)
    N = length(fs)
    amps = Int[]
    ininds = Int[]
    outinds = Int[]
    sizehint!(amps, N)
    sizehint!(ininds, N)
    sizehint!(outinds, N)
    for f in fs
        n = state_index(f, H)
        newfockstate, amp = op(f)
        if !iszero(amp)
            push!(amps, amp)
            push!(ininds, n)
            push!(outinds, state_index(newfockstate, H))
        end
    end
    return SparseArrays.sparse!(outinds, ininds, amps, N, N)
end


@testitem "Parity and number operator" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: parityoperator, numberoperator, fermion_sparse_matrix
    numopvariant(H) = sum(l -> fermion_sparse_matrix(l, H)' * fermion_sparse_matrix(l, H), 1:2)
    @fermions f
    H = hilbert_space(f, 1:2)
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    H = hilbert_space(f, 1:2, ParityConservation())
    @test parityoperator(H) == Diagonal([-1, -1, 1, 1])
    @test numberoperator(H) == Diagonal([1, 1, 0, 2]) == numopvariant(H)

    H = hilbert_space(f, 1:2, NumberConservation())
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    ## Truncated Hilbert space
    states = map(FockNumber, 0:2)
    H = hilbert_space(f, 1:2, states)
    @test parityoperator(H) == Diagonal([1, -1, -1])
    @test numberoperator(H) == Diagonal([0, 1, 1])

    states = map(FockNumber, 2:2)
    H = hilbert_space(f, 1:2, states)
    @test parityoperator(H) == Diagonal([-1])
    @test numberoperator(H) == Diagonal([1])
end

"""
    fermions(H)

Return a dictionary of fermionic annihilation operators for the Hilbert space `H`.
"""
function fermions(H)
    M = length(atomic_factors(H))
    labelvec = map(label, atomic_factors(H))
    reps = [fermion_sparse_matrix(n, H) for n in 1:M]
    OrderedDict(zip(labelvec, reps))
end

"""
    majoranas(fermions)

Return a dictionary of Majorana operators for the Hilbert space `H`.
"""
function majoranas(fermions::AbstractDict, majlabels=(:-, :+))
    labels = Base.product(collect(keys(fermions)), majlabels)
    fs = values(fermions)
    majA = map(f -> f + f', fs)
    majB = map(f -> 1im * (f - f'), fs)
    majs = vcat(majA, majB)
    OrderedDict(zip(labels, majs))
end

@testitem "Canonical anticommutation relations (CAR)" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: fermions
    @fermions f
    for qn in [NoSymmetry(), ParityConservation(), NumberConservation()]
        c = fermions(hilbert_space(f, 1:2, qn))
        @test c[1] * c[1] == 0I
        @test c[1]' * c[1] + c[1] * c[1]' == I
        @test c[1]' * c[2] + c[2] * c[1]' == 0I
        @test c[1] * c[2] + c[2] * c[1] == 0I
    end
end

@testitem "Majorana operators" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: majoranas, fermions
    @fermions f
    H = hilbert_space(f, 1:2)
    γ = majoranas(fermions(H))
    # There should be 4 Majorana operators for 2 modes
    @test length(γ) == 4
    # Test Hermiticity: γ₁ = c + c†, γ₂ = i(c - c†) are Hermitian
    for op in values(γ)
        @test op ≈ op'
    end
    # Test anticommutation: {γ_i, γ_j} = 2δ_{ij}I
    γops = values(γ)
    for γ1 in γops, γ2 in γops
        anticom = γ1 * γ2 + γ2 * γ1
        @test anticom ≈ 2I * (γ1 == γ2)
    end
end


_sym_space_match(basis::SymbolicFermionBasis, space::AbstractFockHilbertSpace) = true
_sym_space_match(basis::SymbolicFermionBasis, space::AbstractHilbertSpace) = false

label(H::AbstractFockHilbertSpace) = only(mode_ordering(H))
function _sym_space_match(sym, space::AbstractHilbertSpace)
    label(sym) == label(space)
end
function _sym_space_match(sym::AbstractFermionSym, space::AbstractFockHilbertSpace)
    label(sym) in keys(space)
end
FermionicMode(f::FermionSym) = FermionicMode(f.label, f.basis)
_find_position(f::FermionSym, H::AbstractHilbertSpace) = _find_position(FermionicMode(f), H)
_find_position(f::FermionSym, H::ProductSpace) = _find_position(FermionicMode(f), H)
hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector) = FermionCluster(map(l -> FermionicMode(a[l]), labels))
hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector, constraint) = constrain_space(hilbert_space(a, labels), constraint)

issubsystem(Hsub::AbstractHilbertSpace, H::FermionCluster) = isorderedsubsystem(Hsub, H)

@testitem "Commuting fermionic operators: Lindbladian and number conservation" begin
    @fermions c_l
    @fermions c_r

    hamiltonian(c) = c[1]' * c[1]
    jump_op(c) = c[1]'
    lindbladian = let Hl = hamiltonian(c_l), Hr = hamiltonian(c_r), Ll = jump_op(c_l), Lr = jump_op(c_r)
        1im * (Hl - Hr) + Ll * Lr - 0.5 * (Ll' * Ll + Lr' * Lr)
    end

    Hl = hilbert_space(c_l[1])
    Hr = hilbert_space(c_r[1])
    Hlr = tensor_product((Hl, Hr))
    mat = matrix_representation(lindbladian, Hlr)
    Hcons = constrain_space(Hlr, NumberConservation(-1:1, [Hl, Hr], [1, -1])) # The difference between left fermions and right fermions is conserved
    @test size(matrix_representation(lindbladian, Hcons), 1) == dim(Hcons)

    blocks = map(sectors(Hcons)) do Hsector
        matrix_representation(lindbladian, Hsector)
    end
    @test cat(blocks...; dims=(1, 2)) == matrix_representation(lindbladian, Hcons)
end