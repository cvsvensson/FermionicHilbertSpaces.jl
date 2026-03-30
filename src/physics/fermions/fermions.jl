

struct FermionicMode{L,S} <: AbstractAtomicHilbertSpace{FockNumber{UInt}}
    label::L
    symbolic_basis::S
end
fermionic_group(h::FermionicMode) = fermionic_group(h.symbolic_basis)
cluster_id(h::FermionicMode) = fermionic_group(h)
atomic_id(H::FermionicMode) = (H.symbolic_basis, H.label)

modes(H::FermionicMode) = (H.symbolic_basis[H.label],)
Base.:(==)(m1::FermionicMode, m2::FermionicMode) = m1.label == m2.label && m1.symbolic_basis == m2.symbolic_basis
Base.hash(m::FermionicMode, h::UInt) = hash(m.label, hash(m.symbolic_basis, h))
function Base.show(io::IO, c::FermionicMode)
    get(io, :compact, false) || return print(io, "FermionicMode(", c.symbolic_basis.name, "[", c.label, "])")
    print(io, c.symbolic_basis.name, "[", c.label, "]")
end
combine_states(states, ::AbstractAtomicHilbertSpace) = (only(states),), (1,)
combine_into_cluster(group::FermionicGroup, fermions) = all(f -> cluster_id(f) == group, fermions) ? FermionCluster(fermions, group) : throw(ArgumentError("Not all fermions belong to the same group"))
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

    FermionCluster(modes, only(unique(map(cluster_id, modes))))
end
maximum_particles(H::FermionCluster) = nbr_of_modes(H)
Base.:(==)(c1::FermionCluster, c2::FermionCluster) = c1.modes == c2.modes && c1.group == c2.group
Base.hash(c::FermionCluster, h::UInt) = hash(c.modes, hash(c.group, h))
basisstates(H::FermionCluster{F}) where F = TypedIterator{F}(Iterators.map(F, UnitRange{UInt64}(0, dim(H) - 1)))
basisstate(ind, ::FermionCluster{F}) where F = (F ∘ FockNumber)(ind - 1)
state_index(state::FockNumber, ::FermionCluster) = state.f + 1
function dim(H::FermionCluster)
    N = nbr_of_modes(H)
    N < 63 ? 2^N : BigInt(2)^N
end
atomic_factors(H::FermionCluster) = H.modes
nbr_of_modes(H::FermionCluster) = length(H.modes)
cluster_id(H::FermionCluster) = H.group
mode_ordering(H::FermionCluster) = H.mode_ordering
modes(H::FermionCluster) = H.modes
_find_position(f::FermionicMode, H::FermionCluster) = get(H.mode_ordering, f, 0)
_find_position(f::FermionicMode, H::FermionicMode) = f == H ? 1 : 0
_find_position(H::AbstractHilbertSpace, ordering::AbstractDict) = get(ordering, H, 0)
operators(H::FermionCluster) = fermions(H)
operators(H::FermionicMode) = fermions(H)
isconstrained(H::FermionCluster) = false
combine_states(states, H::FermionCluster{F}) where F = (catenate_fock_states(states, H.modes, F),), (1,)

state_mapper(H::FermionCluster, Hs::AbstractHilbertSpace) = state_mapper(H, (Hs,))
function state_mapper(H::FermionCluster, Hs)
    fermionpositions = [[_find_position(atom, H) for atom in atomic_factors(cluster)] for cluster in Hs]
    all(x -> x > 0, Iterators.flatten(fermionpositions)) || throw(ArgumentError("All subspaces must be part of the cluster"))
    FockMapper(Tuple(fermionpositions))
end

_truncate(items, max, edge) =
    length(items) <= max ? items : [items[1:edge]; "..."; items[end-edge+1:end]]

function _compact_fermionic_modes(io::IO, c::FermionCluster;
    max_groups=typemax(Int), edge_groups=3,
    max_labels_per_group=typemax(Int), edge_labels=3)

    isempty(c.modes) && return print(io, "FermionCluster()")

    # Group consecutive modes by symbolic basis name
    groups = Tuple{Any,Vector{Any}}[]
    for m in c.modes
        if !isempty(groups) && last(groups)[1] == m.symbolic_basis.name
            push!(last(groups)[2], m.label)
        else
            push!(groups, (m.symbolic_basis.name, Any[m.label]))
        end
    end

    fmt((name, labels)) = "$name[$(join(_truncate(labels, max_labels_per_group, edge_labels), ", "))]"
    print(io, "(", join(_truncate(map(fmt, groups), max_groups, edge_groups), ", "), ")")
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



struct CombineFockNumbersProcessor{T} end
function (processor::CombineFockNumbersProcessor{T})(full_state, spaces) where T
    catenate_fock_states(full_state, spaces, T)
end
# _init_results(spaces, ::CombineFockNumbersProcessor{T}) where T = T[]

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
            return (newfocknbr,), (zero(fermionstatistics),)
        end
        fermionstatistics *= jwstring(digitpos, newfocknbr)
        newfocknbr = op ⊻ newfocknbr
    end
    return (newfocknbr,), (fermionstatistics,)
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

FermionicMode(f::FermionSym) = FermionicMode(f.label, f.basis)
_find_position(f::FermionSym, H::AbstractHilbertSpace) = _find_position(FermionicMode(f), H)
_find_position(f::FermionSym, H::ProductSpace) = _find_position(FermionicMode(f), H)
hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector) = FermionCluster(map(l -> FermionicMode(a[l]), labels))
hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector, states::AbstractVector{<:AbstractBasisState}) = ConstrainedSpace(hilbert_space(a, labels), states)
hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector, constraint::AbstractConstraint) = tensor_product(map(l -> hilbert_space(a[l]), labels); constraint)

function hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector, constraint::ParityConservation{Missing})
    H = hilbert_space(a, labels)
    states = if constraint.allowed_parities == [-1, 1]
        basisstates(H)
    else
        valid_parity = only(constraint.allowed_parities)
        Iterators.filter(isequal(valid_parity) ∘ parity, basisstates(H))
    end
    block_space(H, states, parity)
end
function hilbert_space(a::SymbolicFermionBasis, labels::AbstractVector, constraint::NumberConservation{T,Missing,Missing}) where T
    H = hilbert_space(a, labels)
    N = nbr_of_modes(H)
    numbers = T === Missing ? (0:N) : constraint.total
    state_blocks = map(n -> fixed_particle_number_fockstates(N, n), numbers)
    dict = OrderedDict(zip(numbers, state_blocks))
    _block_space(H, dict)
end

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

    #The difference between left fermions and right fermions is conserved
    constraint = NumberConservation(-1:1, [Hl, Hr], [1, -1])
    Hcons = tensor_product((Hl, Hr); constraint)
    @test size(matrix_representation(lindbladian, Hcons), 1) == dim(Hcons)

    blocks = map(sectors(Hcons)) do Hsector
        matrix_representation(lindbladian, Hsector)
    end
    @test cat(blocks...; dims=(1, 2)) == matrix_representation(lindbladian, Hcons)
end

@testitem "Tensor product of FermionicMode and FermionCluster" begin
    import FermionicHilbertSpaces: constrain_space, FermionCluster
    @fermions a

    # FermionicMode: tensor product of two single modes gives a FermionCluster
    H1 = hilbert_space(a[1])
    H2 = hilbert_space(a[2])
    Hw = tensor_product([H1, H2])
    H3 = hilbert_space(a, 1:2)
    @test Hw == H3
    @test Hw isa FermionCluster
    @test dim(H1) * dim(H2) == dim(Hw)

    # FermionCluster: tensor product of two clusters gives a larger FermionCluster
    H1 = hilbert_space(a, 1:2)
    H2 = hilbert_space(a, 3:4)
    Hw = tensor_product([H1, H2])
    H3 = hilbert_space(a, 1:4)
    @test Hw == H3
    @test dim(H1) * dim(H2) == dim(Hw)

    # NumberConservation constrained FermionClusters
    H1 = constrain_space(hilbert_space(a, 1:2), NumberConservation(1))
    H2 = constrain_space(hilbert_space(a, 3:4), NumberConservation(1))
    Hw = tensor_product((H1, H2))
    H3 = constrain_space(hilbert_space(a, 1:4), NumberConservation(2))
    @test Set(basisstates(Hw)) == Set(basisstates(constrain_space(Hw, NumberConservation(2))))
    @test issubset(collect(basisstates(Hw)), collect(basisstates(H3)))
    @test dim(H1) * dim(H2) == dim(Hw)
end



@testitem "Fermionic tensor product properties" begin
    # Properties from J. Phys. A: Math. Theor. 54 (2021) 393001
    # Eq. 16
    using Random, Base.Iterators, LinearAlgebra
    import FermionicHilbertSpaces: embedding_unitary, project_on_parity, project_on_parities

    @fermions a
    Random.seed!(3)
    N = 8
    rough_size = 4
    fine_size = 2
    rough_partitions = sort.(collect(partition(randperm(N), rough_size)))
    # divide each part of rough partition into finer partitions
    fine_partitions = map(rough_partition -> sort.(collect(partition(shuffle(rough_partition), fine_size))), rough_partitions)
    H = hilbert_space(a, 1:N)
    Hs_rough = [hilbert_space(a, r_p) for r_p in rough_partitions]
    Hs_fine = map(f_p_list -> Base.Fix1(hilbert_space, a).(f_p_list), fine_partitions)

    ops_rough = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    ops_fine = map(f_p_list -> [rand(ComplexF64, 2^length(f_p), 2^length(f_p)) for f_p in f_p_list], fine_partitions)

    # Associativity (Eq. 16)
    rhs = generalized_kron(reduce(vcat, ops_fine), reduce(vcat, Hs_fine), H)
    finetensor_products = [generalized_kron(ops_vec, cs_vec, c_rough) for (ops_vec, cs_vec, c_rough) in zip(ops_fine, Hs_fine, Hs_rough)]
    lhs = generalized_kron(finetensor_products, Hs_rough, H)
    @test lhs ≈ rhs

    physical_ops_rough = [project_on_parity(op, H, 1) for (op, H) in zip(ops_rough, Hs_rough)]

    # Eq. 18
    As = ops_rough
    Bs = map(r_p -> rand(ComplexF64, 2^length(r_p), 2^length(r_p)), rough_partitions)
    lhs = tr(generalized_kron(As, Hs_rough, H)' * generalized_kron(Bs, Hs_rough, H))
    rhs = mapreduce((A, B) -> tr(A' * B), *, As, Bs)
    @test lhs ≈ rhs

    # Fermionic embedding

    # Eq. 19 
    As_modes = [rand(ComplexF64, 2, 2) for _ in 1:N]
    ξ = vcat(fine_partitions...)
    ξbases = vcat(Hs_fine...)
    modebases = [hilbert_space(a[j]) for j in 1:N]
    lhs = prod(j -> embed(As_modes[j], modebases[j], H), 1:N)
    rhs_ordered_prod(X, basis) = mapreduce(j -> Matrix(embed(As_modes[j], modebases[j], basis)), *, X)
    rhs = generalized_kron([rhs_ordered_prod(X, H) for (X, H) in zip(ξ, ξbases)], ξbases, H)
    @test lhs ≈ rhs

    # Associativity (Eq. 21)
    @test embed(embed(ops_fine[1][1], Hs_fine[1][1], Hs_rough[1]), Hs_rough[1], H) ≈ embed(ops_fine[1][1], Hs_fine[1][1], H)
    @test all(map(Hs_rough, Hs_fine, ops_fine) do cr, cfs, ofs
        all(map(cfs, ofs) do cf, of
            embed(embed(of, cf, cr), cr, H) ≈ embed(of, cf, H)
        end)
    end)

    ## Eq. 22
    HX = Hs_rough[1]
    Ux = embedding_unitary(Hs_rough, H)
    A = ops_rough[1]
    @test Ux !== I
    @test embed(A, HX, H) ≈ Ux * embed(A, HX, H; phase_factors=false) * Ux'
    # Eq. 93
    @test tensor_product(physical_ops_rough, Hs_rough, H) ≈ Ux * generalized_kron(physical_ops_rough, Hs_rough, H; phase_factors=false) * Ux'

    # Eq. 23
    X = rough_partitions[1]
    HX = Hs_rough[1]
    A = ops_rough[1]
    B = rand(ComplexF64, 2^length(X), 2^length(X))
    #Eq 5a and 5br are satisfied also when embedding matrices in larger subsystems
    @test embed(A, HX, H)' ≈ embed(A', HX, H)
    @test embed(A, HX, H; phase_factors=false) * embed(B, HX, H; phase_factors=false) ≈ embed(A * B, HX, H; phase_factors=false)
    for cmode in modebases
        #Eq 5bl
        local A = rand(ComplexF64, 2, 2)
        local B = rand(ComplexF64, 2, 2)
        @test embed(A, cmode, H) * embed(B, cmode, H) ≈ embed(A * B, cmode, H)
    end

    # Ordered product of embeddings

    # Eq. 31
    A = ops_rough[1]
    X = rough_partitions[1]
    Xbar = setdiff(1:N, X)
    HX = Hs_rough[1]
    HXbar = hilbert_space(a, Xbar)
    corr = embed(A, HX, H)
    @test corr ≈ generalized_kron([A, I], [HX, HXbar], H) ≈ tensor_product([A, I], [HX, HXbar], H) ≈ tensor_product([I, A], [HXbar, HX], H)

    # Eq. 32
    @test tensor_product(As_modes, modebases, H) ≈ generalized_kron(As_modes, modebases, H)

    ## Fermionic partial trace

    # Eq. 36
    #X = rough_partitions[1]
    A = ops_rough[1]
    B = rand(ComplexF64, dim(H), dim(H))
    HX = Hs_rough[1]
    lhs = tr(embed(A, HX => H)' * B)
    rhs = tr(A' * partial_trace(B, H => HX))
    @test lhs ≈ rhs

    # Eq. 38 (using A, X, HX, HXbar from above)
    B = rand(ComplexF64, 2^length(Xbar), 2^length(Xbar))
    Hs = [HX, HXbar]
    ops = [A, B]
    @test partial_trace(generalized_kron(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(ops, Hs, H), H, HX) ≈ partial_trace(tensor_product(reverse(ops), reverse(Hs), H), H, HX) ≈ A * tr(B)

    # Eq. 39
    A = rand(ComplexF64, 2^N, 2^N)
    X = fine_partitions[1][1]
    Y = rough_partitions[1]
    HX = Hs_fine[1][1]
    HY = Hs_rough[1]
    HZ = H
    Z = 1:N
    rhs = partial_trace(A, HZ, HX)
    lhs = partial_trace(partial_trace(A, HZ, HY), HY, HX)
    @test lhs ≈ rhs

    # Eq. 41
    HY = H
    @test partial_trace(A', HY, HX) ≈ partial_trace(A, HY, HX)'

    # Eq. 95
    ξ = rough_partitions
    Asphys = physical_ops_rough
    Bs = map(X -> rand(ComplexF64, 2^length(X), 2^length(X)), ξ)
    Bsphys = [project_on_parity(B, H, 1) for (B, H) in zip(Bs, Hs_rough)]
    lhs1 = tensor_product(Asphys, Hs_rough, H) * tensor_product(Bsphys, Hs_rough, H)
    rhs1 = tensor_product(Asphys .* Bsphys, Hs_rough, H)
    @test lhs1 ≈ rhs1
    @test tensor_product(Asphys, Hs_rough, H)' ≈ tensor_product(adjoint.(Asphys), Hs_rough, H)


    ## Unitary equivalence between tensor_product and kron
    ops = reduce(vcat, ops_fine)
    Hs = reduce(vcat, Hs_fine)
    physical_ops = [project_on_parity(op, H, 1) for (op, H) in zip(ops, Hs)]
    # Eq. 93 implies that the unitary equivalence holds for the physical operators
    @test svdvals(Matrix(tensor_product(physical_ops, Hs, H))) ≈ svdvals(Matrix(generalized_kron(physical_ops, Hs, H; phase_factors=false)))
    # However, it is more general. The unitary equivalence holds as long as all except at most one of the operators has a definite parity:

    numberops = map(numberoperator, Hs)
    Uemb = embedding_unitary(Hs, H)
    fine_partition = reduce(vcat, fine_partitions)
    for parities in Base.product([[-1, 1] for _ in 1:length(Hs)]...)
        projected_ops = [project_on_parity(op, H, p) for (op, H, p) in zip(ops, Hs, parities)] # project on local parity
        opsk = [[projected_ops[1:k-1]..., ops[k], projected_ops[k+1:end]...] for k in 1:length(ops)] # switch out one operator of definite parity for an operator of indefinite parity
        embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
        kron_prods = [generalized_kron(ops, Hs, H; phase_factors=false) for ops in opsk]

        @test all(svdvals(Matrix(op1)) ≈ svdvals(Matrix(op2)) for (op1, op2) in zip(embedding_prods, kron_prods))
    end

    # Explicit construction of unitary equivalence in case of all even (except one) 
    function phase(k, f)
        fines = collect(Iterators.flatten(Hs_fine))
        Xkmask = FermionicHilbertSpaces.focknbr_from_site_labels(fines[k], H)
        iseven(count_ones(f & Xkmask)) && return 1
        phase = 1
        for r in 1:k-1
            Xrmask = FermionicHilbertSpaces.focknbr_from_site_labels(fines[r], H)
            phase *= (-1)^(count_ones(f & Xrmask))
        end
        return phase
    end
    opsk = [[physical_ops[1:k-1]..., ops[k], physical_ops[k+1:end]...] for k in 1:length(ops)]
    unitaries = [Diagonal([phase(k, f) for f in basisstates(H)]) * Uemb for k in 1:length(opsk)]
    embedding_prods = [tensor_product(ops, Hs, H) for ops in opsk]
    kron_prods = [generalized_kron(ops, Hs, H; phase_factors=false) for ops in opsk]
    @test all(op1 ≈ U * op2 * U for (op1, op2, U) in zip(embedding_prods, kron_prods, unitaries))

end


@testitem "Tensor product of fermionic operators" begin
    using Random, LinearAlgebra
    import SparseArrays: SparseMatrixCSC
    import FermionicHilbertSpaces: fermions
    Random.seed!(1234)
    @fermions f
    for qn in [NoSymmetry(), ParityConservation(), NumberConservation()]
        H1 = hilbert_space(f, 1:1, qn)
        H2 = hilbert_space(f, 1:3, qn)
        @test_throws ArgumentError tensor_product(H1, H2)
        H2 = hilbert_space(f, 2:3, qn)
        H3 = hilbert_space(f, 1:3, qn)
        H3w = tensor_product(H1, H2)
        @test H3w == tensor_product((H1, H2)) == tensor_product([H1, H2])
        Hs = [H1, H2]
        b1 = fermions(H1)
        b2 = fermions(H2)
        b3 = fermions(H3w)

        #test that they keep sparsity
        @test typeof(tensor_product((b1[1], b2[2]), Hs => H3)) == typeof(b1[1])
        @test typeof(generalized_kron((b1[1], b2[2]), Hs, H3)) == typeof(b1[1])
        @test typeof(tensor_product((b1[1], I), Hs => H3)) == typeof(b1[1])
        @test typeof(generalized_kron((b1[1], I), Hs, H3)) == typeof(b1[1])
        @test tensor_product((I, I), Hs => H3) isa SparseMatrixCSC
        @test generalized_kron((I, I), Hs, H3) isa SparseMatrixCSC

        # Test zero-mode error
        @test_throws ArgumentError hilbert_space(f, 1:0, qn)
    end

    #Test basis compatibility
    H1 = hilbert_space(f, 1:2, ParityConservation())
    H2 = hilbert_space(f, 2:4, ParityConservation())
    @test_throws ArgumentError tensor_product(H1, H2)
end

