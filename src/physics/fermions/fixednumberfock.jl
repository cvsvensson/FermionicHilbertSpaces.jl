struct FixedNumberFockState{N} <: AbstractFockState
    sites::NTuple{N,Int}
    FixedNumberFockState{N}(sites::NTuple{N,Int}) where N = new{N}(TupleTools.sort(sites))
end
FixedNumberFockState(sites::NTuple{N,Int}) where N = FixedNumberFockState{N}(TupleTools.sort(sites))
Base.:(==)(f1::FixedNumberFockState, f2::FixedNumberFockState) = f1.sites == f2.sites
Base.hash(f::FixedNumberFockState, h::UInt) = hash(f.sites, h)

const SingleParticleState = FixedNumberFockState{1}
SingleParticleState(site::Int) = FixedNumberFockState((site,))
function jwstring_left_bool(site, f::FixedNumberFockState)
    sign = false
    for s in f.sites
        if s < site
            sign ⊻= true
        end
    end
    return sign
end
function jwstring_right_bool(site, f::FixedNumberFockState)
    sign = false
    for s in f.sites
        if s > site
            sign ⊻= true
        end
    end
    return sign
end
FockNumber(f::FixedNumberFockState) = focknbr_from_site_indices(f.sites)
FockNumber{I}(f::FixedNumberFockState) where I = FockNumber{I}(focknbr_from_site_indices(f.sites, I))
FixedNumberFockState{N}(f::FixedNumberFockState{M}) where {M,N} = FixedNumberFockState{N}((f.sites))
FixedNumberFockState(f::FockNumber) = FixedNumberFockState{count_ones(f)}(f)
function FixedNumberFockState{N}(f::FockNumber) where N
    site = 1
    count = 0
    sites = Int[]
    while count < N
        if _bit(f, site)
            count += 1
            push!(sites, site)
        end
        site += 1
    end
    FixedNumberFockState{N}(Tuple(sites))
end

function insert_bits(f::FixedNumberFockState, positions)
    FixedNumberFockState(insert_bits(FockNumber(f), positions))
end
Base.:(|)(f1::FixedNumberFockState, f2::FixedNumberFockState) = FixedNumberFockState((f1.sites..., f2.sites...))
Base.:(|)(f1::FixedNumberFockState, f2::FockNumber) = FixedNumberFockState(FockNumber(f1) | f2)

_bit(f::FixedNumberFockState, k) = k in f.sites
function substate(siteindices, f::FixedNumberFockState)
    subsites = Int[]
    for (n, site) in enumerate(siteindices)
        site in f.sites && push!(subsites, n)
    end
    return FixedNumberFockState(Tuple(subsites))
end

Base.isless(a::FixedNumberFockState, b::FixedNumberFockState) = a.sites < b.sites

@testitem "FixedNumberFockState" begin
    import FermionicHilbertSpaces: jwstring_left, jwstring_right, FixedNumberFockState, FockNumber, SingleParticleState, _bit, substate, state_mapper, combine_states
    @fermions a
    f = FixedNumberFockState((1, 3, 5))
    f2 = FockNumber(f)
    @test f == FixedNumberFockState(f2)
    @test jwstring_left(2, f) == -1 == jwstring_left(2, f2)
    @test jwstring_left(4, f) == 1 == jwstring_left(4, f2)
    @test jwstring_right(2, f) == 1 == jwstring_right(2, f2)
    @test jwstring_right(4, f) == -1 == jwstring_right(4, f2)

    # test _bit
    @test _bit(f, 1) == true == _bit(f2, 1)
    @test _bit(f, 2) == false == _bit(f2, 2)
    @test _bit(f, 3) == true == _bit(f2, 3)
    @test _bit(f, 4) == false == _bit(f2, 4)
    @test _bit(f, 5) == true == _bit(f2, 5)

    # test substate
    @test substate((1,), FixedNumberFockState((1,))) == FixedNumberFockState((1,))
    @test substate((2,), FixedNumberFockState((2,))) == FixedNumberFockState((1,))
    @test substate((2,), FixedNumberFockState((1,))) == FixedNumberFockState(())
    @test substate((1, 2, 3), FixedNumberFockState((1, 3, 5))) == FixedNumberFockState((1, 3))
    @test substate((5, 10, 100, 20), FixedNumberFockState((1, 10, 100))) == FixedNumberFockState((2, 3))

    # test permutation and FockMapper
    H1 = hilbert_space(a, 1:2)
    H2 = hilbert_space(a, 3:4)
    H12 = hilbert_space(a, [1, 3, 2, 4])
    fm = state_mapper(H12, (H1, H2))
    for (f1, f2) in Base.product(basisstates(H1), basisstates(H2))
        f1fix = FixedNumberFockState(f1)
        f2fix = FixedNumberFockState(f2)
        f12 = only(first(combine_states((f1, f2), fm)))
        f12fix = only(first(combine_states((f1fix, f2fix), fm)))
        @test f12 == FockNumber(f12fix)
    end

    h = a[1]' * a[2] + 1im * a[1]' * a[2]' + hc
    H = hilbert_space(a, 1:2, FermionicHilbertSpaces.SingleParticleState.(1:3))
    @test_throws MethodError matrix_representation(h, H)

    N = 10
    H = hilbert_space(a, 1:N, SingleParticleState.(1:N))
    Hf = hilbert_space(a, 1:N, NumberConservation(1))
    @test length(basisstates(H)) == length(basisstates(Hf)) == N
    @test map(FockNumber, basisstates(H)) == basisstates(Hf)
    op = sum(rand() * a[k1]'a[k2] + rand(ComplexF64) * a[k1]a[k2]' for (k1, k2) in Base.product(1:N, 1:N))
    @test matrix_representation(op, H) ≈ matrix_representation(op, Hf)
end

# _precomputation_before_operator_application(op::NCMul, space::SingleParticleHilbertSpace) = (println("ASD"); map(op -> _find_position(op, space), op.factors))
function apply_local_operators(op::NCMul, f::FixedNumberFockState, H::AbstractHilbertSpace, sites; kwargs...)
    # sites = Iterators.map(op -> _find_position(op, H), reverse(ops))
    daggers = Iterators.map(op -> op.creation, reverse(op.factors))
    state, amp = togglefermions(Iterators.reverse(sites), daggers, f)
    return (state,), (amp * op.coeff,)
end
function togglefermions(sites, daggers, f::FixedNumberFockState)
    # Check if operation results in vacuum or not,
    # short circuiting if so to avoid allocating fsites
    for site in sites
        last_dagger = false
        for s in f.sites
            if s == site
                last_dagger = true
            end
        end
        for (s, dagger) in zip(sites, daggers)
            if s == site
                if dagger == last_dagger
                    return f, false
                end
                last_dagger = dagger
            end
        end
    end

    fsites = collect(f.sites) # Lots of allocations
    fermionstatistics = 1
    for (site, dagger) in zip(sites, daggers)
        if dagger
            if site in fsites
                return f, false
            end
            fsites = push!(fsites, site)
        else
            if !(site in fsites)
                return f, false
            end
            deleteat!(fsites, findfirst(isequal(site), fsites))
        end
        sign = 1
        for s in fsites
            if s < site
                sign *= -1
            end
        end
        fermionstatistics *= sign
    end
    sort!(fsites)
    return FixedNumberFockState(Tuple(fsites)), fermionstatistics
end
Base.zero(::FixedNumberFockState) = FixedNumberFockState(())
function concatenate((lastf, lastwidth)::Tuple{FixedNumberFockState,Int}, (f, width)::Tuple{FixedNumberFockState,Int})
    return (FixedNumberFockState((lastf.sites..., (lastwidth .+ f.sites)...)), lastwidth + width)
end
function permute(f::FixedNumberFockState, permutation::BitPermutations.AbstractBitPermutation)
    p = Vector(permutation')
    return FixedNumberFockState(map(s -> p[s], f.sites))
end

struct SingleParticleHilbertSpace{H}
    parent::H
    function SingleParticleHilbertSpace(f::SymbolicFermionBasis, labels)
        states = [SingleParticleState(i) for (i, label) in enumerate(labels)]
        H = hilbert_space(f, labels, states)
        return new{typeof(H)}(H)
    end
end
dim(h::SingleParticleHilbertSpace) = dim(h.parent)
Base.parent(h::SingleParticleHilbertSpace) = h.parent
Base.keys(h::SingleParticleHilbertSpace) = keys(h.parent)
mode_ordering(h::SingleParticleHilbertSpace) = mode_ordering(h.parent)
modes(h::SingleParticleHilbertSpace) = modes(h.parent)
basisstates(h::SingleParticleHilbertSpace) = basisstates(h.parent)
Base.:(==)(a::SingleParticleHilbertSpace, b::SingleParticleHilbertSpace) = a === b || a.parent == b.parent
Base.hash(x::SingleParticleHilbertSpace, h::UInt) = hash(x.parent, h)
"""
    single_particle_hilbert_space(labels)

A hilbert space suitable for non-interacting systems with fermion number conservation. Matrix representations of symbolic operators give the single particle hamiltonian, without any contribution from the identity matrix.
"""
single_particle_hilbert_space(f::SymbolicFermionBasis, labels) = SingleParticleHilbertSpace(f, labels)
basisstate(ind, H::SingleParticleHilbertSpace) = basisstate(ind, parent(H))
state_index(state::AbstractFockState, H::SingleParticleHilbertSpace) = state_index(state, parent(H))

@testitem "Single particle hilbert space" begin
    using LinearAlgebra
    @fermions f
    H = single_particle_hilbert_space(f, 1:2)
    opmul = f[1]' * f[2]
    @test matrix_representation(opmul, H) ≈ matrix_representation(opmul, parent(H))
    opadd = opmul + hc
    ham = matrix_representation(opadd, H)
    @test ham ≈ matrix_representation(opadd, parent(H))
    @test matrix_representation(opadd + I, H) == matrix_representation(opadd, H)
    @test matrix_representation(opadd + I, H) == matrix_representation(opadd + I, parent(H)) - I
end

@testitem "Partial trace consistency: FockNumber vs FixedNumberFockState" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: FixedNumberFockState
    @fermions f
    # Define Hilbert spaces for 5 sites, 2 particles
    N = 5
    n_particles = 2
    # FockNumber-based Hilbert space
    H_fock = hilbert_space(f, 1:N, NumberConservation(n_particles))
    # FixedNumberFockState-based Hilbert space
    H_fixed = hilbert_space(f, 1:N, FixedNumberFockState{n_particles}.(basisstates(H_fock)))

    # Define a random Hermitian operator
    sym_ham = sum(rand() * f[n]'f[n] for n in 1:N) + sum(f[n+1]'f[n] + hc for n in 1:N-1)
    ham_fock = matrix_representation(sym_ham, H_fock)
    ham_fixed = matrix_representation(sym_ham, H_fixed)
    @test ham_fock ≈ ham_fixed
    # Diagonalize to get ground state
    Ψ_fock = eigvecs(collect(ham_fock))[:, 1]
    Ψ_fixed = eigvecs(collect(ham_fixed))[:, 1]

    # Subregion: first two sites
    Hsub = hilbert_space(f, [1, 3, 5])
    Hsub_fock = subregion(Hsub, H_fock)
    Hsub_fixed = subregion(Hsub, H_fixed)
    @test FockNumber.(basisstates(Hsub_fixed)) == basisstates(Hsub_fock)

    # Partial trace
    ρsub_fock = partial_trace(Ψ_fock * Ψ_fock', H_fock => Hsub_fock)
    ρsub_fixed = partial_trace(Ψ_fixed * Ψ_fixed', H_fixed => Hsub_fixed)
    @test ρsub_fock ≈ ρsub_fixed

    ρsub_fixed = partial_trace(Ψ_fixed * Ψ_fixed', H_fixed => Hsub_fixed; complement=FermionicHilbertSpaces.complementary_subsystem(H_fixed, Hsub_fixed))
    @test ρsub_fock ≈ ρsub_fixed
end


function matrix_representation(op, H::SingleParticleHilbertSpace)
    isquadratic(op) && isnumberconserving(op) || throw(ArgumentError("Only quadratic, number conserving operators supported for SingleParticleHilbertSpace"))
    _matrix_representation_single_space(remove_identity(op), H)
end
_find_position(op, H::SingleParticleHilbertSpace) = _find_position(op, parent(H))
function operator_indices_and_amplitudes!((outinds, ininds, amps), op::NCMul, H::SingleParticleHilbertSpace; kwargs...)
    ordering = mode_ordering(H)
    if length(op.factors) != 2
        throw(ArgumentError("Only two-fermion operators supported for free fermions"))
    end
    fockstates = (SingleParticleState(_find_position(op.factors[1], H)), SingleParticleState(_find_position(op.factors[2], H)))
    inind = state_index(fockstates[2], H)
    outind = state_index(fockstates[1], H)
    sign = (-1)^op.factors[2].creation
    push!(outinds, outind)
    push!(ininds, inind)
    push!(amps, sign * op.coeff)
    return (outinds, ininds, amps)
end


function nextfockstate_with_same_number(f::FockNumber{T}) where T
    FockNumber{T}(nextfockstate_with_same_number(f.f))
end
function nextfockstate_with_same_number(v::Integer)
    #http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (v | (v - 1)) + 1
    t | (((div((t & -t), (v & -v))) >> 1) - 1)
end
"""
    fixed_particle_number_fockstates(M, n)

Generate a list of Fock states with `n` occupied fermions in a system with `M` different fermions.
"""
function fixed_particle_number_fockstates(M, n, ::Type{T}=default_fock_representation(M)) where T
    iszero(n) && return [FockNumber{T}(zero(T))]
    v = focknbr_from_bits([k <= n for k in 1:M])
    maxv = v << (M - n)
    states = Vector{FockNumber{T}}(undef, binomial(M, n))
    count = 1
    while v <= maxv
        states[count] = v
        v = nextfockstate_with_same_number(v)
        count += 1
    end
    states
end
