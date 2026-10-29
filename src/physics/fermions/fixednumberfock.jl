struct FixedNumberFockState{N} <: AbstractFockState
    sites::NTuple{N,Int}
    FixedNumberFockState{N}(sites::NTuple{N,Int}) where N = new{N}(TupleTools.sort(sites))
end
FixedNumberFockState(sites::NTuple{N,Int}) where N = FixedNumberFockState{N}(TupleTools.sort(sites))
Base.:(==)(f1::FixedNumberFockState, f2::FixedNumberFockState) = f1.sites == f2.sites
Base.hash(f::FixedNumberFockState, h::UInt) = hash(f.sites, h)

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
FixedNumberFockState{N}(f::FixedNumberFockState{N}) where {N} = f
FixedNumberFockState(f::FockNumber) = FixedNumberFockState{count_ones(f)}(f)
function FixedNumberFockState{N}(f::FockNumber) where N
    sites = ntuple(i -> 0, N)
    count_ones(f) == N || return FixedNumberFockState{N}(sites)
    bits = f.f
    for k in 1:N
        sites = Base.setindex(sites, trailing_zeros(bits) + 1, k)
        bits &= bits - one(bits)
    end
    return FixedNumberFockState{N}(sites)
end


physical_rep(state::T, ::Type{FixedNumberFockState{N}}) where {T,N} = FixedNumberFockState{N}(physical_rep(state, FockNumber))
internal_rep(state::FixedNumberFockState, space::FermionicSpace, ::Type{T}) where T = internal_rep(FockNumber(state), space, T)


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
function split_state(f::FixedNumberFockState, fm::FockMapper{N}) where N
    state = map(pos -> substate(pos, f), fm.fermionpositions)
    (state,), (1,)
end

Base.isless(a::FixedNumberFockState, b::FixedNumberFockState) = a.sites < b.sites

@testitem "FixedNumberFockState" begin
    import FermionicHilbertSpaces: jwstring_left, jwstring_right, FixedNumberFockState, FockNumber, _bit, substate, state_mapper, combine_states
    using FermionicHilbertSpaces.FreeFermions: SingleParticleState
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
    H = hilbert_space(a, 1:2, SingleParticleState.(1:3))
    @test_throws ArgumentError representation(h, H)
    @test_throws ArgumentError representation(h, H)

    N = 10
    H = hilbert_space(a, 1:N, SingleParticleState.(1:N))
    Hf = hilbert_space(a, 1:N, NumberConservation(1))
    @test length(basisstates(H)) == length(basisstates(Hf)) == N
    @test map(FockNumber, basisstates(H)) == basisstates(Hf)
    op = sum(rand() * a[k1]'a[k2] + rand(ComplexF64) * a[k1]a[k2]' for (k1, k2) in Base.product(1:N, 1:N))
    @test representation(op, H) ≈ representation(op, Hf)
end

@testitem "Kets of a space of FixedNumberFockState" begin
    import FermionicHilbertSpaces: FixedNumberFockState, vector_representation
    @fermions f
    Hx = hilbert_space(f, 1:3, FixedNumberFockState{1}.(basisstates(hilbert_space(f, 1:3, NumberConservation(1)))))
    for i in 1:dim(Hx)
        @test Ket(basisstate(i, Hx), Hx) == Kets(Hx)[i]
        @test vector_representation(Ket(basisstate(i, Hx), Hx), Hx) == [j == i for j in 1:dim(Hx)]
    end
    @test Ket("010", Hx) == Kets(Hx)[2]
end

function _precomputation_before_operator_application(op::FermionSym, space::AbstractHilbertSpace{<:FixedNumberFockState})
    position = mode_position(op, space)
    position == 0 && throw(ArgumentError("The operator `$op` contains a factor that is not part of the fermionic space `$space`."))
    return position
end
# The change in particle number is stored as a type parameter so that the state produced by a term has a concrete type.
function precomputation_before_operator_application(op::NCMul, space::AbstractHilbertSpace{<:FixedNumberFockState})
    sites = Tuple(map(factor -> _precomputation_before_operator_application(factor, space), op.factors))
    return sites, Val(sum(factor -> factor.creation ? 1 : -1, op.factors; init = 0))
end
function apply_local_operators(op::NCMul, f::FixedNumberFockState{N}, H::AbstractHilbertSpace, (sites, _)::Tuple{Any, Val{Δ}}; transpose) where {N, Δ}
    if !transpose
        daggers = Iterators.map(op -> op.creation, Iterators.reverse(op.factors))
        state, amp = togglefermions(reverse(sites), daggers, f, Val(N + Δ))
    else
        daggers = Iterators.map(op -> !op.creation, op.factors)
        state, amp = togglefermions(sites, daggers, f, Val(N - Δ))
    end
    return state, amp * op.coeff
end
"""
    togglefermions(sites, daggers, f::FixedNumberFockState, ::Val{M})

Apply creation (`dagger == true`) and annihilation operators at `sites`, in order, to `f`. Returns the
resulting state with `M` particles and the Jordan-Wigner sign, or a sign of `0` if an operator annihilates the state.
"""
function togglefermions(sites, daggers, f::FixedNumberFockState, ::Val{M}) where {M}
    M < 0 && return f, 0
    jw = false
    for (j, (site, dagger)) in enumerate(zip(sites, daggers))
        occupied = _bit(f, site)
        below = count(<(site), f.sites)
        for k in 1:(j - 1)
            occupied ⊻= sites[k] == site
            below += sites[k] < site
        end
        occupied == dagger && return FixedNumberFockState{M}(ntuple(i -> 0, Val(M))), 0
        jw ⊻= isodd(below)
    end
    # A mode is occupied at the end if it appears an odd number of times among the initial and toggled sites.
    candidates = (f.sites..., sites...)
    final = ntuple(Val(length(candidates))) do j
        c = candidates[j]
        isodd(count(==(c), candidates)) && findfirst(==(c), candidates) == j ? c : typemax(Int)
    end
    sorted = TupleTools.sort(final)
    return FixedNumberFockState{M}(ntuple(i -> sorted[i], Val(M))), jw ? -1 : 1
end
Base.zero(::FixedNumberFockState) = FixedNumberFockState(())
function concatenate((lastf, lastwidth)::Tuple{FixedNumberFockState,Int}, (f, width)::Tuple{FixedNumberFockState,Int})
    return (FixedNumberFockState((lastf.sites..., (lastwidth .+ f.sites)...)), lastwidth + width)
end
function concatenate((lastf, lastwidth)::Tuple{FockNumber,Int}, (f, width)::Tuple{FixedNumberFockState,Int})
    concatenate((FixedNumberFockState(lastf), lastwidth), (f, width))
end
function concatenate((lastf, lastwidth)::Tuple{FixedNumberFockState,Int}, (f, width)::Tuple{FockNumber,Int})
    concatenate((lastf, lastwidth), (FixedNumberFockState(f), width))
end
function permute(f::FixedNumberFockState, permutation::AbstractBitPermutation)
    p = Vector(permutation')
    return FixedNumberFockState(map(s -> p[s], f.sites))
end

@testitem "Vector tensor product of fixed-number states with reordered modes" begin
    import FermionicHilbertSpaces: FixedNumberFockState
    @fermions f
    H1 = hilbert_space(f, 1:2, NumberConservation(1))
    H2 = hilbert_space(f, 3:3, NumberConservation(1))
    H = hilbert_space(f, [1, 3, 2], NumberConservation(2))
    H1fix = hilbert_space(f, 1:2, FixedNumberFockState{1}.(basisstates(H1)))
    H2fix = hilbert_space(f, 3:3, FixedNumberFockState{1}.(basisstates(H2)))
    Hfix = hilbert_space(f, [1, 3, 2], FixedNumberFockState{2}.(basisstates(H)))
    v = tensor_product(([1.0, 2.0], [1.0]), (H1, H2), H)
    vfix = tensor_product(([1.0, 2.0], [1.0]), (H1fix, H2fix), Hfix)
    @test vfix ≈ v
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
    sym_ham = sum(rand() * f[n]'f[n] for n in 1:N) + sum(f[n+1]'f[n] + hc for n in 1:(N-1))
    ham_fock = representation(sym_ham, H_fock)
    ham_fixed = representation(sym_ham, H_fixed)
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

    msub = rand(dim(Hsub_fock), dim(Hsub_fock))
    m_fock = embed(msub, Hsub_fock => H_fock; skipmissing=true)
    m_fixed = embed(msub, Hsub_fixed => H_fixed; skipmissing=true)
    @test m_fock ≈ m_fixed

    Hcomp_fixed = FermionicHilbertSpaces.complementary_subsystem(H_fixed, Hsub_fixed)
    Hcomp_fock = FermionicHilbertSpaces.complementary_subsystem(H_fock, Hsub_fock)
    mcomp = rand(dim(Hcomp_fixed), dim(Hcomp_fixed))
    H_fock = tensor_product(Hsub_fock, Hcomp_fock)
    H_fixed = tensor_product(Hsub_fixed, Hcomp_fixed)
    @test dim(H_fock) == dim(H_fixed)
    m_fock = tensor_product((msub, mcomp), (Hsub_fock, Hcomp_fock) => H_fock)
    m_fixed = tensor_product((msub, mcomp), (Hsub_fixed, Hcomp_fixed) => H_fixed)
    @test m_fock ≈ m_fixed
end

function phase_factor_h(f1::FixedNumberFockState, f2::FixedNumberFockState, partition, masks)
    phase_factor_h(FockNumber(f1), FockNumber(f2), partition, masks)
end
phase_factor_u(partition, masks, state::FixedNumberFockState) = phase_factor_u(partition, masks, FockNumber(state))


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
