struct FixedNumberFockState{N} <: AbstractFockState
    sites::NTuple{N,Int}
    FixedNumberFockState{N}(sites::NTuple{N,Int}) where N = new{N}(TupleTools.sort(sites))
end
FixedNumberFockState(sites::NTuple{N,Int}) where N = FixedNumberFockState{N}(TupleTools.sort(sites))
Base.:(==)(f1::FixedNumberFockState, f2::FixedNumberFockState) = f1.sites == f2.sites
Base.hash(f::FixedNumberFockState, h::UInt) = hash(f.sites, h)

const SingleParticleState = FixedNumberFockState{1}
SingleParticleState(site::Int) = FixedNumberFockState((site,))
function jwstring_left(site, f::FixedNumberFockState)
    sign = 1
    for s in f.sites
        if s < site
            sign *= -1
        end
    end
    return sign
end
function jwstring_right(site, f::FixedNumberFockState)
    sign = 1
    for s in f.sites
        if s > site
            sign *= -1
        end
    end
    return sign
end
FockNumber(f::FixedNumberFockState) = focknbr_from_site_indices(f.sites)
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
combine_states(f1::FixedNumberFockState, f2::FixedNumberFockState, H1, H2) = FixedNumberFockState((f1.sites..., f2.sites...))
_bit(f::FixedNumberFockState, k) = k in f.sites
function substate(siteindices, f::FixedNumberFockState)
    # subsite = 0
    subsites = Int[]
    for (n, site) in enumerate(siteindices)
        site in f.sites && push!(subsites, n)
    end
    # sites = filter(s -> s in siteindices, f.sites)
    return FixedNumberFockState(Tuple(subsites))
end

Base.isless(a::FixedNumberFockState, b::FixedNumberFockState) = a.sites < b.sites

@testitem "FixedNumberFockState" begin
    import FermionicHilbertSpaces: jwstring_left, jwstring_right, FixedNumberFockState, FockNumber, SingleParticleState, _bit, substate
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
    H1 = hilbert_space(1:2)
    H2 = hilbert_space(3:4)
    H12 = hilbert_space((4, 2, 1, 3))
    fm = FermionicHilbertSpaces.FockMapper((H1, H2), H12)
    for (f1, f2) in Base.product(basisstates(H1), basisstates(H2))
        f1fix = FixedNumberFockState(f1)
        f2fix = FixedNumberFockState(f2)
        f12 = fm((f1, f2))
        f12fix = fm((f1fix, f2fix))
        @test f12 == FockNumber(f12fix)
    end

    @fermions f
    h = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
    H = hilbert_space(1:2, FermionicHilbertSpaces.SingleParticleState.(1:3))
    @test_throws KeyError matrix_representation(h, H)

    N = 10
    H = hilbert_space(1:N, SingleParticleState.(1:N))
    Hf = hilbert_space(1:N, FermionConservation(1))
    @test length(basisstates(H)) == length(basisstates(Hf)) == N

    @fermions f
    op = sum(rand() * f[k1]'f[k2] + rand(ComplexF64) * f[k1]f[k2]' for (k1, k2) in Base.product(1:N, 1:N))
    @test matrix_representation(op, H) ≈ matrix_representation(op, Hf)
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
function concatenate((lastf, lastwidth)::Tuple{FixedNumberFockState,Int}, (f, width)::Tuple{FockNumber,Int})
    concatenate((FockNumber(lastf), lastwidth), (f, width))
end
function permute(f::FixedNumberFockState, permutation::BitPermutations.AbstractBitPermutation)
    p = Vector(permutation')
    return FixedNumberFockState(map(s -> p[s], f.sites))
end

struct SingleParticleHilbertSpace{H} <: AbstractFockHilbertSpace
    parent::H
    function SingleParticleHilbertSpace(labels)
        states = [SingleParticleState(i) for (i, label) in enumerate(labels)]
        H = hilbert_space(labels, states)
        return new{typeof(H)}(H)
    end
end
Base.size(h::SingleParticleHilbertSpace) = size(h.parent)
Base.size(h::SingleParticleHilbertSpace, dim) = size(h.parent, dim)
Base.parent(h::SingleParticleHilbertSpace) = h.parent
Base.keys(h::SingleParticleHilbertSpace) = keys(h.parent)
mode_ordering(h::SingleParticleHilbertSpace) = mode_ordering(h.parent)
modes(h::SingleParticleHilbertSpace) = modes(h.parent)
basisstates(h::SingleParticleHilbertSpace) = basisstates(h.parent)
single_particle_hilbert_space(labels) = SingleParticleHilbertSpace(labels)
matrix_representation(op, H::SingleParticleHilbertSpace) = matrix_representation(remove_identity(op), parent(H))
basisstate(ind, H::SingleParticleHilbertSpace) = basisstate(ind, parent(H))
state_index(state::AbstractFockState, H::SingleParticleHilbertSpace) = state_index(state, parent(H))

@testitem "Single particle hilbert space" begin
    using LinearAlgebra
    @fermions f
    H = single_particle_hilbert_space(1:2)
    opmul = f[1]' * f[2]
    @test matrix_representation(opmul, H) ≈ matrix_representation(opmul, parent(H))
    opadd = opmul + hc
    ham = matrix_representation(opadd, H)
    @test ham ≈ matrix_representation(opadd, parent(H))
    @test matrix_representation(opadd + I, H) == matrix_representation(opadd, H)
    @test matrix_representation(opadd + I, H) == matrix_representation(opadd + I, parent(H)) - I
end