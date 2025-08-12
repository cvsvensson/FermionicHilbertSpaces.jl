abstract type AbstractFockState <: AbstractBasisState end
"""
    FockNumber
A type representing a Fock state as the bitstring of an integer.
"""
struct FockNumber{I<:Integer} <: AbstractFockState
    f::I
end
FockNumber(f::FockNumber) = f
Base.:(==)(f1::FockNumber, f2::FockNumber) = f1.f == f2.f
Base.hash(f::FockNumber, h::UInt) = hash(f.f, h)
Base.isless(f1::FockNumber, f2::FockNumber) = f1.f < f2.f

"""
    JordanWignerOrdering
A type representing the ordering of fermionic modes.
"""
struct JordanWignerOrdering{L}
    labels::Vector{L}
    ordering::OrderedDict{L,Int}
    function JordanWignerOrdering(labels)
        ls = vec(collect(labels))
        dict = OrderedDict(zip(ls, Base.OneTo(length(ls))))
        new{eltype(ls)}(ls, dict)
    end
end
JordanWignerOrdering(jw::JordanWignerOrdering) = jw
Base.length(jw::JordanWignerOrdering) = length(jw.labels)
Base.:(==)(jw1::JordanWignerOrdering, jw2::JordanWignerOrdering) = jw1.labels == jw2.labels && jw1.ordering == jw2.ordering
Base.keys(jw::JordanWignerOrdering) = jw.labels
Base.iterate(jw::JordanWignerOrdering) = iterate(jw.labels)
Base.iterate(jw::JordanWignerOrdering, state) = iterate(jw.labels, state)
Base.eltype(::JordanWignerOrdering{L}) where L = L
Base.getindex(jw::JordanWignerOrdering, key) = getindex(jw.ordering, key)

siteindex(label, ordering::JordanWignerOrdering) = ordering.ordering[label]
siteindices(labels, jw::JordanWignerOrdering) = map(Base.Fix2(siteindex, jw), labels)
siteindices(labels, H::AbstractFockHilbertSpace) = siteindices(labels, mode_ordering(H))

label_at_site(n, ordering::JordanWignerOrdering) = ordering.labels[n]
focknbr_from_site_label(label, jw::JordanWignerOrdering) = focknbr_from_site_index(siteindex(label, jw))
focknbr_from_site_labels(labels, jw::JordanWignerOrdering) = mapreduce(Base.Fix2(focknbr_from_site_label, jw), +, labels, init=FockNumber(0))
focknbr_from_site_labels(labels::JordanWignerOrdering, jw::JordanWignerOrdering) = focknbr_from_site_labels(labels.labels, jw)

Base.:+(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f + f2.f)
Base.:-(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f - f2.f)
Base.:⊻(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f ⊻ f2.f)
Base.:⊻(f1::Integer, f2::FockNumber) = FockNumber(f1 ⊻ f2.f)
Base.:&(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f & f2.f)
Base.:&(f1::Integer, f2::FockNumber) = FockNumber(f1 & f2.f)
Base.:|(f1::FockNumber, f2::FockNumber) = FockNumber(f1.f | f2.f)
Base.iszero(f::FockNumber) = iszero(f.f)
Base.:*(b::Bool, f::FockNumber) = FockNumber(b * f.f)
Base.:~(f::FockNumber) = FockNumber(~f.f)
Base.:>>(f::FockNumber, n::Integer) = FockNumber(f.f >> n)
Base.:<<(f::FockNumber, n::Integer) = FockNumber(f.f << n)
Base.:|(n::Integer, f::FockNumber) = FockNumber(n | f.f)
Base.zero(::FockNumber{T}) where T = zero(FockNumber{T})
Base.zero(::Type{FockNumber{T}}) where T = FockNumber(zero(T))


focknbr_from_bits(bits, ::Type{T}=(length(bits) > 63 ? BigInt : Int)) where T = FockNumber{T}(reduce((x, y) -> x << 1 + y, Iterators.reverse(bits); init=zero(T)))
focknbr_from_site_index(site::Integer, ::Type{T}=site > 63 ? BigInt : Int) where T = FockNumber{T}(T(1) << (site - 1))
focknbr_from_site_indices(sites, ::Type{T}=(maximum(sites, init=0) > 63 ? BigInt : Int)) where T = mapreduce(focknbr_from_site_index, +, sites, init=FockNumber(zero(T)))

bits(f::FockNumber, N) = digits(Bool, f.f, base=2, pad=N)
parity(f::FockNumber) = iseven(fermionnumber(f)) ? 1 : -1
fermionnumber(f::FockNumber) = count_ones(f)
Base.count_ones(f::FockNumber) = count_ones(f.f)

fermionnumber(fs::FockNumber, mask) = count_ones(fs & mask)

"""
    jwstring(site, focknbr)
    
Parity of the number of fermions to the right of site.
"""
jwstring(site, focknbr) = jwstring_left(site, focknbr)
jwstring_anti(site, focknbr) = jwstring_right(site, focknbr)
jwstring_right(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f >> site)) ? 1 : -1
jwstring_left(site, focknbr::FockNumber) = iseven(count_ones(focknbr.f) - count_ones(focknbr.f >> (site - 1))) ? 1 : -1

struct FockMapper{P}
    fermionpositions::P
end

(fm::FockMapper)(f::NTuple{N,<:FockNumber}) where {N} = mapreduce(insert_bits, +, f, fm.fermionpositions)

function insert_bits(_x::FockNumber, positions)
    x = _x.f
    result = 0
    bit_index = 1
    for pos in positions
        if x & (1 << (bit_index - 1)) != 0
            result |= (1 << (pos - 1))
        end
        bit_index += 1
    end
    return FockNumber(result)
end

struct FockMapperBitPermutations{P1,P2}
    fermionpositions::P1
    widths::Vector{Int}
    permutation::P2
end
FockMapper_collect(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix2(siteindices, jw) ∘ collect ∘ keys, jws)) #faster construction
FockMapper_tuple(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix2(siteindices, jw) ∘ Tuple ∘ keys, jws)) #faster application, but type instability
FockMapper(Hs, H::AbstractFockHilbertSpace) = FockMapper(map(mode_ordering, Hs), mode_ordering(H))
StateExtender(Hs, H::AbstractFockHilbertSpace) = FockMapper(Hs, H)

function FockMapper_bp(jws, jw)
    widths = collect(map(length, jws))
    fermionpositions = map(Base.Fix2(siteindices, jw) ∘ collect ∘ keys, jws)
    permutation = BitPermutation{UInt}(reduce(vcat, fermionpositions))
    FockMapperBitPermutations(fermionpositions, widths, permutation')
end
FockMapper(jws, jw::JordanWignerOrdering) = FockMapper_bp(jws, jw)
(fm::FockMapperBitPermutations)(fs) = concatenate_and_permute(fs, fm.widths, fm.permutation)

function concatenate_and_permute(fs, widths, permutation)
    mask = foldl(concatenate, zip(fs, widths); init=(zero(first(fs)), 0)) |> first
    # mask = concatenate_bitmasks(masks, widths)
    permute(mask, permutation)
end
function concatenate((lastf, lastwidth)::Tuple{FockNumber,Int}, (f, width)::Tuple{FockNumber,Int})
    return (lastf | (f << lastwidth), lastwidth + width)
end
# function concatenate_bitmasks(masks, widths)
#     result = zero(first(masks))
#     return foldl(((result, lastwidth), (mask, width)) -> (result | (mask << lastwidth), lastwidth + width), zip(masks, widths); init=(result, 0)) |> first
# end

permute(f::FockNumber{T}, p) where T = FockNumber{T}(bitpermute(f.f, p))

shift_right(f::FockNumber, M) = FockNumber(f.f << M)
FockSplitter(H::AbstractFockHilbertSpace, Hs) = FockSplitter(mode_ordering(H), map(mode_ordering, Hs))
StateSplitter(H::AbstractFockHilbertSpace, Hs) = FockSplitter(H, Hs)


@testitem "Fock" begin
    using Random
    using FermionicHilbertSpaces: bits, _bit, focknbr_from_bits, focknbr_from_site_indices, focknbr_from_site_index
    Random.seed!(1234)

    N = 6
    focknumber = FockNumber(20) # = 16+4 = 00101
    fbits = bits(focknumber, N)
    @test fbits == [0, 0, 1, 0, 1, 0]

    @test focknbr_from_bits(fbits) == focknumber
    @test focknbr_from_bits(Tuple(fbits)) == focknumber
    @test !_bit(focknumber, 1)
    @test !_bit(focknumber, 2)
    @test _bit(focknumber, 3)
    @test !_bit(focknumber, 4)
    @test _bit(focknumber, 5)

    @test focknbr_from_site_indices((3, 5)) == focknumber
    @test focknbr_from_site_indices([3, 5]) == focknumber

    @testset "removefermion" begin
        focknbr = FockNumber(rand(1:2^N) - 1)
        fockbits = bits(focknbr, N)
        function test_remove(n)
            FermionicHilbertSpaces.removefermion(n, focknbr) == (fockbits[n] ? (focknbr - FockNumber(2^(n - 1)), (-1)^sum(fockbits[1:n-1])) : (FockNumber(0), 0))
        end
        @test all([test_remove(n) for n in 1:N])
    end

    @testset "ToggleFermions" begin
        focknbr = FockNumber(177) # = 1000 1101, msb to the right
        digitpositions = Vector([7, 8, 2, 3])
        daggers = BitVector([1, 0, 1, 1])
        newfocknbr, sign = FermionicHilbertSpaces.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == 1
        # swap two operators
        digitpositions = Vector([7, 2, 8, 3])
        daggers = BitVector([1, 1, 0, 1])
        newfocknbr, sign = FermionicHilbertSpaces.togglefermions(digitpositions, daggers, focknbr)
        @test newfocknbr == FockNumber(119) # = 1110 1110
        @test sign == -1

        # annihilate twice
        digitpositions = Vector([5, 3, 5])
        daggers = BitVector([0, 1, 0])
        _, sign = FermionicHilbertSpaces.togglefermions(digitpositions, daggers, focknbr)
        @test sign == 0
    end

    fs = FermionicHilbertSpaces.fixed_particle_number_fockstates(10, 5)
    @test length(fs) == binomial(10, 5)
    @test allunique(fs)
    @test all(FermionicHilbertSpaces.fermionnumber.(fs) .== 5)

    @testset "Large site indices" begin
        f = focknbr_from_site_indices((1, 1000,))
        f isa FockNumber{BigInt}
        f.f == 1 + BigInt(2)^(1000 - 1)
        focknbr_from_site_index(1000).f == f.f - 1
    end
end


##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)

function FockSplitter(jw::JordanWignerOrdering, jws)
    fermionpositions = Tuple(map(Base.Fix2(siteindices, jw) ∘ Tuple ∘ collect ∘ keys, jws))
    Base.Fix2(split_state, fermionpositions)
end
# function split_focknumber(f::FockNumber, fermionpositions)
#     map(positions -> focknbr_from_bits(Iterators.map(i -> _bit(f, i), positions)), fermionpositions)
# end
function split_state(f::AbstractFockState, fermionpositions)
    map(site_indices -> substate(site_indices, f), fermionpositions)
end
function split_state(f::AbstractFockState, fockmapper::FockMapper)
    split_state(f, fockmapper.fermionpositions)
end
function split_state(f::AbstractFockState, fockmapper::FockMapperBitPermutations)
    split_state(f, fockmapper.fermionpositions)
end
combine_states(f1::FockNumber, f2::FockNumber, H1, H2) = f1 + shift_right(f2, length(H1.jw))

@testitem "Split and join focknumbers" begin
    import FermionicHilbertSpaces: focknbr_from_site_indices as fock
    jw1 = JordanWignerOrdering((1, 3))
    jw2 = JordanWignerOrdering((2, 4))
    jw = JordanWignerOrdering(1:4)
    focksplitter = FermionicHilbertSpaces.FockSplitter(jw, (jw1, jw2))
    @test focksplitter(fock((1, 2, 3, 4))) == (fock((1, 2)), fock((1, 2)))
    @test focksplitter(fock((1,))) == (fock((1,)), fock(()))
    @test focksplitter(fock(())) == (fock(()), fock(()))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2, 4))) == (fock(()), fock((1, 2)))
    @test focksplitter(fock((3, 2))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3, 4))) == (fock((2,)), fock((2,)))

    fockmapper = FermionicHilbertSpaces.FockMapper((jw1, jw2), jw)
    @test FermionicHilbertSpaces.split_focknumber(fock((1, 2, 4)), fockmapper) == focksplitter(fock((1, 2, 4)))

    # test all cases above with fockmapper
    @test fock((1, 2, 3, 4)) == fockmapper((fock((1, 2)), fock((1, 2))))
    @test fock((1,)) == fockmapper((fock((1,)), fock(())))
    @test fock(()) == fockmapper((fock(()), fock(())))
    @test fock((1, 2, 3)) == fockmapper((fock((1, 2)), fock((1,))))
    @test fock((1, 3)) == fockmapper((fock((1, 2)), fock(())))
    @test fock((2, 4)) == fockmapper((fock(()), fock((1, 2))))
    @test fock((3, 2)) == fockmapper((fock((2,)), fock((1,))))
    @test fock((3, 4)) == fockmapper((fock((2,)), fock((2,))))

    # test splitting with different sizes
    jw1 = JordanWignerOrdering((1, 2))
    jw2 = JordanWignerOrdering((3,))
    jw = JordanWignerOrdering((1, 2, 3))
    focksplitter = FermionicHilbertSpaces.FockSplitter(jw, (jw1, jw2))
    @test focksplitter(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test focksplitter(fock((1, 3))) == (fock((1,)), fock((1,)))
    @test focksplitter(fock((1, 2))) == (fock((1, 2)), fock(()))
    @test focksplitter(fock((2,))) == (fock((2,)), fock(()))
    @test focksplitter(fock((2, 3))) == (fock((2,)), fock((1,)))
    @test focksplitter(fock((3,))) == (fock(()), fock((1)))

    # test all cases above with fockmapper
    fockmapper = FermionicHilbertSpaces.FockMapper((jw1, jw2), jw)
    @test FermionicHilbertSpaces.split_focknumber(fock((1, 2, 3)), fockmapper) == focksplitter(fock((1, 2, 3)))
    @test fock((1, 3)) == fockmapper((fock((1,)), fock((1,))))
    @test fock((1, 2)) == fockmapper((fock((1, 2)), fock(())))
    @test fock((2,)) == fockmapper((fock((2,)), fock(())))
    @test fock((2, 3)) == fockmapper((fock((2,)), fock((1,))))
    @test fock((3,)) == fockmapper((fock(()), fock((1,))))

    # test splitting with different sizes
    jw1 = JordanWignerOrdering((1, 4))
    jw2 = JordanWignerOrdering((2,))
    jw3 = JordanWignerOrdering((3,))
    jw = JordanWignerOrdering(1:4)
    focksplitter = FermionicHilbertSpaces.FockSplitter(jw, (jw1, jw2, jw3))
    fockmapper = FermionicHilbertSpaces.FockMapper((jw1, jw2, jw3), jw)
    ident = fockmapper ∘ focksplitter
    @test all(ident(FockNumber(k)) == FockNumber(k) for k in 0:2^length(jw)-1)
end

## N-particle fock state
using TupleTools
struct FixedNumberFockState{N} <: AbstractFockState
    sites::NTuple{N,Int}
    FixedNumberFockState{N}(sites::NTuple{N,Int}) where N = new{N}(TupleTools.sort(sites))
end
FixedNumberFockState(sites::NTuple{N,Int}) where N = FixedNumberFockState{N}(TupleTools.sort(sites))
Base.:(==)(f1::FixedNumberFockState, f2::FixedNumberFockState) = f1.sites == f2.sites
Base.hash(f::FixedNumberFockState, h::UInt) = hash(FockNumber(f), h)
Base.:(==)(f1::FixedNumberFockState, f2::FockNumber) = f2 == f1
Base.:(==)(f1::FockNumber, f2::FixedNumberFockState) = f1 == FockNumber(f2)
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