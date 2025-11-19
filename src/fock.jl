abstract type AbstractFockState <: AbstractBasisState end
"""
    FockNumber
A type representing a Fock state as the bitstring of an integer.
"""
struct FockNumber{I} <: AbstractFockState
    f::I
end
FockNumber(f::FockNumber) = f
FockNumber{I}(f::FockNumber) where I<:Integer = FockNumber{I}(f.f)
Base.:(==)(f1::FockNumber, f2::FockNumber) = f1.f == f2.f
Base.hash(f::FockNumber, h::UInt) = hash(f.f, h)
Base.isless(f1::FockNumber, f2::FockNumber) = f1.f < f2.f
Base.show(io::IO, f::FockNumber{T}) where T = get(io, :compact, false) ? print(io, "FockNumber{T}(", f.f, ")") : print(io, "FockNumber(", f.f, ")")
"""
    JordanWignerOrdering
A type representing the ordering of fermionic modes.
"""
struct JordanWignerOrdering{L}
    ordering::OrderedDict{L,Int}
    function JordanWignerOrdering(labels)
        dict = OrderedDict(zip(labels, Base.OneTo(length(labels))))
        new{keytype(dict)}(dict)
    end
end
JordanWignerOrdering(jw::JordanWignerOrdering) = jw
Base.length(jw::JordanWignerOrdering) = length(jw.ordering)
Base.:(==)(jw1::JordanWignerOrdering, jw2::JordanWignerOrdering) = jw1.ordering == jw2.ordering
Base.keys(jw::JordanWignerOrdering) = jw.ordering.keys
Base.iterate(jw::JordanWignerOrdering) = iterate(keys(jw))
Base.iterate(jw::JordanWignerOrdering, state) = iterate(keys(jw), state)
Base.eltype(::JordanWignerOrdering{L}) where L = L

Base.getindex(ordering::JordanWignerOrdering, label) = ordering.ordering[label]
getindices(jw::JordanWignerOrdering, labels) = map(Base.Fix1(getindex, jw), labels)
getindices(H::AbstractFockHilbertSpace, labels) = getindices(mode_ordering(H), labels)

label_at_site(n, jw::JordanWignerOrdering) = keys(jw)[n]
focknbr_from_site_label(label, jw::JordanWignerOrdering) = focknbr_from_site_index(getindex(jw, label))
focknbr_from_site_labels(labels, jw::JordanWignerOrdering) = mapreduce(Base.Fix2(focknbr_from_site_label, jw), +, labels, init=FockNumber(0))
focknbr_from_site_labels(labels::JordanWignerOrdering, jw::JordanWignerOrdering) = focknbr_from_site_labels(keys(labels), jw)

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


integer_from_bits(bits, ::Type{T}=default_fock_representation(length(bits))) where T = reduce((x, y) -> x << 1 + y, Iterators.reverse(bits); init=zero(T))
focknbr_from_bits(bits, ::Type{T}=default_fock_representation(length(bits))) where T = FockNumber{T}(integer_from_bits(bits, T))
focknbr_from_site_index(site::Integer, ::Type{T}=default_fock_representation(site)) where T = FockNumber{T}(one(T) << (site - 1))
focknbr_from_site_indices(sites, ::Type{T}=default_fock_representation(maximum(sites, init=0))) where T = mapreduce(focknbr_from_site_index, +, sites, init=FockNumber{T}(zero(T)))

bits(f::FockNumber, N) = digits(Bool, f.f, base=2, pad=N)
parity(f::FockNumber) = iseven(fermionnumber(f)) ? 1 : -1
fermionnumber(f::FockNumber) = count_ones(f)
Base.count_ones(f::FockNumber) = count_ones(f.f)

fermionnumber(f::FockNumber{<:Integer}, mask) = count_weighted_ones(f.f, mask)
count_weighted_ones(x, mask::Integer) = count_ones(x & mask)
count_weighted_ones(x, weights::Union{Vector,Tuple}) = sum(w for (i, w) in enumerate(weights) if _bit(x, i))

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
FockMapper_collect(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix1(getindices, jw) ∘ collect ∘ keys, jws)) #faster construction
FockMapper_tuple(jws, jw::JordanWignerOrdering) = FockMapper(map(Base.Fix1(getindices, jw) ∘ Tuple ∘ keys, jws)) #faster application, but type instability
FockMapper(Hs, H::AbstractFockHilbertSpace) = FockMapper(map(mode_ordering, Hs), mode_ordering(H))
StateExtender(Hs, H::AbstractFockHilbertSpace) = FockMapper(Hs, H)

function FockMapper_bp(jws, jw)
    widths = collect(map(length, jws))
    fermionpositions = map(Base.Fix1(getindices, jw) ∘ collect ∘ keys, jws)
    permutation = BitPermutation{UInt}(reduce(vcat, fermionpositions))
    FockMapperBitPermutations(fermionpositions, widths, permutation')
end
FockMapper(jws, jw::JordanWignerOrdering) = FockMapper_bp(jws, jw)
(fm::FockMapperBitPermutations)(fs) = concatenate_and_permute(fs, fm.widths, fm.permutation)

function concatenate_and_permute(fs, widths, permutation)
    mask = foldl(concatenate, zip(fs, widths); init=(zero(first(fs)), 0)) |> first
    permute(mask, permutation)
end
function concatenate((lastf, lastwidth)::Tuple{FockNumber,Int}, (f, width)::Tuple{FockNumber,Int})
    return (lastf | (f << lastwidth), lastwidth + width)
end

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

_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)
_bit(f::Integer, k) = Bool((f >> (k - 1)) & 1)

function FockSplitter(jw::JordanWignerOrdering, jws)
    fermionpositions = Tuple(map(Base.Fix1(getindices, jw) ∘ Tuple ∘ collect ∘ keys, jws))
    Base.Fix2(split_state, fermionpositions)
end
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
    @test FermionicHilbertSpaces.split_state(fock((1, 2, 4)), fockmapper) == focksplitter(fock((1, 2, 4)))

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
    @test FermionicHilbertSpaces.split_state(fock((1, 2, 3)), fockmapper) == focksplitter(fock((1, 2, 3)))
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
