"""
    FockNumber
A type representing a Fock state as the bitstring of an integer.
"""
struct FockNumber{I} <: AbstractFockState
    f::I
end
FockNumber(f::FockNumber) = f
FockNumber{I}(f::FockNumber) where I = FockNumber{I}(I(f.f))
Base.convert(::Type{FockNumber{I}}, f::FockNumber) where {I} = FockNumber{I}(I(f.f))
Base.:(==)(f1::FockNumber, f2::FockNumber) = f1.f == f2.f
Base.hash(f::FockNumber, h::UInt) = hash(f.f, h)
Base.isless(f1::FockNumber, f2::FockNumber) = f1.f < f2.f

function default_fock_representation(::Val{N}) where N
    N < 64 ? UInt64 : BigInt
end
function default_fock_representation(N)
    N < 64 ? UInt64 : BigInt
end

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
Base.zero(::Type{FockNumber{T}}) where T = FockNumber{T}(zero(T))


integer_from_bits(bits, ::Type{T}=default_fock_representation(length(bits))) where T = reduce((x, y) -> x << 1 + y, Iterators.reverse(bits); init=zero(T))
focknbr_from_bits(bits, ::Type{T}=default_fock_representation(length(bits))) where T = FockNumber{T}(integer_from_bits(bits, T))
focknbr_from_site_index(site::Integer, ::Type{T}=default_fock_representation(site)) where T = FockNumber{T}(one(T) << (site - 1))
focknbr_from_site_indices(sites, ::Type{T}=default_fock_representation(maximum(sites, init=0))) where T =
    FockNumber{T}(reduce((acc, s) -> acc | (one(T) << (s - 1)), sites; init=zero(T)))

bits(f::FockNumber, N) = digits(Bool, f.f, base=2, pad=N)
parity(f::FockNumber) = iseven(fermionnumber(f)) ? 1 : -1
fermionnumber(f::FockNumber) = count_ones(f)
Base.count_ones(f::FockNumber) = count_ones(f.f)
particle_number(s::FockNumber) = fermionnumber(s)

# function substate(siteindices, f::FockNumber{T}) where T
#     subbits = Iterators.map(i -> _bit(f, i), siteindices)
#     return focknbr_from_bits(subbits, T)
# end
function substate(siteindices, f::FockNumber{T}) where T
    val = zero(T)
    for idx in Iterators.reverse(siteindices)
        val = (val << 1) | _bit(f, idx)
    end
    return FockNumber{T}(val)
end


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


struct FockMapper{N,P1,W,P2} <: AbstractStateMapper
    fermionpositions::P1
    widths::W
    permutation::P2
    FockMapper(fermionpositions::P1, widths::W, permutation::P2, nbr_of_modes::Int) where {P1,W,P2} = new{nbr_of_modes,P1,W,P2}(fermionpositions, widths, permutation)
end
function FockMapper(fermionpositions::P) where P
    widths = map(length, fermionpositions)
    nbr_of_modes = maximum(maximum, fermionpositions)
    perm = mapreduce(collect, vcat, fermionpositions)
    all(issorted, fermionpositions) || throw(ArgumentError("The order of fermions in each subsystem should be ordered as in the full system, but the provided fermion positions are not sorted: $fermionpositions"))
    permutation = isperm(perm) && nbr_of_modes < 64 ? BitPermutation{UInt}(perm)' : nothing
    FockMapper(fermionpositions, widths, permutation, nbr_of_modes)
end
unique_split(::FockMapper) = true
unique_combine(::FockMapper) = true
function Base.show(io::IO, fm::FockMapper{N}) where N
    print(io, "FockMapper($N modes → ")
    print(io, join(["[" * join(pos, ",") * "]" for pos in fm.fermionpositions], " ⊗ "))
    print(io, ")")
end
function combine_states(fs, fm::FockMapper{N,<:Any,<:Any,<:BitPermutation}) where N
    state = concatenate_and_permute(fs, fm.widths, fm.permutation, FockNumber{default_fock_representation(Val(N))})
    (state,), (1,)
end
function split_state(f::AbstractFockState, fm::FockMapper{N}) where N
    state = map(site_indices -> substate(site_indices, f), fm.fermionpositions)
    (state,), (1,)
end


function combine_states(f, fm::FockMapper{N}) where N
    IT = default_fock_representation(Val(N))
    result = zero(IT)
    for (fock, positions) in zip(f, fm.fermionpositions)
        remaining = FockNumber{IT}(fock).f
        for pos in positions
            iszero(remaining) && break
            result |= (remaining & one(IT)) << (pos - 1)
            remaining >>= one(IT)
        end
    end
    (FockNumber{IT}(result),), (1,)
end

function catenate_fock_states(fock_states, spaces, T)
    num = zero(T)
    shift = 0
    for (state, space) in zip(fock_states, spaces)
        num |= state << shift
        shift += nbr_of_modes(space)
    end
    num
end

# function combine_states(f, fm::FockMapper{N}) where N
#     T = FockNumber{default_fock_representation(Val(N))}
#     state = mapfoldr(tup -> insert_bits(T(tup[1]), tup[2]), |, zip(f, fm.fermionpositions); init=zero(T))
#     (state,), (1,)
# end
function insert_bits(_x::FockNumber{T}, positions) where T
    x = _x.f
    result = zero(T)
    bit_index = 1
    for pos in positions
        if x & (one(T) << (bit_index - 1)) != 0
            result |= (one(T) << (pos - 1))
        end
        bit_index += 1
    end
    return FockNumber{T}(result)
end

function concatenate_and_permute(fs, widths, permutation, ::Type{T}) where T
    mask = foldl(concatenate, zip(fs, widths); init=(zero(T), 0)) |> first
    permute(mask, permutation)
end
function concatenate((lastf, lastwidth)::Tuple{FockNumber,Int}, (f, width)::Tuple{FockNumber,Int})
    return (lastf | (f << lastwidth), lastwidth + width)
end

permute(f::FockNumber{T}, p) where T = FockNumber{T}(bitpermute(f.f, p))

shift_right(f::FockNumber, M) = FockNumber(f.f << M)


@testitem "Fock" begin
    using Random
    using FermionicHilbertSpaces: bits, _bit, focknbr_from_bits, focknbr_from_site_indices, focknbr_from_site_index
    Random.seed!(1234)

    N = 6
    @testset "FockNumber bits and construction" begin
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
    end

    @testset "removefermion" begin
        focknbr = FockNumber(rand(1:2^N) - 1)
        fockbits = bits(focknbr, N)
        function test_remove(n)
            FermionicHilbertSpaces.removefermion(n, focknbr) == (fockbits[n] ? (FockNumber(focknbr.f - 2^(n - 1)), (-1)^sum(fockbits[1:n-1])) : (FockNumber(0), 0))
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

    @testset "Fixed particle number Fock states" begin
        fs = FermionicHilbertSpaces.fixed_particle_number_fockstates(10, 5)
        @test length(fs) == binomial(10, 5)
        @test allunique(fs)
        @test all(FermionicHilbertSpaces.fermionnumber.(fs) .== 5)
    end

    @testset "Large site indices use BigInt" begin
        f = focknbr_from_site_indices((1, 1000,))
        @test f isa FockNumber{BigInt}
        @test f.f == 1 + BigInt(2)^(1000 - 1)
        @test focknbr_from_site_index(1000).f == f.f - 1
    end
end

_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)
_bit(f::Integer, k) = Bool((f >> (k - 1)) & 1)

@testitem "Split and join focknumbers" begin
    import FermionicHilbertSpaces: FockMapper, split_state, combine_states, focknbr_from_site_indices as fock
    fockmapper = FockMapper(((1, 3), (2, 4)))
    _split(state, fockmapper) = only(first(split_state(state, fockmapper)))
    _combine(states, fockmapper) = only(first(combine_states(states, fockmapper)))

    split = Base.Fix2(_split, fockmapper)
    combine = Base.Fix2(_combine, fockmapper)
    @test split(fock((1, 2, 3, 4))) == (fock((1, 2)), fock((1, 2)))
    @test split(fock((1,))) == (fock((1,)), fock(()))
    @test split(fock(())) == (fock(()), fock(()))
    @test split(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test split(fock((1, 3))) == (fock((1, 2)), fock(()))
    @test split(fock((2, 4))) == (fock(()), fock((1, 2)))
    @test split(fock((3, 2))) == (fock((2,)), fock((1,)))
    @test split(fock((3, 4))) == (fock((2,)), fock((2,)))

    # test all cases above with combiner
    @test fock((1, 2, 3, 4)) == combine((fock((1, 2)), fock((1, 2))))
    @test fock((1,)) == combine((fock((1,)), fock(())))
    @test fock(()) == combine((fock(()), fock(())))
    @test fock((1, 2, 3)) == combine((fock((1, 2)), fock((1,))))
    @test fock((1, 3)) == combine((fock((1, 2)), fock(())))
    @test fock((2, 4)) == combine((fock(()), fock((1, 2))))
    @test fock((3, 2)) == combine((fock((2,)), fock((1,))))
    @test fock((3, 4)) == combine((fock((2,)), fock((2,))))

    # test splitting with different sizes
    fockmapper = FockMapper(((1, 2), (3,)))
    split = Base.Fix2(_split, fockmapper)
    combine = Base.Fix2(_combine, fockmapper)
    @test split(fock((1, 2, 3))) == (fock((1, 2)), fock((1,)))
    @test split(fock((1, 3))) == (fock((1,)), fock((1,)))
    @test split(fock((1, 2))) == (fock((1, 2)), fock(()))
    @test split(fock((2,))) == (fock((2,)), fock(()))
    @test split(fock((2, 3))) == (fock((2,)), fock((1,)))
    @test split(fock((3,))) == (fock(()), fock((1)))

    # test all cases above with combine
    @test fock((1, 3)) == combine((fock((1,)), fock((1,))))
    @test fock((1, 2)) == combine((fock((1, 2)), fock(())))
    @test fock((2,)) == combine((fock((2,)), fock(())))
    @test fock((2, 3)) == combine((fock((2,)), fock((1,))))
    @test fock((3,)) == combine((fock(()), fock((1,))))

    # test splitting with different sizes
    fockmapper = FockMapper(((1, 4), (2,), (3,)))
    split = Base.Fix2(_split, fockmapper)
    combine = Base.Fix2(_combine, fockmapper)
    ident = combine ∘ split
    @test all(ident(FockNumber(k)) == FockNumber(k) for k in 0:2^4-1)

    # test subsystem splits
    fockmapper = FockMapper(((1, 3),))
    split = only ∘ Base.Fix2(_split, fockmapper)
    # split(state) = only(only(first(split_state(state, fockmapper))))
    @test split(fock((1, 2, 3, 4))) == fock((1, 2))
    @test split(fock((1,))) == fock((1,))
    @test split(fock(())) == fock(())
    @test split(fock((1, 2, 3))) == fock((1, 2))
    @test split(fock((1, 3))) == fock((1, 2))
    @test split(fock((2, 4))) == fock(())
    @test split(fock((3, 2))) == fock((2,))
    @test split(fock((3, 4))) == fock((2,))
end



@testitem "fockstates handles large M without overflow or duplicates" begin
    import FermionicHilbertSpaces: fixed_particle_number_fockstates
    # For large M, fockstates should use BigInt and not overflow
    M = 70
    n = 3
    states = fixed_particle_number_fockstates(M, n)
    # All states should be unique
    @test length(states) == length(unique(states))
    # All states should be non-negative
    @test all(f -> f.f >= 0, states)
    # Check that the type is FockNumber{BigInt} for large M
    @test all(f -> f isa FockNumber{BigInt}, states)
end
