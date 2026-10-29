"""
    FockNumber
A type representing a Fock state as the bitstring of an integer. Bit `k - 1` (counting from the
least significant bit) is the occupation of the `k`-th mode of the space, in the order of the labels
given to `hilbert_space`. In an unconstrained space the basis state at index `i` is
`FockNumber(i - 1)`.
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

"""
    default_fock_representation(N)

The integer type for Fock states of `N` modes: `UInt64` below 63 modes, so that `2^N` fits in
an `Int`, otherwise the smallest of `UInt128` to `UInt8192` with more than `N` bits, or `BigInt`.
"""
function default_fock_representation(N)
    N < 63 && return UInt64
    N < 128 && return UInt128
    N < 256 && return UInt256
    N < 512 && return UInt512
    N < 1024 && return UInt1024
    N < 2048 && return UInt2048
    N < 4096 && return UInt4096
    N < 8192 && return UInt8192
    return BigInt
end
default_fock_representation(::Val{N}) where N = default_fock_representation(N)

"""
    fock_index_type(I)

The integer type of the dimension and state indices of a space with Fock states stored in `I`.
"""
fock_index_type(::Type{I}) where I = sizeof(I) <= sizeof(Int) ? Int : I
fock_index_type(::Type{BigInt}) = BigInt

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
_nbr_differing_modes(f1::FockNumber, f2::FockNumber, N::Int) = count_ones(f1.f ⊻ f2.f)

internal_rep(state::FockNumber, ::AbstractHilbertSpace, ::Type{T}) where T<:Integer = T(state.f)
physical_rep(state::T, ::Type{FockNumber{I}}) where {T,I} = FockNumber{I}(state)

fermionnumber(f::FockNumber{<:Integer}, mask) = count_weighted_ones(f.f, mask)
count_weighted_ones(x, mask::Integer) = count_ones(x & mask)
count_weighted_ones(x, weights::Union{Vector,Tuple}) = sum(w for (i, w) in enumerate(weights) if _bit(x, i))

"""
    jwstring(site, focknbr)
    
Parity of the number of fermions to the right of site.
"""
jwstring(site, focknbr) = jwstring_left(site, focknbr)
jwstring_anti(site, focknbr) = jwstring_right(site, focknbr)
jwstring_bool(site, focknbr) = jwstring_left_bool(site, focknbr)
jwstring_anti_bool(site, focknbr) = jwstring_right_bool(site, focknbr)
jwstring_right(site, focknbr) = jwstring_right_bool(site, focknbr) ? -1 : 1
jwstring_left(site, focknbr) = jwstring_left_bool(site, focknbr) ? -1 : 1
jwstring_right_bool(site, focknbr::FockNumber) = isodd(count_ones(focknbr.f >> site)) # true represents -1
jwstring_left_bool(site, focknbr::FockNumber) = isodd(count_ones(focknbr.f) - count_ones(focknbr.f >> (site - 1)))


struct FockMapper{N,P1,W,P2,M} <: AbstractStateMapper
    fermionpositions::P1
    widths::W
    permutation::P2
    masks::M
    isfullpartition::Bool
end
# `N` is the number of modes of the full space and sets the integer type of combined states.
# The bit permutation is only built when the subsystems cover all `N` modes.
FockMapper(fermionpositions) = FockMapper(fermionpositions, maximum(maximum, fermionpositions))
function FockMapper(fermionpositions::P, N) where P
    widths = map(length, fermionpositions)
    Nmax = maximum(maximum, fermionpositions)
    Nmax <= N || throw(ArgumentError("Fermion positions must be at most the number of modes $N, got $fermionpositions."))
    perm = mapreduce(collect, vcat, fermionpositions)
    all(issorted, fermionpositions) || throw(ArgumentError("The fermions in each subsystem must be ordered as in the full system, got unsorted fermion positions $fermionpositions."))
    T = default_fock_representation(N)
    permutation = length(perm) == N && isperm(perm) && N < 64 ? BitPermutation{T}(perm)' : nothing
    masks = map(Xp -> focknbr_from_site_indices(Xp), fermionpositions)
    isfullpartition = ispartition(fermionpositions, N)
    P1 = typeof(fermionpositions)
    W = typeof(widths)
    P2 = typeof(permutation)
    M = typeof(masks)
    FockMapper{N,P1,W,P2,M}(fermionpositions, widths, permutation, masks, isfullpartition)
end
unique_split(::FockMapper) = true
unique_combine(::FockMapper) = true
function Base.show(io::IO, fm::FockMapper{N}) where N
    print(io, "FockMapper($N modes → ")
    print(io, join(["[" * join(pos, ",") * "]" for pos in fm.fermionpositions], " ⊗ "))
    print(io, ")")
end

function split_state(f::FockNumber, fm::FockMapper{N}) where N
    state = map(mask -> substate(mask, f), fm.masks)
    (state,), (1,)
end

function substate(mask::FockNumber{T1}, f::FockNumber{T2}) where {T1,T2}
    T = promote_type(T1, T2)
    return FockNumber{T}(BitPermutations.pext(T(f.f), T(mask.f)))
end
function _require_full_partition(fm::FockMapper)
    fm.isfullpartition && return fm
    throw(ArgumentError("`combine_states` requires a full non-overlapping partition, but `split_state` also accepts partial or overlapping fermion subregions, got $(fm.fermionpositions)."))
end

function combine_states(fs, fm::FockMapper{N,<:Any,<:Any,<:AbstractBitPermutation}) where {N}
    _require_full_partition(fm)
    state = concatenate_and_permute(fs, fm.widths, fm.permutation, FockNumber{default_fock_representation(Val(N))})
    (state,), (1,)
end
function combine_states(f, fm::FockMapper{N}) where N
    _require_full_partition(fm)
    IT = default_fock_representation(Val(N))
    result = zero(IT)
    for (fock, mask) in zip(f, fm.masks)
        result |= BitPermutations.pdep(IT(fock.f), IT(mask.f))
    end
    (FockNumber{IT}(result),), (1,)
end
function catenate_fock_states(fock_states, spaces, ::Type{T}) where T
    num = zero(T)
    shift = 0
    for (state, space) in zip(fock_states, spaces)
        num |= convert(T, state) << shift
        shift += nbr_of_modes(space)
    end
    shift <= _max_fock_modes(T) || throw(ArgumentError("Cannot store $shift modes in a $T."))
    num
end
_max_fock_modes(::Type{FockNumber{I}}) where {I<:Integer} = isbitstype(I) ? 8 * sizeof(I) - (I <: Signed) : typemax(Int)

function concatenate_and_permute(fs, widths, permutation, ::Type{T}) where T
    mask = foldl(concatenate, zip(fs, widths); init=(zero(T), 0)) |> first
    permute(mask, permutation)
end
function concatenate((lastf, lastwidth)::Tuple{FockNumber,Int}, (f, width)::Tuple{FockNumber,Int})
    return (lastf | (f << lastwidth), lastwidth + width)
end

permute(f::FockNumber{T}, p) where T = FockNumber{T}(bitpermute(f.f, p))

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
        focknbr = FockNumber(rand(1:(2^N)) - 1)
        fockbits = bits(focknbr, N)
        function test_remove(n)
            FermionicHilbertSpaces.removefermion(n, focknbr) == (fockbits[n] ? (FockNumber(focknbr.f - 2^(n - 1)), (-1)^sum(fockbits[1:(n-1)])) : (FockNumber(0), 0))
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

    @testset "Large site indices" begin
        for site in (100, 1000, 2000)
            f = focknbr_from_site_indices((1, site))
            @test f.f == 1 + big(2)^(site - 1)
            @test focknbr_from_site_index(site).f == f.f - 1
        end
    end
end

_bit(f::FockNumber, k) = Bool((f.f >> (k - 1)) & 1)
_bit(f::Integer, k) = Bool((f >> (k - 1)) & 1)

@testitem "FockMapper partitions" begin
    import FermionicHilbertSpaces: FockMapper, split_state, combine_states, focknbr_from_site_indices as fock
    _split(state, fockmapper) = only(first(split_state(state, fockmapper)))
    _combine(states, fockmapper) = only(first(combine_states(states, fockmapper)))

    @testset "full partitions" begin
        fockmapper = FockMapper(((1, 3), (2, 4)))
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

        @test fock((1, 2, 3, 4)) == combine((fock((1, 2)), fock((1, 2))))
        @test fock((1,)) == combine((fock((1,)), fock(())))
        @test fock(()) == combine((fock(()), fock(())))
        @test fock((1, 2, 3)) == combine((fock((1, 2)), fock((1,))))
        @test fock((1, 3)) == combine((fock((1, 2)), fock(())))
        @test fock((2, 4)) == combine((fock(()), fock((1, 2))))
        @test fock((3, 2)) == combine((fock((2,)), fock((1,))))
        @test fock((3, 4)) == combine((fock((2,)), fock((2,))))

        fockmapper = FockMapper(((1, 4), (2,), (3,)))
        split = Base.Fix2(_split, fockmapper)
        combine = Base.Fix2(_combine, fockmapper)
        ident = combine ∘ split
        @test all(ident(FockNumber(k)) == FockNumber(k) for k in 0:(2^4-1))
    end

    @testset "partial partitions" begin
        fockmapper = FockMapper(((1, 3),))
        split = only ∘ Base.Fix2(_split, fockmapper)
        @test split(fock((1, 2, 3, 4))) == fock((1, 2))
        @test split(fock((1,))) == fock((1,))
        @test split(fock(())) == fock(())
        @test split(fock((1, 2, 3))) == fock((1, 2))
        @test split(fock((1, 3))) == fock((1, 2))
        @test split(fock((2, 4))) == fock(())
        @test split(fock((3, 2))) == fock((2,))
        @test split(fock((3, 4))) == fock((2,))
        @test_throws ArgumentError _combine((fock((1, 2)),), fockmapper)
    end

    @testset "overlapping partitions" begin
        fockmapper = FockMapper(((1, 2), (2, 3)))
        split = Base.Fix2(_split, fockmapper)
        @test split(fock((1, 2, 3))) == (fock((1, 2)), fock((1, 2)))
        @test split(fock((2,))) == (fock((2,)), fock((1,)))
        @test split(fock((1, 3))) == (fock((1,)), fock((2,)))
        @test_throws ArgumentError _combine((fock((1, 2)), fock((1, 2))), fockmapper)
    end

    @testset "number of modes of the full space" begin
        # Mappers for different subsystems of one space have the same type
        @test typeof(FockMapper(([1, 2],), 6)) == typeof(FockMapper(([3, 5],), 6)) == typeof(FockMapper(([4, 6],), 6))
        @test_throws ArgumentError FockMapper(((1, 7),), 6)
        @test_throws ArgumentError _combine((fock((1, 2)), fock((1,))), FockMapper(((1, 3), (2,)), 70))
        fockmapper = FockMapper(((1, 3), (2,)), 3)
        @test _combine((fock((1, 2)), fock((1,))), fockmapper) == fock((1, 2, 3))
        @test !isnothing(fockmapper.permutation)
    end

    @testset "wide integers" begin
        for N in (100, 200, 400, 800, 1100)
            fermionpositions = (collect(1:3:N), setdiff(1:N, 1:3:N))
            fockmapper = FockMapper(fermionpositions, N)
            sites = [n for n in 1:N if n % 5 in (0, 2)]
            substates = map(pos -> fock(findall(in(sites), pos)), fermionpositions)
            @test _split(fock(sites), fockmapper) == substates
            @test _combine(substates, fockmapper) == fock(sites)
        end
    end

    @testset "catenate_fock_states" begin
        catenate_fock_states = FermionicHilbertSpaces.catenate_fock_states
        @fermions g
        modes = [hilbert_space(g[k]) for k in 1:64]
        states = [FockNumber{UInt64}(isodd(k) || k == 64) for k in 1:64]
        expected = sum(big(2)^(k - 1) for k in 1:64 if isodd(k) || k == 64)
        @test catenate_fock_states(states, modes, FockNumber{UInt64}) isa FockNumber{UInt64}
        @test catenate_fock_states(states, modes, FockNumber{UInt64}).f == expected
        @test catenate_fock_states(states, modes, FockNumber{BigInt}) == FockNumber(expected)
        @test catenate_fock_states(states[1:63], modes[1:63], FockNumber{Int}) isa FockNumber{Int}
        @test catenate_fock_states(states[1:63], modes[1:63], FockNumber{Int}).f == expected - big(2)^63
        @test_throws ArgumentError catenate_fock_states(states, modes, FockNumber{Int})
    end
end



@testitem "fockstates handles large M without overflow or duplicates" begin
    import FermionicHilbertSpaces: fixed_particle_number_fockstates
    for (M, n, T) in ((70, 3, UInt128), (1100, 1, FermionicHilbertSpaces.UInt2048), (8200, 1, BigInt))
        states = fixed_particle_number_fockstates(M, n)
        @test length(states) == binomial(M, n)
        @test allunique(states)
        @test all(f -> f isa FockNumber{T}, states)
    end
end

@testitem "Number-conserving space with more than 1024 modes" begin
    @fermions f
    M = 1100
    ham = sum(f[n+1]'f[n] + hc for n in 1:M-1)
    H = hilbert_space(f, 1:M, NumberConservation(1))
    @test representation(ham, H) == representation(ham, hilbert_space(f, 1:M, SingleParticle()))
end

@testitem "Spaces with more than 64 modes" begin
    @fermions f
    N = 70
    H = hilbert_space(f, 1:N, NumberConservation(1))
    Hsubsmall = subregion([f[n] for n in 1:10], H)
    Hsubbig = subregion([f[n] for n in 11:N], H)
    Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsubsmall)
    @test Hcomp == Hsubbig

    emb = embed(Hsubsmall => H; complement=Hsubbig)
    pt = partial_trace(H => Hsubsmall)
    @test emb.map == pt'.map

    emb = embed(Hsubbig => H; complement=Hsubsmall)
    pt = partial_trace(H => Hsubbig)
    @test emb.map == pt'.map
end

@testitem "Spaces with more than 128 modes" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: focknbr_from_site_index, SubsystemPartialTraceAlg, FullPartialTraceAlg
    @fermions f
    N = 200
    H = hilbert_space(f, 1:N, NumberConservation(1))
    hopping = representation(sum(f[n]' * f[n+1] + hc for n in 1:N-1), H)
    @test eigvals(Hermitian(Matrix(hopping))) ≈ sort(2cos.(π * (1:N) / (N + 1)))

    # A one-particle state reduced to every third mode has eigenvalues p and 1 - p
    subsites = 1:3:N
    Hsub = subregion([f[n] for n in subsites], H)
    ψ = normalize(sin.(1:dim(H)))
    p = sum(abs2(ψ[state_index(focknbr_from_site_index(n), H)]) for n in subsites)
    for alg in (SubsystemPartialTraceAlg(), FullPartialTraceAlg())
        ρ = partial_trace(ψ * ψ', H => Hsub; alg)
        @test sort(eigvals(Hermitian(ρ)))[end-1:end] ≈ sort([p, 1 - p])
        @test tr(ρ) ≈ 1
    end
end