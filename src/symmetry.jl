abstract type AbstractSymmetry end

"""
    NoSymmetry
A symmetry type indicating no symmetry constraints.
"""
struct NoSymmetry <: AbstractSymmetry end

"""
    struct FockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry

FockSymmetry represents a symmetry that is diagonal in fock space, i.e. particle number conservation, parity, spin consvervation.

## Fields
- `basisstates::IF`: A vector of Fock numbers, which are integers representing the occupation of each mode.
- `state_indexdict::FI`: A dictionary mapping Fock states to indices.
- `qntofockstates::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to Fock states.
- `conserved_quantity::QNfunc`: A function that computes the conserved quantity from a fock number.
"""
struct FockSymmetry{IF,FI,QN,I,QNfunc} <: AbstractSymmetry
    basisstates::IF
    state_indexdict::FI
    qntofockstates::Dictionary{QN,Vector{FockNumber{I}}}
    conserved_quantity::QNfunc
end

Base.:(==)(sym1::FockSymmetry, sym2::FockSymmetry) = sym1.basisstates == sym2.basisstates && sym1.state_indexdict == sym2.state_indexdict && sym1.qntofockstates == sym2.qntofockstates


"""
    focksymmetry(basisstates, qn)

Constructs a `FockSymmetry` object that represents the symmetry of a many-body system. 

# Arguments
- `basisstates`: The basisstates to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function focksymmetry(basisstates, qn)
    qntofockstates = group(f -> qn(f), basisstates)
    sortkeys!(qntofockstates)
    ordered_fockstates = vcat(qntofockstates...)
    state_indexdict = Dictionary(ordered_fockstates, 1:length(ordered_fockstates))
    FockSymmetry(ordered_fockstates, state_indexdict, qntofockstates, qn)
end
focksymmetry(::AbstractVector, ::NoSymmetry) = NoSymmetry()
instantiate(::NoSymmetry, labels) = NoSymmetry()
basisstate(ind, sym::FockSymmetry) = FockNumber(sym.basisstates[ind])
state_index(f, sym::FockSymmetry) = get(sym.state_indexdict, f, missing)
# state_index(f, sym::FockSymmetry) = sym.state_indexdict[f]
basisstates(sym::FockSymmetry) = sym.basisstates

state_index(fs::FockNumber, ::NoSymmetry) = fs.f + 1
basisstate(ind, ::NoSymmetry) = FockNumber(ind - 1)

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

"""
    NumberConservation
A symmetry type representing conservation of total fermion number.
"""
struct NumberConservation <: AbstractSymmetry
    sectors::Union{Vector{Int},Missing}
end
NumberConservation(s::Int) = NumberConservation([s])
NumberConservation() = NumberConservation(missing)
sectors(qn::NumberConservation) = qn.sectors

instantiate(qn::NumberConservation, ::JordanWignerOrdering) = qn

(qn::NumberConservation)(f) = fermionnumber(f)

@testitem "ConservedFermions" begin
    import FermionicHilbertSpaces: number_conservation
    labels = 1:4
    conservedlabels = 1:4
    c1 = hilbert_space(labels, number_conservation(label -> label in conservedlabels))
    c2 = hilbert_space(labels, number_conservation())
    @test c1 == c2

    conservedlabels = 2:2
    c1 = hilbert_space(labels, number_conservation(label -> label in conservedlabels))
    @test all(length.(c1.symmetry.qntofockstates) .== 2^(length(labels) - length(conservedlabels)))
end

struct ProductSymmetry{T} <: AbstractSymmetry
    symmetries::T
end
instantiate(qn::ProductSymmetry, labels) = prod(instantiate(sym, labels) for sym in qn.symmetries)
(qn::ProductSymmetry)(fs) = map(sym -> sym(fs), qn.symmetries)
Base.:*(sym1::AbstractSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1, sym2))
Base.:*(sym1::AbstractSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1, sym2.symmetries...))
Base.:*(sym1::ProductSymmetry, sym2::AbstractSymmetry) = ProductSymmetry((sym1.symmetries..., sym2))
Base.:*(sym1::ProductSymmetry, sym2::ProductSymmetry) = ProductSymmetry((sym1.symmetries..., sym2.symmetries...))

"""
    ParityConservation
A symmetry type representing conservation of fermion parity.
"""
struct ParityConservation <: AbstractSymmetry
    sectors::Vector{Int}
    function ParityConservation(sectors)
        length(sectors) <= 2 || throw(ArgumentError("ParityConservation can only have two sectors, -1 and 1."))
        allunique(sectors) || throw(ArgumentError("ParityConservation sectors must be unique."))
        allowed_qns = Set((-1, 1))
        all(in(allowed_qns), sectors) || throw(ArgumentError("ParityConservation can only have sectors -1 and 1."))
        sectorvec = sort!(Int[sectors...])
        new(sectorvec)
    end
end
(qn::ParityConservation)(fs) = parity(fs)
instantiate(qn::ParityConservation, labels) = qn
ParityConservation() = ParityConservation([-1, 1])
sectors(qn::ParityConservation) = qn.sectors

@testitem "ProductSymmetry" begin
    import FermionicHilbertSpaces: number_conservation
    labels = 1:4
    qn = number_conservation() * ParityConservation()
    H = hilbert_space(labels, qn)
    @test keys(H.symmetry.qntofockstates).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(number_conservation(label -> label == l) for l in labels)
    @test all(length.(hilbert_space(labels, qn).symmetry.qntofockstates) .== 1)
end


@testitem "IndexConservation" begin
    import FermionicHilbertSpaces: number_conservation
    labels = 1:4
    qn = number_conservation(==(1))
    qn2 = number_conservation(label -> label in 1:1)
    H = hilbert_space(labels, qn)
    H2 = hilbert_space(labels, qn2)
    @test H == H2

    spatial_labels = 1:1
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = number_conservation(label -> label[2] == :↑) * number_conservation(label -> label[2] == :↓)
    H = hilbert_space(all_labels, qn)
    @test all(length.(H.symmetry.qntofockstates) .== 1)

    spatial_labels = 1:2
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = number_conservation(1, label -> label[2] == :↑)
    H = hilbert_space(all_labels, qn)
    @test length(basisstates(H)) == 2^3
end

function allowed_qn(qn, sym::ProductSymmetry)
    for (q, sym) in zip(qn, sym.symmetries)
        allowed_qn(q, sym) || return false
    end
    return true
end
function allowed_qn(qn, sym::AbstractSymmetry)
    ismissing(sectors(sym)) && return true
    in(qn, sectors(sym)) || return false
    return true
end
function instantiate_and_get_basisstates(jw::JordanWignerOrdering, _qn)
    qn = instantiate(_qn, jw)
    fs = basisstates(jw, qn)
    return qn, fs
end
basisstates(jw::JordanWignerOrdering, ::NoSymmetry) = map(FockNumber, UnitRange{UInt64}(0, 2^length(jw) - 1))
function basisstates(jw::JordanWignerOrdering, qn::ParityConservation)
    s = sectors(qn)
    fs = basisstates(jw, NoSymmetry())
    ismissing(s) && return fs
    filt = in(s) ∘ parity
    filter!(filt, fs)
end
function basisstates(jw::JordanWignerOrdering, qn::NumberConservation)
    s = sectors(qn)
    ismissing(s) && return basisstates(jw, NoSymmetry())
    N = length(jw)
    mapreduce(n -> fixed_particle_number_fockstates(N, n), vcat, s)
end

function basisstates(jw::JordanWignerOrdering, sym::ProductSymmetry)
    filter!(f -> allowed_qn(sym(f), sym), basisstates(jw, first(sym.symmetries)))
end

struct UninstantiatedNumberConservations{F,S} <: AbstractSymmetry
    weight_functions::F
    sectors::S
end
struct NumberConservations{M,S} <: AbstractSymmetry
    weights::M
    sectors::S
    function NumberConservations(weights::M, sectors, N) where M
        length(weights) == length(sectors) || throw(ArgumentError("Number of masks must match number of sectors."))
        canon_sectors = map((s, m) -> collect(canonicalize_sector(s, m, N)), sectors, weights)
        # check if all weights are 0 or 1, if so, convert to Integer masks
        for m in weights
            all(w -> w == 0 || w == 1, m) && continue
            return new{M,typeof(canon_sectors)}(weights, canon_sectors)
        end
        intmasks = map(m -> integer_from_bits(m, default_fock_representation(N)), weights)
        new{typeof(intmasks),typeof(canon_sectors)}(intmasks, canon_sectors)
    end
end
instantiate(qn::NumberConservations, ::JordanWignerOrdering) = qn
Base.show(io::IO, qn::NumberConservations) = print(io, "Number conservation for ", length(qn.weights), " subsets")
(qn::NumberConservations)(f::FockNumber) = length(qn.weights) == 1 ? fermionnumber(f, only(qn.weights)) : map((m -> fermionnumber(f, m)), qn.weights)
(qn::NumberConservations)(f::Integer) = length(qn.weights) == 1 ? count_weighted_ones(f, only(qn.weights)) : map((m -> count_weighted_ones(f, m)), qn.weights)


"""
    number_conservation(sectors=missing, weight_function=x -> true)

Constructs a `UninstantiatedNumberConservations` symmetry object that represents conservation of fermion number. 'sectors' can be an integer or a collection of integers specifying the allowed fermion numbers. 'weight_function' is a function that takes a label and returns an integer weight (which can be negative) indicating how that label contributes to the fermion number.
"""
function number_conservation(sectors=missing, weight_function=x -> true)
    if weight_function == x -> true
        return NumberConservation(sectors)
    end
    UninstantiatedNumberConservations((weight_function,), (sectors,))
end
number_conservation(weight_function::F) where F<:Function = number_conservation(missing, weight_function)

Base.:*(qn1::UninstantiatedNumberConservations, qn2::UninstantiatedNumberConservations) = UninstantiatedNumberConservations((qn1.weight_functions..., qn2.weight_functions...), (qn1.sectors..., qn2.sectors...))

canonicalize_sector(::Missing, mask, N) = 0:mask_region_size(mask)
canonicalize_sector(sectors::Integer, mask, N) = (sectors,)
canonicalize_sector(sector::AbstractVector{<:Integer}, mask, N) = sector
canonicalize_sector(sector::NTuple{M,<:Integer}, mask, N) where M = sector

default_fock_representation(N) = N < 64 ? UInt64 : BigInt

instantiate(qn::UninstantiatedNumberConservations, jw::JordanWignerOrdering,) = instantiate(qn, keys(jw))
function instantiate(qn::UninstantiatedNumberConservations, labels)
    N = length(labels)
    # T = default_fock_representation(N)
    weights = map(f -> map(l -> f(l), labels), qn.weight_functions)
    NumberConservations(weights, qn.sectors, N)
end
basisstates(jw::JordanWignerOrdering, qn::NumberConservations, ::Type{F}=FockNumber) where F = sort!(map(F, generate_states(qn.weights, qn.sectors, length(jw))))
function allowed_qn(qn, sym::NumberConservations)
    for (q, sector) in zip(qn, sym.sectors)
        ismissing(sector) && continue
        in(q, sector) || return false
    end
    return true
end

@testitem "Symmetry basisstates" begin
    import FermionicHilbertSpaces: instantiate_and_get_basisstates, fermionnumber, number_conservation
    H = hilbert_space(1:5)
    @test length(collect(basisstates(H.jw, ParityConservation()))) == 2^5
    @test length(collect(basisstates(H.jw, ParityConservation(1)))) == 2^4
    odd_focks = basisstates(H.jw, ParityConservation(-1))
    @test all(isodd ∘ fermionnumber, odd_focks)
    @test length(collect(basisstates(H.jw, ParityConservation([-1, 1])))) == 2^5

    ## ProductSymmetry
    qn, fs = instantiate_and_get_basisstates(H.jw, ParityConservation([1]) * number_conservation(1:2, in(1:3)))
    H2 = hilbert_space(1:5, qn, fs)
    @test sort(basisstates(H2)) == sort(fs)
    @test all(iseven ∘ fermionnumber, fs)
    @test all(fermionnumber(f, qn.symmetries[2].weights[1]) in 1:2 for f in fs)
end

@testitem "sector" begin
    import FermionicHilbertSpaces: sector
    # Create a Hilbert space with parity symmetry
    labels = 1:3
    qn = ParityConservation()
    H = hilbert_space(labels, qn)
    n = length(basisstates(H.symmetry))
    m = reshape(1:(n^2), n, n)  # simple test matrix
    # Get the sector for parity = 1
    even_sector = sector(m, 1, H)
    # The size of the even sector block should match the number of even-parity states
    even_states = [f for f in basisstates(H) if qn(f) == 1]
    @test size(even_sector, 1) == length(even_states)
    @test size(even_sector, 2) == length(even_states)
    # The values should match the corresponding block in m
    # Get the indices of even states in the full basisstates list
    even_inds = findall(f -> qn(f) == 1, basisstates(H.symmetry))
    @test even_sector == m[even_inds, even_inds]
    # Test that an invalid sector throws an error
    @test_throws ArgumentError sector(m, 99, H)

    # Test with NumberConservation
    qn_f = number_conservation([1, 2])
    Hf = hilbert_space(labels, qn_f)
    n_f = length(basisstates(Hf.symmetry))
    m_f = reshape(1:(n_f^2), n_f, n_f)
    # Test sector for fermion number = 1
    sector1 = sector(m_f, 1, Hf)
    states1 = [f for f in basisstates(Hf) if qn_f(f) == 1]
    inds1 = findall(f -> qn_f(f) == 1, basisstates(Hf.symmetry))
    @test size(sector1, 1) == length(states1)
    @test size(sector1, 2) == length(states1)
    @test sector1 == m_f[inds1, inds1]
    # Test sector for fermion number = 2
    sector2 = sector(m_f, 2, Hf)
    states2 = [f for f in basisstates(Hf) if qn_f(f) == 2]
    inds2 = findall(f -> qn_f(f) == 2, basisstates(Hf.symmetry))
    @test size(sector2, 1) == length(states2)
    @test size(sector2, 2) == length(states2)
    @test sector2 == m_f[inds2, inds2]
    # Test that an invalid sector throws an error
    @test_throws ArgumentError sector(m_f, 99, Hf)
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
