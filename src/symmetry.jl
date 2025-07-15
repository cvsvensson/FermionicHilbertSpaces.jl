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

function nextfockstate_with_same_number(v)
    #http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (v | (v - 1)) + 1
    t | (((div((t & -t), (v & -v))) >> 1) - 1)
end
"""
    fixed_particle_number_fockstates(M, n)

Generate a list of Fock states with `n` occupied fermions in a system with `M` different fermions.
"""
function fixed_particle_number_fockstates(M, n, ::Type{T}=(M > 63 ? BigInt : Int)) where T
    iszero(n) && return map(FockNumber{T}, [0])
    v = T(focknbr_from_bits([true for _ in 1:n]).f)
    maxv = v << (M - n)
    states = Vector{T}(undef, binomial(M, n))
    count = 1
    while v <= maxv
        states[count] = v
        v = nextfockstate_with_same_number(v)
        count += 1
    end
    map(FockNumber{T}, states)
end

"""
    FermionConservation
A symmetry type representing conservation of total fermion number.
"""
struct FermionConservation <: AbstractSymmetry
    sectors::Union{Vector{Int},Missing}
end
FermionConservation(s::Int) = FermionConservation([s])
FermionConservation() = FermionConservation(missing)
sectors(qn::FermionConservation) = qn.sectors

struct FermionSubsetConservation <: AbstractSymmetry
    mask::FockNumber
    sectors::Union{Vector{Int},Missing}
    function FermionSubsetConservation(mask::FockNumber, sectors)
        allunique(sectors) || throw(ArgumentError("FermionSubsetConservation sectors must be unique."))
        sectorvec = sort!(Int[sectors...])
        new(mask, sectorvec)
    end
    FermionSubsetConservation(mask::FockNumber, sectors::Missing) = new(mask, missing)
end
sectors(qn::FermionSubsetConservation) = qn.sectors

struct UninstantiatedFermionSubsetConservation{L} <: AbstractSymmetry
    labels::L
    sectors::Union{Vector{Int},Missing}
    function UninstantiatedFermionSubsetConservation(labels::L, sectors=missing) where L
        ismissing(sectors) && new{L}(labels, missing)
        allunique(labels) || throw(ArgumentError("FermionSubsetConservation labels must be unique."))
        allunique(sectors) || throw(ArgumentError("FermionSubsetConservation sectors must be unique."))
        allowed_qns = Set(0:length(labels))
        all(in(allowed_qns), sectors) || throw(ArgumentError("FermionSubsetConservation can only have sectors 0 up to the number of particles."))
        sectorvec = sort!(Int[sectors...])
        new{L}(labels, sectorvec)
    end

end
FermionSubsetConservation(::Nothing) = NoSymmetry()
FermionSubsetConservation(labels, jw::JordanWignerOrdering, sectors=0:length(labels)) = FermionSubsetConservation(focknbr_from_site_labels(labels, jw), sectors)
FermionSubsetConservation(labels, sectors=0:length(labels)) = UninstantiatedFermionSubsetConservation(labels, sectors)
instantiate(qn::UninstantiatedFermionSubsetConservation, jw::JordanWignerOrdering) = FermionSubsetConservation(qn.labels, jw, qn.sectors)
instantiate(qn::FermionSubsetConservation, ::JordanWignerOrdering) = qn
instantiate(qn::FermionConservation, ::JordanWignerOrdering) = qn

(qn::FermionSubsetConservation)(fs) = fermionnumber(fs, qn.mask)
(qn::FermionConservation)(fs) = fermionnumber(fs)

@testitem "ConservedFermions" begin
    import FermionicHilbertSpaces: FermionSubsetConservation
    labels = 1:4
    conservedlabels = 1:4
    c1 = hilbert_space(labels, FermionSubsetConservation(conservedlabels))
    c2 = hilbert_space(labels, FermionConservation())
    @test c1 == c2

    conservedlabels = 2:2
    c1 = hilbert_space(labels, FermionSubsetConservation(conservedlabels))
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
    import FermionicHilbertSpaces: FermionSubsetConservation
    labels = 1:4
    qn = FermionConservation() * ParityConservation()
    H = hilbert_space(labels, qn)
    @test keys(H.symmetry.qntofockstates).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(FermionSubsetConservation([l], H.jw) for l in labels)
    @test all(length.(hilbert_space(labels, qn).symmetry.qntofockstates) .== 1)
end

"""
    IndexConservation
A symmetry type representing conservation of the numbers of modes which contains a specific index or set of indices."""
struct IndexConservation{L} <: AbstractSymmetry
    labels::L
    sectors::Union{Vector{Int},Missing}
    function IndexConservation(labels::L, sectors) where L
        ismissing(sectors) && return new{L}(labels, missing)
        allunique(labels) || throw(ArgumentError("IndexConservation labels must be unique."))
        allunique(sectors) || throw(ArgumentError("IndexConservation sectors must be unique."))
        allowed_qns = Set(0:length(labels))
        all(in(allowed_qns), sectors) || throw(ArgumentError("IndexConservation can only have sectors 0 up to the number of particles."))
        sectorvec = sort!(Int[sectors...])
        new{L}(labels, sectorvec)
    end
end
sectors(qn::IndexConservation) = qn.sectors
IndexConservation(labels) = IndexConservation(labels, missing)
instantiate(qn::IndexConservation, jw::JordanWignerOrdering) = IndexConservation(qn.labels, jw, qn.sectors)
IndexConservation(index, jw::JordanWignerOrdering, sectors) = FermionSubsetConservation(filter(label -> index in label || index == label, jw.labels), jw, sectors)
@testitem "IndexConservation" begin
    import FermionicHilbertSpaces: FermionSubsetConservation
    labels = 1:4
    qn = IndexConservation(1)
    qn2 = FermionSubsetConservation(1:1)
    H = hilbert_space(labels, qn)
    H2 = hilbert_space(labels, qn2)
    @test H == H2

    spatial_labels = 1:1
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = IndexConservation(:↑) * IndexConservation(:↓)
    H = hilbert_space(all_labels, qn)
    @test all(length.(H.symmetry.qntofockstates) .== 1)
end

instantiate(f::F, labels) where {F} = f


promote_symmetry(s1::FockSymmetry{<:Any,<:Any,<:Any,F}, s2::FockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(s1::FockSymmetry{<:Any,<:Any,<:Any,F1}, s2::FockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = s1 == s2 ? s1.conserved_quantity : NoSymmetry()
promote_symmetry(::NoSymmetry, ::S) where {S} = NoSymmetry()
promote_symmetry(::S, ::NoSymmetry) where {S} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::NoSymmetry) = NoSymmetry()

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
basisstates(jw::JordanWignerOrdering, ::NoSymmetry) = map(FockNumber, 0:2^length(jw)-1)
function basisstates(jw::JordanWignerOrdering, qn::ParityConservation)
    s = sectors(qn)
    fs = basisstates(jw, NoSymmetry())
    ismissing(s) && return fs
    filt = in(s) ∘ parity
    filter!(filt, fs)
end
function basisstates(jw::JordanWignerOrdering, qn::FermionConservation)
    s = sectors(qn)
    ismissing(s) && return basisstates(jw, NoSymmetry())
    N = length(jw)
    mapreduce(n -> fixed_particle_number_fockstates(N, n), vcat, s)
end
function basisstates(jw::JordanWignerOrdering, qn::FermionSubsetConservation)
    s = sectors(qn)
    mask = qn.mask
    fs = basisstates(jw, NoSymmetry())
    ismissing(s) && return fs
    filt = in(s) ∘ fermionnumber ∘ (f -> f & mask)
    filter!(filt, fs)
end
function basisstates(jw::JordanWignerOrdering, sym::ProductSymmetry)
    filter!(f -> allowed_qn(sym(f), sym), basisstates(jw, NoSymmetry()))
end

@testitem "Symmetry basisstates" begin
    import FermionicHilbertSpaces: instantiate_and_get_basisstates, fermionnumber, FermionSubsetConservation
    H = hilbert_space(1:5)
    @test length(collect(basisstates(H.jw, ParityConservation()))) == 2^5
    @test length(collect(basisstates(H.jw, ParityConservation(1)))) == 2^4
    odd_focks = basisstates(H.jw, ParityConservation(-1))
    @test all(isodd ∘ fermionnumber, odd_focks)
    @test length(collect(basisstates(H.jw, ParityConservation([-1, 1])))) == 2^5

    ## ProductSymmetry
    qn, fs = instantiate_and_get_basisstates(H.jw, ParityConservation([1]) * FermionSubsetConservation(1:3, 1:1))
    @test all(iseven ∘ fermionnumber, fs)
    @test all(fermionnumber(f & qn.symmetries[2].mask) == 1 for f in fs)
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

    # Test with FermionConservation
    qn_f = FermionConservation([1, 2])
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
