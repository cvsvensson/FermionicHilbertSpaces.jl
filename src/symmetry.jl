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
- `focknumbers::IF`: A vector of Fock numbers, which are integers representing the occupation of each mode.
- `focktoinddict::FI`: A dictionary mapping Fock states to indices.
- `qntofockstates::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to Fock states.
- `conserved_quantity::QNfunc`: A function that computes the conserved quantity from a fock number.
"""
struct FockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry
    focknumbers::IF
    focktoinddict::FI
    qntofockstates::Dictionary{QN,Vector{FockNumber}}
    conserved_quantity::QNfunc
end

Base.:(==)(sym1::FockSymmetry, sym2::FockSymmetry) = sym1.focknumbers == sym2.focknumbers && sym1.focktoinddict == sym2.focktoinddict && sym1.qntofockstates == sym2.qntofockstates


"""
    focksymmetry(focknumbers, qn)

Constructs a `FockSymmetry` object that represents the symmetry of a many-body system. 

# Arguments
- `focknumbers`: The focknumbers to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function focksymmetry(focknumbers, qn)
    qntofockstates = group(f -> qn(f), focknumbers)
    sortkeys!(qntofockstates)
    ordered_fockstates = vcat(qntofockstates...)
    focktoinddict = Dictionary(ordered_fockstates, 1:length(ordered_fockstates))
    FockSymmetry(ordered_fockstates, focktoinddict, qntofockstates, qn)
end
focksymmetry(::AbstractVector, ::NoSymmetry) = NoSymmetry()
instantiate(::NoSymmetry, labels) = NoSymmetry()
indtofock(ind, sym::FockSymmetry) = FockNumber(sym.focknumbers[ind])
focktoind(f, sym::FockSymmetry) = sym.focktoinddict[f]
focknumbers(sym::FockSymmetry) = sym.focknumbers

focktoind(fs::FockNumber, ::NoSymmetry) = fs.f + 1
indtofock(ind, ::NoSymmetry) = FockNumber(ind - 1)

function nextfockstate_with_same_number(v)
    #http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = (v | (v - 1)) + 1
    t | (((div((t & -t), (v & -v))) >> 1) - 1)
end
"""
    fockstates(M, n)

Generate a list of Fock states with `n` occupied fermions in a system with `M` different fermions.
"""
function fockstates(M, n)
    v::Int = focknbr_from_bits(ntuple(i -> true, n)).f
    maxv = v * 2^(M - n)
    states = Vector{FockNumber}(undef, binomial(M, n))
    count = 1
    while v <= maxv
        states[count] = FockNumber(v)
        v = nextfockstate_with_same_number(v)
        count += 1
    end
    states
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
function instantiate_and_get_focknumbers(jw, _qn)
    qn = instantiate(_qn, jw)
    fs = focknumbers(jw, qn)
    return qn, fs
end
focknumbers(jw::JordanWignerOrdering, ::NoSymmetry) = map(FockNumber, 0:2^length(jw)-1)
function focknumbers(jw::JordanWignerOrdering, qn::ParityConservation)
    s = sectors(qn)
    fs = focknumbers(jw, NoSymmetry())
    ismissing(s) && return fs
    filt = in(s) ∘ parity
    filter!(filt, fs)
end
function focknumbers(jw::JordanWignerOrdering, qn::FermionConservation)
    s = sectors(qn)
    ismissing(s) && return focknumbers(jw, NoSymmetry())
    N = length(jw)
    mapreduce(n -> fockstates(N, n), vcat, s)
end
function focknumbers(jw::JordanWignerOrdering, qn::FermionSubsetConservation)
    s = sectors(qn)
    mask = qn.mask
    fs = focknumbers(jw, NoSymmetry())
    ismissing(s) && return fs
    filt = in(s) ∘ fermionnumber ∘ (f -> f & mask)
    filter!(filt, fs)
end
function focknumbers(jw::JordanWignerOrdering, sym::ProductSymmetry)
    filter!(f -> allowed_qn(sym(f), sym), focknumbers(jw, NoSymmetry()))
end

@testitem "Symmetry focknumbers" begin
    import FermionicHilbertSpaces: instantiate_and_get_focknumbers, fermionnumber, FermionSubsetConservation
    H = hilbert_space(1:5)
    @test length(collect(focknumbers(H.jw, ParityConservation()))) == 2^5
    @test length(collect(focknumbers(H.jw, ParityConservation(1)))) == 2^4
    odd_focks = focknumbers(H.jw, ParityConservation(-1))
    @test all(isodd ∘ fermionnumber, odd_focks)
    @test length(collect(focknumbers(H.jw, ParityConservation([-1, 1])))) == 2^5

    ## ProductSymmetry
    qn, fs = instantiate_and_get_focknumbers(H.jw, ParityConservation([1]) * FermionSubsetConservation(1:3, 1:1))
    @test all(iseven ∘ fermionnumber, fs)
    @test all(fermionnumber(f & qn.symmetries[2].mask) == 1 for f in fs)
end
