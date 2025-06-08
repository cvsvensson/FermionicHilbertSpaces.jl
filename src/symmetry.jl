abstract type AbstractSymmetry end
struct NoSymmetry <: AbstractSymmetry end

"""
    struct FockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry

FockSymmetry represents a symmetry that is diagonal in fock space, i.e. particle number conservation, parity, spin consvervation.

## Fields
- `indtofockdict::IF`: A dictionary mapping indices to Fock states.
- `focktoinddict::FI`: A dictionary mapping Fock states to indices.
- `qntoblocksizes::Dictionary{QN,Int}`: A dictionary mapping quantum numbers to block sizes.
- `qntofockstates::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to Fock states.
- `qntoinds::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to indices.
- `conserved_quantity::QNfunc`: A function that computes the conserved quantity from a fock number.
"""
struct FockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry
    indtofockdict::IF
    focktoinddict::FI
    qntoblocksizes::Dictionary{QN,Int}
    qntofockstates::Dictionary{QN,Vector{FockNumber}}
    qntoinds::Dictionary{QN,Vector{Int}}
    conserved_quantity::QNfunc
end

Base.:(==)(sym1::FockSymmetry, sym2::FockSymmetry) = sym1.indtofockdict == sym2.indtofockdict && sym1.focktoinddict == sym2.focktoinddict && sym1.qntoblocksizes == sym2.qntoblocksizes && sym1.qntofockstates == sym2.qntofockstates && sym1.qntoinds == sym2.qntoinds #&& sym1.conserved_quantity == sym2.conserved_quantity


"""
    focksymmetry(focknumbers, qn)

Constructs a `FockSymmetry` object that represents the symmetry of a many-body system. 

# Arguments
- `focknumbers`: The focknumbers to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function focksymmetry(_focknumbers, qn; allowed_qns=nothing)
    focknumbers = copy(_focknumbers)
    if allowed_qns !== nothing
        allowed_qns_set = Set(allowed_qns)
        filter!(f -> qn(f) in allowed_qns_set, focknumbers)
    end
    sort!(focknumbers, by=f -> f.f)
    oldinds = eachindex(focknumbers)
    qntooldinds = group(ind -> qn(focknumbers[ind]), oldinds)
    sortkeys!(qntooldinds)
    oldindfromnew = vcat(qntooldinds...)
    blocksizes = map(length, qntooldinds)
    newindfromold = map(first, sort!(collect(enumerate(oldindfromnew)), by=last))
    indtofockdict = map(i -> focknumbers[i], oldindfromnew)
    indtofock(ind) = indtofockdict[ind]
    focktoinddict = Dictionary(focknumbers, newindfromold)
    qntoinds = map(oldinds -> map(oldind -> newindfromold[oldind], oldinds), qntooldinds)
    qntofockstates = map(oldinds -> focknumbers[oldinds], qntooldinds)
    FockSymmetry(indtofockdict, focktoinddict, blocksizes, qntofockstates, qntoinds, qn)
end
focksymmetry(::AbstractVector, ::NoSymmetry) = NoSymmetry()
instantiate(::NoSymmetry, labels) = NoSymmetry()
indtofock(ind, sym::FockSymmetry) = FockNumber(sym.indtofockdict[ind])
focktoind(f, sym::FockSymmetry) = sym.focktoinddict[f]
focknumbers(sym::FockSymmetry) = sym.indtofockdict

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

struct FermionConservation <: AbstractSymmetry end
struct FermionSubsetConservation{M} <: AbstractSymmetry
    mask::M
end
struct UninstantiatedFermionSubsetConservation{L} <: AbstractSymmetry
    labels::L
end
FermionSubsetConservation(::Nothing) = NoSymmetry()
FermionConservation(labels, jw::JordanWignerOrdering) = FermionSubsetConservation(focknbr_from_site_labels(labels, jw))
FermionConservation(labels) = UninstantiatedFermionSubsetConservation(labels)
instantiate(qn::UninstantiatedFermionSubsetConservation, jw::JordanWignerOrdering) = FermionConservation(qn.labels, jw)
instantiate(qn::FermionSubsetConservation, ::JordanWignerOrdering) = qn
instantiate(qn::FermionConservation, ::JordanWignerOrdering) = qn

(qn::FermionSubsetConservation)(fs) = fermionnumber(fs, qn.mask)
(qn::FermionConservation)(fs) = fermionnumber(fs)

@testitem "ConservedFermions" begin
    labels = 1:4
    conservedlabels = 1:4
    qn = FermionConservation(conservedlabels)
    c1 = hilbert_space(labels, qn)
    c2 = hilbert_space(labels, FermionConservation())
    @test c1 == c2

    conservedlabels = 2:2
    qn = FermionConservation(conservedlabels)
    c1 = hilbert_space(labels, qn)
    @test all(c1.symmetry.qntoblocksizes .== 2^(length(labels) - length(conservedlabels)))
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

struct ParityConservation <: AbstractSymmetry end
(qn::ParityConservation)(fs) = parity(fs)
instantiate(qn::ParityConservation, labels) = qn

@testitem "ProductSymmetry" begin
    labels = 1:4
    qn = FermionConservation() * ParityConservation()
    c = hilbert_space(labels, qn)
    @test keys(c.symmetry.qntoinds).values == [(n, (-1)^n) for n in 0:4]
    qn = prod(FermionConservation([l], c.jw) for l in labels)
    @test all(hilbert_space(labels, qn).symmetry.qntoblocksizes .== 1)
end

struct IndexConservation{L} <: AbstractSymmetry
    labels::L
end
instantiate(qn::IndexConservation, jw::JordanWignerOrdering) = IndexConservation(qn.labels, jw)
IndexConservation(index, jw::JordanWignerOrdering) = FermionConservation(filter(label -> index in label || index == label, jw.labels), jw)
@testitem "IndexConservation" begin
    labels = 1:4
    qn = IndexConservation(1)
    qn2 = FermionConservation(1:1)
    c = hilbert_space(labels, qn)
    c2 = hilbert_space(labels, qn2)
    @test c == c2

    spatial_labels = 1:1
    spin_labels = (:↑, :↓)
    all_labels = Base.product(spatial_labels, spin_labels)
    qn = IndexConservation(:↑) * IndexConservation(:↓)
    c = hilbert_space(all_labels, qn)
    @test all(c.symmetry.qntoblocksizes .== 1)
end

instantiate(f::F, labels) where {F} = f


promote_symmetry(s1::FockSymmetry{<:Any,<:Any,<:Any,F}, s2::FockSymmetry{<:Any,<:Any,<:Any,F}) where {F} = s1.conserved_quantity
promote_symmetry(s1::FockSymmetry{<:Any,<:Any,<:Any,F1}, s2::FockSymmetry{<:Any,<:Any,<:Any,F2}) where {F1,F2} = s1 == s2 ? s1.conserved_quantity : NoSymmetry()
promote_symmetry(::NoSymmetry, ::S) where {S} = NoSymmetry()
promote_symmetry(::S, ::NoSymmetry) where {S} = NoSymmetry()
promote_symmetry(::NoSymmetry, ::NoSymmetry) = NoSymmetry()
