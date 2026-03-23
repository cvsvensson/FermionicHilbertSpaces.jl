abstract type AbstractSymmetry end

"""
    NoSymmetry
A symmetry type indicating no symmetry constraints.
"""
struct NoSymmetry <: AbstractSymmetry end

""" 
    FockSymmetryFunction{F} <: AbstractSymmetry

FockSymmetryFunction represents a symmetry defined by a function that maps Fock states to quantum numbers. The function should return 'missing' for states which should be discarded from the hilbert space.
"""
struct FockSymmetryFunction{F} <: AbstractSymmetry
    qn::F
end
focksymmetry(f) = FockSymmetryFunction(f)
instantiate_and_get_basisstates(jw::JordanWignerOrdering, sym::FockSymmetryFunction) = focksymmetry(basisstates(jw, NoSymmetry()), sym.qn)
instantiate(sym::FockSymmetryFunction, ::JordanWignerOrdering) = sym
(qn::FockSymmetryFunction)(f) = qn.qn(f)

"""
    struct FockSymmetry{IF,FI,QN,QNfunc} <: AbstractSymmetry

FockSymmetry represents a symmetry that is diagonal in fock space, i.e. particle number conservation, parity, spin conservation.

## Fields
- `basisstates::IF`: A vector of Fock numbers, which are integers representing the occupation of each mode.
- `state_indexdict::FI`: A dictionary mapping Fock states to indices.
- `qntofockstates::Dictionary{QN,Vector{Int}}`: A dictionary mapping quantum numbers to Fock states.
- `conserved_quantity::QNfunc`: A function that computes the conserved quantity from a fock number.
"""
struct FockSymmetry{IF,FI,QN,V,QNfunc} <: AbstractSymmetry
    basisstates::IF
    state_indexdict::FI
    qntofockstates::Dictionary{QN,V}
    conserved_quantity::QNfunc
end
Base.:(==)(sym1::FockSymmetry, sym2::FockSymmetry) = sym1.basisstates == sym2.basisstates && sym1.state_indexdict == sym2.state_indexdict && sym1.qntofockstates == sym2.qntofockstates
statetype(sym::FockSymmetry) = eltype(sym.basisstates)

"""
    focksymmetry(basisstates, qn)

Constructs a `FockSymmetry` object that represents the symmetry of a many-body system. 

# Arguments
- `basisstates`: The basisstates to iterate over
- `qn`: A function that takes an integer representing a fock state and returns corresponding quantum number.
"""
function focksymmetry(basisstates, qn)
    _qntofockstates = group(f -> qn(f), basisstates)
    inds = keys(_qntofockstates)
    filt_inds = (filter(!ismissing, inds))
    qntofockstates = getindices(_qntofockstates, filt_inds)
    sortkeys!(qntofockstates)
    ordered_fockstates = reduce(vcat, qntofockstates)
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

function (qn::NumberConservation)(f)
    n = fermionnumber(f)
    ismissing(qn.sectors) && return n
    n in qn.sectors ? n : missing
end

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
function (qn::ProductSymmetry)(fs)
    qns = map(sym -> sym(fs), qn.symmetries)
    any(ismissing, qns) && return missing
    return qns
end
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
function (qn::ParityConservation)(fs)
    p = parity(fs)
    # ismissing(qn.sectors) && return p
    p in qn.sectors ? p : missing
end
instantiate(qn::ParityConservation, labels) = qn
ParityConservation() = ParityConservation([-1, 1])
sectors(qn::ParityConservation) = qn.sectors

# @testitem "ProductSymmetry" begin
#     import FermionicHilbertSpaces: number_conservation
#     labels = 1:4
#     qn = number_conservation() * ParityConservation()
#     H = hilbert_space(labels, qn)
#     @test keys(H.symmetry.qntofockstates).values == [(n, (-1)^n) for n in 0:4]
#     qn = prod(number_conservation(label -> label == l) for l in labels)
#     @test all(length.(hilbert_space(labels, qn).symmetry.qntofockstates) .== 1)
# end


# @testitem "IndexConservation" begin
#     import FermionicHilbertSpaces: number_conservation
#     labels = 1:4
#     qn = number_conservation(==(1))
#     qn2 = number_conservation(label -> label in 1:1)
#     H = hilbert_space(labels, qn)
#     H2 = hilbert_space(labels, qn2)
#     @test H == H2

#     spatial_labels = 1:1
#     spin_labels = (:↑, :↓)
#     all_labels = Base.product(spatial_labels, spin_labels)
#     qn = number_conservation(label -> label[2] == :↑) * number_conservation(label -> label[2] == :↓)
#     H = hilbert_space(all_labels, qn)
#     @test all(length.(H.symmetry.qntofockstates) .== 1)

#     spatial_labels = 1:2
#     spin_labels = (:↑, :↓)
#     all_labels = Base.product(spatial_labels, spin_labels)
#     qn = number_conservation(1, label -> label[2] == :↑)
#     H = hilbert_space(all_labels, qn)
#     @test length(basisstates(H)) == 2^3
# end

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
    filter!(f -> !ismissing(sym(f)), basisstates(jw, first(sym.symmetries)))
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
function (qn::NumberConservations)(f::FockNumber)
    if length(qn.weights) == 1
        n = fermionnumber(f, only(qn.weights))
        return n in only(qn.sectors) ? n : missing
    else
        ns = map(Base.Fix1(fermionnumber, f), qn.weights)
        for (n, sec) in zip(ns, qn.sectors)
            !in(n, sec) && return missing
        end
        return ns
    end
end
(qn::NumberConservations)(f::Integer) = qn(FockNumber(f))

"""
    number_conservation(sectors=missing, weight_function=label -> true)

Constructs a `UninstantiatedNumberConservations` symmetry object that represents conservation of fermion number. 'sectors' can be an integer or a collection of integers specifying the allowed fermion numbers. 'weight_function' is a function that takes a label and returns an integer weight (which can be negative) indicating how that label contributes to the fermion number.
"""
function number_conservation(sectors=missing, weight_function=:alltrue)
    if weight_function == :alltrue
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


instantiate(qn::UninstantiatedNumberConservations, jw::JordanWignerOrdering,) = instantiate(qn, keys(jw))
function instantiate(qn::UninstantiatedNumberConservations, labels)
    N = length(labels)
    weights = map(f -> map(l -> f(l), labels), qn.weight_functions)
    NumberConservations(weights, qn.sectors, N)
end
basisstates(jw::JordanWignerOrdering, qn::NumberConservations, ::Type{F}=FockNumber) where F = sort!(map(F, generate_states(qn.weights, qn.sectors, length(jw))))

@testitem "Symmetry basisstates" begin
    import FermionicHilbertSpaces: instantiate_and_get_basisstates, fermionnumber, number_conservation
    @fermions f
    H = hilbert_space(f, 1:5)
    Hcons = constrain_space(H, ParityConservation())
    @test dim(Hcons) == 2^5
    Hcons = constrain_space(H, ParityConservation(1))
    @test dim(Hcons) == 2^4
    # @test length(collect(basisstates(H.jw, ParityConservation()))) == 2^5
    # @test length(collect(basisstates(H.jw, ParityConservation(1)))) == 2^4
    odd_focks = basisstates(constrain_space(H, ParityConservation(-1)))
    @test all(isodd ∘ fermionnumber, odd_focks)
    @test dim(constrain_space(H, ParityConservation([-1, 1]))) == 2^5

    ## ProductSymmetry
    qn = ParityConservation([1]) * NumberConservation(1:2, H.modes[1:3])
    states = FermionicHilbertSpaces.generate_states(H, qn)
    H2 = hilbert_space(f, 1:5, qn)
    @test sort(basisstates(H2)) == sort(map(state -> FermionicHilbertSpaces.catenate_fock_states(state, H.modes), states))
    @test all(iseven ∘ fermionnumber, basisstates(H2))
end

@testitem "sector" begin
    import FermionicHilbertSpaces: sector, parity
    # Create a Hilbert space with parity symmetry
    @fermions f
    labels = 1:3
    qn = ParityConservation()
    H = hilbert_space(f, labels, qn)
    n = length(basisstates(H))
    m = reshape(1:(n^2), n, n)  # simple test matrix
    # Get the sector for parity = 1
    even_inds = indices(1, H)
    even_sector = m[even_inds, even_inds]
    # The size of the even sector block should match the number of even-parity states
    even_states = [f for f in basisstates(H) if parity(f) == 1]
    @test size(even_sector, 1) == length(even_states)
    # The values should match the corresponding block in m
    # Get the indices of even states in the full basisstates list
    even_inds = findall(f -> parity(f) == 1, basisstates(H))
    @test even_sector == m[even_inds, even_inds]

    # Test with NumberConservation
    import FermionicHilbertSpaces: fermionnumber
    qn_f = NumberConservation([1, 2])
    Hf = hilbert_space(f, labels, qn_f)
    n_f = length(basisstates(Hf))
    m_f = reshape(1:(n_f^2), n_f, n_f)
    # Test sector for fermion number = 1
    sector1_inds = indices(1, Hf)
    sector1 = m_f[sector1_inds, sector1_inds]
    states1 = [f for f in basisstates(Hf) if fermionnumber(f) == 1]
    inds1 = findall(f -> fermionnumber(f) == 1, basisstates(H))
    @test size(sector1, 1) == length(states1)
    @test sector1 == m_f[inds1, inds1]
    # Test sector for fermion number = 2
    sector2_inds = indices(2, Hf)
    sector2 = m_f[sector2_inds, sector2_inds]
    states2 = [f for f in basisstates(Hf) if fermionnumber(f) == 2]
    inds2 = findall(f -> fermionnumber(f) == 2, basisstates(Hf))
    @test size(sector2, 1) == length(states2)
    @test sector2 == m_f[inds2, inds2]
    # Test that an invalid sector throws an error
    @test_throws "Dictionaries.IndexError" sector(99, Hf)
    @test_throws "Dictionaries.IndexError" indices(99, Hf)
end

@testitem "No double occupation projection" begin
    @fermions f
    N = 4
    Nup = 2
    Ndn = 1
    spins = (:↑, :↓)
    spatial_labels = 1:N
    labels = collect(Base.product(spatial_labels, spins))
    spin_up_sites = filter(label -> label[2] == :↑, labels)
    spin_up_conservation = NumberConservation(Nup, hilbert_space(f, spin_up_sites))
    spin_down_sites = filter(label -> label[2] == :↓, labels)
    spin_down_conservation = NumberConservation(Ndn, hilbert_space(f, spin_down_sites))
    no_double_occupation = prod(NumberConservation(0:1, hilbert_space(f, [(k, σ) for σ in spins])) for k in spatial_labels)
    # no_double_occupation = prod(number_conservation(0:1, label -> k in label) for k in spatial_labels)

    qn = spin_up_conservation * spin_down_conservation * no_double_occupation
    H = hilbert_space(f, labels, qn)
    H = hilbert_space(f, labels)
    func = FermionicHilbertSpaces.sector_function(qn, H)
    hopping_symham = sum(zip(spatial_labels, spatial_labels[2:end])) do (i, j)
        sum(spins) do σ
            f[(i, σ)]' * f[(j, σ)] + hc
        end
    end
    @test_throws MethodError matrix_representation(hopping_symham, H)
    @test size(matrix_representation(hopping_symham, H; projection=true), 1) == dim(H)
end
