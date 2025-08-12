function Base.show(io::IO, H::Htype) where Htype<:AbstractFockHilbertSpace
    n, m = size(H)
    println(io, "$(n)⨯$m $(Htype.name.name):")
    print(io, "modes: ")
    print(IOContext(io, :compact => true), modes(H))
end
Base.show(io::IO, ::MIME"text/plain", H::AbstractHilbertSpace) = show(io, H)

Base.size(H::AbstractHilbertSpace) = (length(basisstates(H)), length(basisstates(H)))
Base.size(H::AbstractHilbertSpace, i) = i == 1 || i == 2 ? length(basisstates(H)) : throw(BoundsError(H, (i,)))
isorderedpartition(Hs, H::AbstractFockHilbertSpace) = isorderedpartition(map(modes, Hs), H.jw)
isorderedsubsystem(Hsub::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = isorderedsubsystem(Hsub.jw, H.jw)
isorderedsubsystem(Hsub::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = isorderedsubsystem(Hsub.jw, jw)
issubsystem(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = issubsystem(subsystem.jw, jw)
issubsystem(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = issubsystem(subsystem.jw, H.jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = consistent_ordering(subsystem.jw, jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = consistent_ordering(subsystem.jw, H.jw)
focknbr_from_site_labels(H::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = focknbr_from_site_labels(keys(H), jw)
ispartition(Hs, H::AbstractFockHilbertSpace) = ispartition(map(modes, Hs), H.jw)

siteindices(H::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = siteindices(H.jw, jw)

mode_ordering(H::AbstractFockHilbertSpace) = H.jw
# mode_ordering(jw::JordanWignerOrdering) = jw
# mode_ordering(v::AbstractVector) = JordanWignerOrdering(v)
modes(H::AbstractFockHilbertSpace) = H.jw.labels
modes(jw::JordanWignerOrdering) = jw.labels
modes(v::AbstractVector) = v
embedding_unitary(partition, H::AbstractFockHilbertSpace) = embedding_unitary(partition, basisstates(H), H.jw)
bipartite_embedding_unitary(X, Xbar, H::AbstractFockHilbertSpace) = bipartite_embedding_unitary(X, Xbar, basisstates(H), H.jw)


"""
    SimpleFockHilbertSpace
A type representing a simple Fock Hilbert space with all fock states included.
"""
struct SimpleFockHilbertSpace{L} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    function SimpleFockHilbertSpace(labels)
        jw = JordanWignerOrdering(labels)
        new{eltype(jw)}(jw)
    end
end
Base.keys(H::SimpleFockHilbertSpace) = keys(H.jw)
"""
    basisstates(H)
Return an iterator over all basis states for the given Hilbert space `H`.
"""
basisstates(H::SimpleFockHilbertSpace) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)
basisstate(ind, ::SimpleFockHilbertSpace) = FockNumber(ind - 1)
state_index(focknbr::FockNumber, ::SimpleFockHilbertSpace) = focknbr.f + 1
function Base.:(==)(H1::SimpleFockHilbertSpace, H2::SimpleFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    return true
end

"""
    FockHilbertSpace
A type representing a Fock Hilbert space with a given set of modes and Fock states.
"""
struct FockHilbertSpace{L,F,I} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    basisstates::F
    state_index::I
    function FockHilbertSpace(labels, basisstates::F=map(FockNumber, 0:2^length(labels)-1)) where F
        jw = JordanWignerOrdering(labels)
        state_index = Dict(reverse(pair) for pair in enumerate(basisstates))
        new{eltype(jw),F,typeof(state_index)}(jw, basisstates, state_index)
    end
end
Base.keys(H::FockHilbertSpace) = keys(H.jw)
basisstates(H::FockHilbertSpace) = H.basisstates
basisstate(ind, H::FockHilbertSpace) = basisstates(H)[ind]
state_index(fockstate::AbstractFockState, H::FockHilbertSpace) = H.state_index[fockstate]
function Base.:(==)(H1::FockHilbertSpace, H2::FockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.basisstates != H2.basisstates
        return false
    end
    if H1.state_index != H2.state_index
        return false
    end
    return true
end


"""
    SymmetricFockHilbertSpace
A type representing a Fock Hilbert space with fockstates organized by their quantum number.
"""
struct SymmetricFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    symmetry::S
end
function SymmetricFockHilbertSpace(labels, qn::AbstractSymmetry)
    SymmetricFockHilbertSpace(JordanWignerOrdering(labels), qn)
end
function SymmetricFockHilbertSpace(jw::JordanWignerOrdering, qn::AbstractSymmetry)
    labelled_symmetry, basisstates = instantiate_and_get_basisstates(jw, qn)
    sym_concrete = focksymmetry(basisstates, labelled_symmetry)
    SymmetricFockHilbertSpace{eltype(jw),typeof(sym_concrete)}(jw, sym_concrete)
end

function Base.show(io::IO, H::SymmetricFockHilbertSpace)
    n, m = size(H)
    println(io, "$(n)⨯$m SymmetricFockHilbertSpace:")
    print(io, "modes: ")
    println(IOContext(io, :compact => true), modes(H))
    show(io, H.symmetry)
end
Base.show(io::IO, sym::FockSymmetry) = print(io, sym.conserved_quantity)

Base.keys(H::SymmetricFockHilbertSpace) = keys(H.jw)
basisstate(ind, H::SymmetricFockHilbertSpace) = basisstate(ind, H.symmetry)
state_index(f::AbstractFockState, H::SymmetricFockHilbertSpace) = state_index(f, H.symmetry)
basisstates(H::SymmetricFockHilbertSpace) = basisstates(H.symmetry)
basisstates(H::SymmetricFockHilbertSpace{<:Any,NoSymmetry}) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)

function Base.:(==)(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.symmetry != H2.symmetry
        return false
    end
    return true
end

"""
    hilbert_space(labels[, symmetry, basisstates])
Construct a Hilbert space from a set of labels, with optional symmetry and Fock number specification.
"""
hilbert_space(labels) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, basisstates) = FockHilbertSpace(labels, basisstates)
hilbert_space(labels, ::NoSymmetry) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, ::NoSymmetry, basisstates) = FockHilbertSpace(labels, basisstates)
hilbert_space(labels, qn::AbstractSymmetry) = SymmetricFockHilbertSpace(labels, qn)

#= Tests for isorderedsubsystem, issubsystem, and consistent_ordering for Hilbert spaces =#
@testitem "Hilbert space subsystem and ordering" begin
    import FermionicHilbertSpaces: isorderedsubsystem, issubsystem, consistent_ordering
    # Simple Hilbert spaces
    H = hilbert_space([1, 2, 3])
    Hsub1 = hilbert_space([1, 2])
    Hsub2 = hilbert_space([2, 3])
    Hsub3 = hilbert_space([2, 1])
    Hsub4 = hilbert_space([3, 2])
    Hsub5 = hilbert_space([4])

    # issubsystem
    @test issubsystem(Hsub1, H)
    @test issubsystem(Hsub2, H)
    @test !issubsystem(Hsub5, H)
    @test issubsystem(Hsub1, H.jw)
    @test !issubsystem(Hsub5, H.jw)

    # consistent_ordering
    @test consistent_ordering(Hsub1, H)
    @test consistent_ordering(Hsub2, H.jw)
    @test !consistent_ordering(Hsub3, H)
    @test !consistent_ordering(Hsub4, H.jw)

    # isorderedsubsystem
    @test isorderedsubsystem(Hsub1, H)
    @test isorderedsubsystem(Hsub2, H.jw)
    @test !isorderedsubsystem(Hsub3, H)
    @test !isorderedsubsystem(Hsub4, H)
    @test !isorderedsubsystem(Hsub5, H)

    # ispartition and isorderedpartition
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    # Partition of Hilbert spaces (as required by isorderedpartition(Hs, H))
    H = hilbert_space([1, 2, 3])
    Hs1 = [hilbert_space([1]), hilbert_space([2, 3])]
    Hs2 = [hilbert_space([2]), hilbert_space([1, 3])]
    Hs3 = [hilbert_space([1, 2, 3])]
    Hs4 = [hilbert_space([1]), hilbert_space([2])]
    @test ispartition(Hs1, H)
    @test ispartition(map(keys, Hs2), H.jw)
    @test !ispartition(Hs4, H)
    @test isorderedpartition(Hs1, H)
    @test isorderedpartition(map(keys, Hs3), H.jw)
    @test !isorderedpartition(Hs4, H)
end

"""
    subregion(modes, H::AbstractHilbertSpace)

Return a subregion of the Hilbert space `H` that is spanned by the modes in `modes`. Only substates in `H` are included.
"""
function subregion(modes, H::SimpleFockHilbertSpace)
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    SimpleFockHilbertSpace(modes)
end

function subregion(modes, H::FockHilbertSpace)
    if !isorderedsubsystem(submodes, mode_ordering(H))
        throw(ArgumentError("The modes $(submodes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    # loop through all basisstates in H and collect the fock states that are in the subsystem
    outinds = siteindices(modes, H.jw)
    outbits(f) = Iterators.map(i -> _bit(f, i), outinds)
    subfocks = eltype(H.basisstates)[]
    for f in basisstates(H)
        subbits = outbits(f)
        subfock = focknbr_from_bits(subbits)
        push!(subfocks, subfock)
    end
    sort!(unique!(subfocks), by=f -> f.f)
    FockHilbertSpace(modes, subfocks)
end

complementary_subsystem(H::AbstractFockHilbertSpace, Hsub::AbstractFockHilbertSpace) = SimpleFockHilbertSpace(setdiff(collect(keys(H)), collect(keys(Hsub))))

function subregion(modes, H::SymmetricFockHilbertSpace)
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    # loop through all basisstates in H and collect the fock states that are in the subsystem
    outinds = siteindices(modes, H.jw)
    outbits(f) = Iterators.map(i -> _bit(f, i), outinds)
    subfocks = eltype(H.symmetry.basisstates)[]
    for f in basisstates(H)
        subbits = outbits(f)
        subfock = focknbr_from_bits(subbits)
        push!(subfocks, subfock)
    end
    sort!(unique!(subfocks), by=f -> f.f)
    FockHilbertSpace(modes, subfocks)
end

@testitem "subregion function" begin
    using FermionicHilbertSpaces
    # SimpleFockHilbertSpace
    H = hilbert_space([1, 2, 3])
    Hsub = subregion([1, 2], H)
    @test Hsub isa SimpleFockHilbertSpace
    @test keys(Hsub) == [1, 2]
    # FockHilbertSpace
    focks = [FockNumber(1), FockNumber(3)]
    HF = FockHilbertSpace([1, 2], focks)
    HFsub = subregion([1], HF)
    @test HFsub isa FockHilbertSpace
    @test keys(HFsub) == [1]
    basisstates(HFsub) == [FockNumber(1)]
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    HS = hilbert_space([1, 2], qn)
    HSsub = subregion([1], HS)
    @test HSsub isa FockHilbertSpace
    @test keys(HSsub) == [1]
    basisstates(HSsub) == [FockNumber(0), FockNumber(1)]
    # Error on non-subsystem
    @test_throws ArgumentError subregion([4], H)
end



function sector(m::AbstractMatrix, qn::Int, H::SymmetricFockHilbertSpace)
    ls = length.(H.symmetry.qntofockstates)
    startindex = 1
    for (n, l) in pairs(ls)
        if n == qn
            return m[startindex:startindex+l-1, startindex:startindex+l-1]
        end
        startindex += l
    end
    throw(ArgumentError("Sector $qn not found in the matrix."))
end

@testitem "Hilbert space printing" begin
    # Check that printing of Hilbert spaces doesn't error
    using FermionicHilbertSpaces
    io = IOBuffer()
    # SimpleFockHilbertSpace
    H_simple = hilbert_space(1:3)
    @test begin
        show(io, H_simple)
        true
    end
    # FockHilbertSpace
    focks = [FockNumber(0), FockNumber(1), FockNumber(2)]
    H_fock = FockHilbertSpace(1:2, focks)
    @test begin
        show(io, H_fock)
        true
    end
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    H_sym = hilbert_space(1:2, qn)
    @test begin
        show(io, H_sym)
        true
    end
end