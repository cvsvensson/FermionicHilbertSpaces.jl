function Base.show(io::IO, H::Htype) where Htype<:AbstractHilbertSpace
    n, m = size(H)
    println(io, "$(n)⨯$m $(Htype.name.name):")
    print(io, "modes: $(mode_ordering(H))")
end
Base.show(io::IO, ::MIME"text/plain", H::AbstractHilbertSpace) = show(io, H)

Base.size(H::AbstractFockHilbertSpace) = (length(focknumbers(H)), length(focknumbers(H)))
function isorderedpartition(Hs, H::AbstractHilbertSpace)
    partition = map(keys, Hs)
    isorderedpartition(partition, H.jw)
end
isorderedsubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = isorderedsubsystem(Hsub.jw, H.jw)
isorderedsubsystem(Hsub::AbstractHilbertSpace, jw::JordanWignerOrdering) = isorderedsubsystem(Hsub.jw, jw)
issubsystem(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = issubsystem(subsystem.jw, jw)
issubsystem(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = issubsystem(subsystem.jw, H.jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = consistent_ordering(subsystem.jw, jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = consistent_ordering(subsystem.jw, H.jw)
focknbr_from_site_labels(H::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = focknbr_from_site_labels(keys(H), jw)
ispartition(partition, H::AbstractFockHilbertSpace) = ispartition(partition, H.jw)
isorderedpartition(partition, H::AbstractFockHilbertSpace) = isorderedpartition(partition, H.jw)

siteindices(H::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = siteindices(H.jw, jw)

mode_ordering(H::AbstractFockHilbertSpace) = H.jw.labels
mode_ordering(jw::JordanWignerOrdering) = jw.labels
mode_ordering(v::AbstractVector) = v
embedding_unitary(partition, H::AbstractFockHilbertSpace) = embedding_unitary(partition, focknumbers(H), H.jw)
bipartite_embedding_unitary(X, Xbar, H::AbstractFockHilbertSpace) = bipartite_embedding_unitary(X, Xbar, focknumbers(H), H.jw)


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
    focknumbers(H)
Return an iterator over all Fock states for the given Hilbert space `H`.
"""
focknumbers(H::SimpleFockHilbertSpace) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)
indtofock(ind, ::SimpleFockHilbertSpace) = FockNumber(ind - 1)
focktoind(focknbr::FockNumber, ::SimpleFockHilbertSpace) = focknbr.f + 1
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
    focknumbers::F
    focktoind::I
    function FockHilbertSpace(labels, focknumbers::F=map(FockNumber, 0:2^length(labels)-1)) where F
        jw = JordanWignerOrdering(labels)
        focktoind = Dict(reverse(pair) for pair in enumerate(focknumbers))
        new{eltype(jw),F,typeof(focktoind)}(jw, focknumbers, focktoind)
    end
end
Base.keys(H::FockHilbertSpace) = keys(H.jw)
focknumbers(H::FockHilbertSpace) = H.focknumbers
indtofock(ind, H::FockHilbertSpace) = focknumbers(H)[ind]
focktoind(focknbr::FockNumber, H::FockHilbertSpace) = H.focktoind[focknbr]
function Base.:(==)(H1::FockHilbertSpace, H2::FockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.focknumbers != H2.focknumbers
        return false
    end
    if H1.focktoind != H2.focktoind
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
    labelled_symmetry, focknumbers = instantiate_and_get_focknumbers(jw, qn)
    sym_concrete = focksymmetry(focknumbers, labelled_symmetry)
    SymmetricFockHilbertSpace{eltype(jw),typeof(sym_concrete)}(jw, sym_concrete)
end

function Base.show(io::IO, H::SymmetricFockHilbertSpace)
    n, m = size(H)
    println(io, "$(n)⨯$m SymmetricFockHilbertSpace:")
    println(io, "modes: $(mode_ordering(H))")
    show(io, H.symmetry)
end
Base.show(io::IO, sym::FockSymmetry) = print(io, sym.conserved_quantity)

Base.keys(H::SymmetricFockHilbertSpace) = keys(H.jw)
indtofock(ind, H::SymmetricFockHilbertSpace) = indtofock(ind, H.symmetry)
focktoind(f::FockNumber, H::SymmetricFockHilbertSpace) = focktoind(f, H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace) = focknumbers(H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace{<:Any,NoSymmetry}) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)

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
    hilbert_space(labels[, symmetry, focknumbers])
Construct a Hilbert space from a set of labels, with optional symmetry and Fock number specification.
"""
hilbert_space(labels) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, focknumbers) = FockHilbertSpace(labels, focknumbers)
hilbert_space(labels, ::NoSymmetry) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, ::NoSymmetry, focknumbers) = FockHilbertSpace(labels, focknumbers)
hilbert_space(labels, qn::AbstractSymmetry, focknumbers) = SymmetricFockHilbertSpace(labels, qn, focknumbers)
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
    @test ispartition(Hs1, H.jw)
    @test ispartition(map(keys, Hs2), H.jw)
    @test ispartition(map(keys, Hs2), H)
    @test !ispartition(Hs4, H)
    @test isorderedpartition(Hs1, H.jw)
    @test isorderedpartition(map(keys, Hs3), H)
    @test !isorderedpartition(Hs4, H)
end

"""
    subspace(modes, H::AbstractHilbertSpace)

Return a subspace of the Hilbert space `H` that is spanned by the modes in `modes`. Only substates in `H` are included.
"""
function subspace(modes, H::SimpleFockHilbertSpace)
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    SimpleFockHilbertSpace(modes)
end

function subspace(modes, H::FockHilbertSpace)
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    # loop through all focknumbers in H and collect the fock states that are in the subsystem
    outinds = siteindices(modes, H.jw)
    outbits(f) = map(i -> _bit(f, i), outinds)
    subfocks = FockNumber[]
    for f in focknumbers(H)
        subbits = outbits(f)
        subfock = focknbr_from_bits(subbits)
        push!(subfocks, subfock)
    end
    sort!(unique!(subfocks), by=f -> f.f)
    FockHilbertSpace(modes, subfocks)
end

function subspace(modes, H::SymmetricFockHilbertSpace)
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    # loop through all focknumbers in H and collect the fock states that are in the subsystem
    outinds = siteindices(modes, H.jw)
    outbits(f) = map(i -> _bit(f, i), outinds)
    subfocks = FockNumber[]
    for f in focknumbers(H)
        subbits = outbits(f)
        subfock = focknbr_from_bits(subbits)
        push!(subfocks, subfock)
    end
    sort!(unique!(subfocks), by=f -> f.f)
    FockHilbertSpace(modes, subfocks)
end

@testitem "subspace function" begin
    using FermionicHilbertSpaces
    # SimpleFockHilbertSpace
    H = hilbert_space([1, 2, 3])
    Hsub = subspace([1, 2], H)
    @test Hsub isa SimpleFockHilbertSpace
    @test keys(Hsub) == [1, 2]
    # FockHilbertSpace
    focks = [FockNumber(1), FockNumber(3)]
    HF = FockHilbertSpace([1, 2], focks)
    HFsub = subspace([1], HF)
    @test HFsub isa FockHilbertSpace
    @test keys(HFsub) == [1]
    focknumbers(HFsub) == [FockNumber(1)]
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    HS = hilbert_space([1, 2], qn, [FockNumber(1), FockNumber(3)])
    HSsub = subspace([1], HS)
    @test HSsub isa FockHilbertSpace
    @test keys(HSsub) == [1]
    focknumbers(HSsub) == [FockNumber(1)]
    # Error on non-subsystem
    @test_throws ArgumentError subspace([4], H)
end
