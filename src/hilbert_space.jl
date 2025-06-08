
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


struct SimpleFockHilbertSpace{L} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fermionic::Bool
    function SimpleFockHilbertSpace(labels; fermionic=true)
        jw = JordanWignerOrdering(labels)
        new{eltype(jw)}(jw, fermionic)
    end
end
Base.keys(H::SimpleFockHilbertSpace) = keys(H.jw)
isfermionic(H::SimpleFockHilbertSpace) = H.fermionic
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
    if H1.fermionic != H2.fermionic
        return false
    end
    return true
end

struct FockHilbertSpace{L,F,I} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    focknumbers::F
    focktoind::I
    fermionic::Bool
    function FockHilbertSpace(labels, focknumbers::F=map(FockNumber, 0:2^length(labels)-1); fermionic=true) where F
        jw = JordanWignerOrdering(labels)
        focktoind = Dict(reverse(pair) for pair in enumerate(focknumbers))
        new{eltype(jw),F,typeof(focktoind)}(jw, focknumbers, focktoind, fermionic)
    end
end
Base.keys(H::FockHilbertSpace) = keys(H.jw)
isfermionic(H::FockHilbertSpace) = H.fermionic
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
    if H1.fermionic != H2.fermionic
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


struct SymmetricFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    fermionic::Bool
    symmetry::S
end
Base.keys(H::SymmetricFockHilbertSpace) = keys(H.jw)
isfermionic(H::SymmetricFockHilbertSpace) = H.fermionic
indtofock(ind, H::SymmetricFockHilbertSpace) = indtofock(ind, H.symmetry)
focktoind(f::FockNumber, H::SymmetricFockHilbertSpace) = focktoind(f, H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace) = focknumbers(H.symmetry)
focknumbers(H::SymmetricFockHilbertSpace{<:Any,NoSymmetry}) = Iterators.map(FockNumber, 0:2^length(H.jw)-1)
function SymmetricFockHilbertSpace(labels, qn::AbstractSymmetry, focknumbers=map(FockNumber, 0:2^length(labels)-1); fermionic=true)
    jw = JordanWignerOrdering(labels)
    labelled_symmetry = instantiate(qn, jw)
    sym_concrete = focksymmetry(focknumbers, labelled_symmetry)
    SymmetricFockHilbertSpace(jw, fermionic, sym_concrete)
end

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
    if H1.fermionic != H2.fermionic
        return false
    end
    return true
end

qubit_hilbert_space(labels) = SimpleFockHilbertSpace(labels; fermionic=false)
qubit_hilbert_space(labels, focknumbers) = FockHilbertSpace(labels, focknumbers; fermionic=false)
qubit_hilbert_space(labels, qn::AbstractSymmetry, focknumbers) = SymmetricFockHilbertSpace(labels, qn, focknumbers; fermionic=false)
qubit_hilbert_space(labels, qn::AbstractSymmetry) = SymmetricFockHilbertSpace(labels, qn; fermionic=false)

hilbert_space(labels; fermionic=true) = SimpleFockHilbertSpace(labels; fermionic=fermionic)
hilbert_space(labels, focknumbers; fermionic=true) = FockHilbertSpace(labels, focknumbers; fermionic=fermionic)
hilbert_space(labels, ::NoSymmetry; fermionic=true) = SimpleFockHilbertSpace(labels; fermionic=fermionic)
hilbert_space(labels, ::NoSymmetry, focknumbers; fermionic=true) = FockHilbertSpace(labels, focknumbers; fermionic=fermionic)
hilbert_space(labels, qn::AbstractSymmetry, focknumbers; fermionic=true) = SymmetricFockHilbertSpace(labels, qn, focknumbers; fermionic)
hilbert_space(labels, qn::AbstractSymmetry; fermionic=true) = SymmetricFockHilbertSpace(labels, qn; fermionic)

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
