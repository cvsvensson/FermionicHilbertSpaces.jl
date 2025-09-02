function sector(qn::QN, H::SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {QN,L,IF,FI,I,QNfunc}
    return hilbert_space(modes(H), H.symmetry.qntofockstates[qn])
end
function sector(qn::QN, H::MajoranaHilbertSpace{LA, SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}}) where {QN,LA,L,IF,FI,I,QNfunc}
    return sector(qn, H.parent)
end
sector(::Nothing, H::AbstractHilbertSpace) = hilbert_space(modes(H), basisstates(H))
quantumnumbers(H::SymmetricFockHilbertSpace) = collect(keys(H.symmetry.qntofockstates))
quantumnumbers(::AbstractHilbertSpace) = (nothing,)
quantumnumbers(H::MajoranaHilbertSpace) = quantumnumbers(H.parent)
sectors(H::AbstractHilbertSpace) = map(Base.Fix2(sector, H), quantumnumbers(H))

indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = map(Base.Fix2(state_index, H), basisstates(Hsub))
function indices(qn::QN, H::SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {QN,L,IF,FI,I,QNfunc}
    map(Base.Fix2(state_index, H), H.symmetry.qntofockstates[qn])
end
function indices(qn::QN, H::MajoranaHilbertSpace{LA,SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}}) where {QN,LA,L,IF,FI,I,QNfunc}
    return indices(qn, H.parent)
end
indices(::Nothing, H::AbstractHilbertSpace) = collect(1:dim(H))

@testitem "Sector" begin
    import FermionicHilbertSpaces: sector, sectors, indices, quantumnumbers, majorana_hilbert_space
    N = 4
    H = hilbert_space(1:N, NumberConservation())
    @test quantumnumbers(H) == collect(0:N)
    Hns = sectors(H)
    for (ind, n) in enumerate(quantumnumbers(H))
        Hn = hilbert_space(1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(Hns[ind]) # Hn ≠ Hns[ind] since Hn is a SymmetricFockHilbertSpace
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hn, H)]
        @test basisstates(Hn) == basisstates(H)[indices(n, H)]
    end
    # no qns
    Hnoqn = hilbert_space(1:N)
    HMnoqn = majorana_hilbert_space(1:N)
    @test indices(only(quantumnumbers(Hnoqn)), Hnoqn) == collect(1:dim(Hnoqn))
    @test indices(only(quantumnumbers(HMnoqn)), HMnoqn) == collect(1:dim(HMnoqn))
    @test length(sectors(Hnoqn)) == length(sectors(HMnoqn)) == 1
    # Majorana hilbert spaces
    Hm = majorana_hilbert_space(1:N, NumberConservation())
    @test quantumnumbers(Hm) == collect(0:N ÷ 2)
    qn = 1
    Hmqn = majorana_hilbert_space(1:N, NumberConservation(qn))
    @test basisstates(Hmqn) == basisstates(Hm)[indices(qn, Hm)]
    @test length(sectors(Hm)) == N ÷ 2 + 1
end
