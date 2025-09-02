sector(qn, H::SymmetricFockHilbertSpace) = hilbert_space(keys(H), H.symmetry.qntofockstates[qn])
quantumnumbers(H::SymmetricFockHilbertSpace) = collect(keys(H.symmetry.qntofockstates))
sectors(H::SymmetricFockHilbertSpace) = map(Base.Fix2(sector, H), quantumnumbers(H))

indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = map(Base.Fix2(state_index, H), basisstates(Hsub))
function indices(qn::QN, H::SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {QN,L,IF,FI,I,QNfunc}
    map(Base.Fix2(state_index, H), H.symmetry.qntofockstates[qn])
end

@testitem "Sector" begin
    import FermionicHilbertSpaces: sector, sectors, indices, quantumnumbers
    N = 4
    H = hilbert_space(1:N, NumberConservation())
    @test quantumnumbers(H) == collect(0:N)
    Hns = sectors(H)
    for n in quantumnumbers(H)
        Hn = hilbert_space(1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(Hns[n + 1]) # Hn ≠ Hns[N + 1] since Hn is a SymmetricFockHilbertSpace
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hn, H)]
        @test basisstates(Hn) == basisstates(H)[indices(n, H)]
    end
end
