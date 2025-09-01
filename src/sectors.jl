sector(qn, H::SymmetricFockHilbertSpace) = hilbert_space(keys(H), H.symmetry.qntofockstates[qn])

indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = map(Base.Fix2(state_index, H), basisstates(Hsub))
function indices(qn::QN, H::SymmetricFockHilbertSpace{L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {QN,L,IF,FI,I,QNfunc}
    map(Base.Fix2(state_index, H), H.symmetry.qntofockstates[qn])
end

@testitem "Sector" begin
    import FermionicHilbertSpaces: sector, indices
    N = 4
    H = hilbert_space(1:N, NumberConservation())
    for n in 1:N
        Hn = hilbert_space(1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hn, H)]
        @test basisstates(Hn) == basisstates(H)[indices(n, H)]
    end
end
