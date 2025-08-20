sector(qn, H::SymmetricFockHilbertSpace) = hilbert_space(keys(H), H.symmetry.qntofockstates[qn])

@testitem "Sector" begin
    import FermionicHilbertSpaces: sector
    N = 4
    H = hilbert_space(1:N, NumberConservation())
    for n in 1:N
        Hn = hilbert_space(1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(sector(n, H))
    end
end