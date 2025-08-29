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

Base.getindex(m::AbstractVecOrMat, p::PairWithHilbertSpaces) = paddable(p) ? pad(m, p) : project(m, p)

function paddable(p)
    Hfrom, Hto = first(p), last(p)
    return dim(Hfrom) < dim(Hto)
end

function pad(m, p::PairWithHilbertSpaces)
    Hsub, H = first(p), last(p)
    inds = [state_index(s, H) for s in basisstates(Hsub)]
    return pad(m, inds, H)
end

function pad(m::AbstractVector{T}, inds, H) where T
    mout = zeros(T, dim(H))
    mout[inds] = m
    return mout
end

function pad(m::AbstractMatrix{T}, inds, H) where T
    mout = zeros(T, dim(H), dim(H))
    mout[inds, inds] = m
    return mout
end

function project(m::AbstractVecOrMat, p::PairWithHilbertSpaces)
    H, Hsub = first(p), last(p)
    inds = [state_index(s, H) for s in basisstates(Hsub)]
    return ndims(m) == 1 ? m[inds] : m[inds, inds]
end

@testitem "pad and project" begin
    import FermionicHilbertSpaces: sector, paddable
    H = hilbert_space(1:3, NumberConservation())
    Hsub = sector(1, H)
    Hsubsym = hilbert_space(1:3, NumberConservation(1))
    @test paddable(Hsub => H)
    @test !paddable(H => Hsub)
    v = rand(dim(Hsub))
    @test v[Hsub => Hsub] == v
    m = rand(dim(Hsub), dim(Hsub))
    @test v[Hsub => H][H => Hsub] == v
    @test m[Hsub => H][H => Hsub] == m
    @test v[Hsubsym => H][H => Hsubsym] == v
    v = zeros(dim(H))
    v[2] = rand() # index 2 is part of Hsub
    @test v[H => Hsub][Hsub => H] == v
end
