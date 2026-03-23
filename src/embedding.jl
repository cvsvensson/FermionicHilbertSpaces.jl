
"""
    embedding_unitary(partition, basisstates, jw)

    Compute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical subspace. 
    # Arguments
    - `partition`: A partition of the labels in `jw` into disjoint sets.
    - `basisstates`: The basis states in the basis
    - `jw`: The Jordan-Wigner ordering.
"""
# function embedding_unitary(partition, basisstates, jw::AbstractDict)
function embedding_unitary(partition, basisstates)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below


    # assumes partition is a list of list of indices
    isorderedpartition(partition, sum(length, partition)) || throw(ArgumentError("Must be an ordered partition to calculate embedding unitary"))
    phases = ones(Int, length(basisstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_indices(Xs)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for i in Xr
                for (n, f) in zip(eachindex(phases), basisstates)
                    if _bit(f, i)
                        phases[n] *= jwstring_anti(i, mask & f)
                    end
                end
            end
        end
    end
    return Diagonal(phases)
end

function bipartite_embedding_unitary(X, Xbar, basisstates)
    #(122a)
    ispartition((X, Xbar), length(X) + length(Xbar)) || throw(ArgumentError("Must be a partition to calculate embedding unitary"))
    phases = ones(Int, length(basisstates))
    mask = focknbr_from_site_indices(X)
    for i in Xbar
        for (n, f) in zip(eachindex(phases), basisstates)
            if _bit(f, i)
                phases[n] *= jwstring_anti(i, mask & f)
            end
        end
    end
    return Diagonal(phases)
end

@testitem "Embedding unitary" begin
    # Appendix C.4
    import FermionicHilbertSpaces: embedding_unitary, bipartite_embedding_unitary, bits
    using LinearAlgebra
    fockstates = sort(map(FockNumber, 0:3), by=Base.Fix2(bits, 2))

    @test embedding_unitary([[1], [2]], fockstates) == I
    @test embedding_unitary([[2], [1]], fockstates) == Diagonal([1, 1, 1, -1])

    # N = 3
    fockstates = sort(map(FockNumber, 0:7), by=Base.Fix2(bits, 3))
    U(p) = embedding_unitary(p, fockstates)
    @test U([[1], [2], [3]]) == U([[1, 2], [3]]) == U([[1], [2, 3]]) == I

    @test U([[2], [1], [3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[2], [3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
    @test U([[3], [1], [2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])
    @test U([[3], [2], [1]]) == Diagonal([1, 1, 1, -1, 1, -1, -1, -1])
    @test U([[1], [3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])

    @test U([[2], [1, 3]]) == Diagonal([1, 1, 1, 1, 1, 1, -1, -1])
    @test U([[3], [1, 2]]) == Diagonal([1, 1, 1, -1, 1, -1, 1, 1])

    @test U([[1, 3], [2]]) == Diagonal([1, 1, 1, -1, 1, 1, 1, -1])
    @test U([[2, 3], [1]]) == Diagonal([1, 1, 1, 1, 1, -1, -1, 1])
end

@testitem "Embedding unitary action" begin
    # Appendix C.4
    import FermionicHilbertSpaces: embedding_unitary, bipartite_embedding_unitary, fermions
    using LinearAlgebra
    @fermions f
    HA = hilbert_space(f, (1, 3))
    HB = hilbert_space(f, (2, 4))
    cA = fermions(HA)
    cB = fermions(HB)
    H = hilbert_space(f, 1:4)
    c = fermions(H)
    Hs = (HA, HB)
    jw = JordanWignerOrdering(1:4).ordering
    @test size(embedding_unitary(Hs, H)) == (dim(H), dim(H))
    @test embed(cA[1], HA, H) ≈ generalized_kron((cA[1], I), Hs, H) ≈ generalized_kron((I, cA[1]), (HB, HA), H)
    Ux = embedding_unitary(Hs, H)
    Ux2 = bipartite_embedding_unitary(HA, HB, H)
    @test Ux ≈ Ux2
    @test embed(cA[1], HA, H) ≈ Ux * embed(cA[1], HA, H; phase_factors=false) * Ux'
end

function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    if isnothing(complement)
        return m
    end
    return generalized_kron((m, I), (Hsub, complement), H; kwargs...)
end
const PairWithHilbertSpaces = Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}

"""
    embed(m, Hsub => H; complement=complementary_subsystem(H, Hsub), kwargs...)

Compute the embedding of a matrix `m` in the basis `Hsub` into the basis `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces. 
"""
embed(m, Hs::PairWithHilbertSpaces; kwargs...) = embed(m, first(Hs), last(Hs); kwargs...)

"""
    embed(Hsub => H; kwargs...)

Compute the embedding map from `Hsub` into `H`. Fermionic phase factors are included if the two spaces are fermionic Hilbert spaces. 
"""
embed(Hs::PairWithHilbertSpaces; kwargs...) = embed_map(first(Hs), last(Hs); kwargs...)
embed_map(Hsub, H; complement=complementary_subsystem(H, Hsub)) = partial_trace_map(H, Hsub; complement)'


@testitem "Partial trace, embed" begin
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsub = hilbert_space(f, (2, 4))
    Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
    pt = partial_trace(H => Hsub)
    emb = embed(Hsub => H)
    @test pt' == emb
end