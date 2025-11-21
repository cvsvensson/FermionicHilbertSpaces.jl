
"""
    embedding_unitary(partition, basisstates, jw)

    Compute the unitary matrix that maps between the tensor embedding and the fermionic embedding in the physical subspace. 
    # Arguments
    - `partition`: A partition of the labels in `jw` into disjoint sets.
    - `basisstates`: The basis states in the basis
    - `jw`: The Jordan-Wigner ordering.
"""
function embedding_unitary(_partition, basisstates, jw::JordanWignerOrdering)
    #for locally physical algebra, ie only for even operators or states of well-defined parity
    #if ξ is ordered, the phases are +1. 
    # Note that the jordan wigner modes are ordered in reverse from the labels, but this is taken care of by direction of the jwstring below
    partition = map(modes, _partition)
    isorderedpartition(partition, jw) || throw(ArgumentError("The partition must be ordered according to jw"))

    phases = ones(Int, length(basisstates))
    for (s, Xs) in enumerate(partition)
        mask = focknbr_from_site_labels(Xs, jw)
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for li in Xr
                i = getindex(jw, li)
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

function bipartite_embedding_unitary(_X, _Xbar, basisstates, jw::JordanWignerOrdering)
    #(122a)
    X = modes(_X)
    Xbar = modes(_Xbar)
    ispartition((X, Xbar), jw) || throw(ArgumentError("The partition must be ordered according to jw"))
    phases = ones(Int, length(basisstates))
    mask = focknbr_from_site_labels(X, jw)
    for li in Xbar
        i = getindex(jw, li)
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
    jw = JordanWignerOrdering(1:2)
    fockstates = sort(map(FockNumber, 0:3), by=Base.Fix2(bits, 2))

    @test embedding_unitary([[1], [2]], fockstates, jw) == I
    @test embedding_unitary([[2], [1]], fockstates, jw) == Diagonal([1, 1, 1, -1])

    # N = 3
    jw = JordanWignerOrdering(1:3)
    fockstates = sort(map(FockNumber, 0:7), by=Base.Fix2(bits, 3))
    U(p) = embedding_unitary(p, fockstates, jw)
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
    import FermionicHilbertSpaces: embedding_unitary, bipartite_embedding_unitary
    using LinearAlgebra
    HA = hilbert_space((1, 3))
    HB = hilbert_space((2, 4))
    cA = fermions(HA)
    cB = fermions(HB)
    H = hilbert_space(1:4)
    c = fermions(H)
    Hs = (HA, HB)
    @test embedding_unitary(Hs, H) == embedding_unitary([[1, 3], [2, 4]], H)
    @test embed(cA[1], HA, H) ≈ generalized_kron((cA[1], I), Hs, H) ≈ generalized_kron((I, cA[1]), (HB, HA), H)
    Ux = embedding_unitary(Hs, H)
    Ux2 = bipartite_embedding_unitary(HA, HB, H)
    @test Ux ≈ Ux2
    @test embed(cA[1], HA, H) ≈ Ux * embed(cA[1], HA, H; phase_factors=false) * Ux'
end


function embed(m, Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), kwargs...)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedsubsystem(Hsub, H) || throw(ArgumentError("Can't embed $Hsub into $H"))
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
embed_map(Hsub, H; phase_factors=use_phase_factors(H) && use_phase_factors(Hsub), complement=complementary_subsystem(H, Hsub)) = partial_trace_map(H, Hsub; phase_factors=phase_factors, complement=complement)'

function extend(m, H::AbstractHilbertSpace, Hbar::AbstractHilbertSpace, Hout=tensor_product(H, Hbar); kwargs...)
    isdisjoint(keys(H), keys(Hbar)) || throw(ArgumentError("The bases of the two Hilbert spaces must be disjoint"))
    Hs = (H, Hbar)
    return generalized_kron((m, I), Hs, Hout; kwargs...)
end

"""
    extend(m, H => Hbar, Hout = tensor_product((H, Hbar)); kwargs...)
Extend an operator or state `m` from Hilbert space `H` into a disjoint space `Hbar`.
"""
extend(m, Hs::PairWithHilbertSpaces, Hout=tensor_product((first(Hs), last(Hs))); kwargs...) = extend(m, first(Hs), last(Hs), Hout; kwargs...)
"""
    extend(H => Hbar, Hout = tensor_product((H, Hbar)); kwargs...)
Compute the extend map from `H` into a disjoint space `Hbar`.
"""
extend(Hs::PairWithHilbertSpaces, Hout=tensor_product((first(Hs), last(Hs))); kwargs...) = extend_map(first(Hs), last(Hs), Hout; kwargs...)
extend_map(H, Hbar, Hout=tensor_product((H, Hbar)); phase_factors=use_phase_factors(H) && use_phase_factors(Hbar)) = embed_map(H, Hout; phase_factors=phase_factors)

@testitem "Partial trace, embed, extend" begin
    H = hilbert_space(1:4)
    Hsub = subregion((2, 4), H)
    Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
    pt = partial_trace(H => Hsub)
    emb = embed(Hsub => H)
    ext = extend(Hsub => Hcomp, H)
    @test ext == emb
    @test pt' == emb
end