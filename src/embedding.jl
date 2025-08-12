
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
    import FermionicHilbertSpaces: embedding_unitary, canonical_embedding, bipartite_embedding_unitary, bits
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
    import FermionicHilbertSpaces: embedding_unitary, canonical_embedding, bipartite_embedding_unitary
    using LinearAlgebra
    HA = hilbert_space((1, 3))
    HB = hilbert_space((2, 4))
    cA = fermions(HA)
    cB = fermions(HB)
    H = hilbert_space(1:4)
    c = fermions(H)
    Hs = (HA, HB)
    @test embedding_unitary(Hs, H) == embedding_unitary([[1, 3], [2, 4]], H)
    @test embedding(cA[1], HA, H) ≈ fermionic_kron((cA[1], I), Hs, H) ≈ fermionic_kron((I, cA[1]), (HB, HA), H)
    Ux = embedding_unitary(Hs, H)
    Ux2 = bipartite_embedding_unitary(HA, HB, H)
    @test Ux ≈ Ux2
    @test embedding(cA[1], HA, H) ≈ Ux * canonical_embedding(cA[1], HA, H) * Ux'
end



"""
    embedding(m, H, Hnew)

Compute the fermionic embedding of a matrix `m` in the basis `Hsub` into the basis `H`.
"""
function embedding(m, Hsub::AbstractFockHilbertSpace, H, phase_factors::Bool=true)
    # See eq. 20 in J. Phys. A: Math. Theor. 54 (2021) 393001
    isorderedsubsystem(Hsub, H) || throw(ArgumentError("Can't embed $Hsub into $H"))
    Hbar = complementary_subsystem(H, Hsub)
    return fermionic_kron((m, I), (Hsub, Hbar), H, phase_factors)
end
const PairWithHilbertSpaces = Pair{<:AbstractFockHilbertSpace,<:AbstractFockHilbertSpace}
embedding(Hs::PairWithHilbertSpaces, phase_factors::Bool=true) = m -> embedding(m, first(Hs), last(Hs), phase_factors)
embedding(m, Hs::PairWithHilbertSpaces, phase_factors::Bool=true) = embedding(m, first(Hs), last(Hs), phase_factors)

"""
    extension(m, H, Hbar[, phase_factors])
Extend an operator or state `m` from Hilbert space `H` into a disjoint space `Hbar`.
"""
function extension(m, H::AbstractFockHilbertSpace, Hbar, phase_factors::Bool=true)
    isdisjoint(keys(H), keys(Hbar)) || throw(ArgumentError("The bases of the two Hilbert spaces must be disjoint"))
    Hs = (H, Hbar)
    Hout = tensor_product(Hs)
    return fermionic_kron((m, I), Hs, Hout, phase_factors)
end
extension(Hs::PairWithHilbertSpaces, phase_factors::Bool=true) = m -> extension(m, first(Hs), last(Hs), phase_factors)
extension(m, Hs::PairWithHilbertSpaces, phase_factors::Bool=true) = extension(m, first(Hs), last(Hs), phase_factors)


## kron, i.e. tensor_product without phase factors
Base.kron(ms, bs, b::AbstractHilbertSpace; kwargs...) = fermionic_kron(ms, bs, b, false; kwargs...)

canonical_embedding(m, b, bnew) = embedding(m, b, bnew, false)
