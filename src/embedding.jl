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
    Hsub = hilbert_space(f, [2, 4])
    Hcomp = FermionicHilbertSpaces.complementary_subsystem(H, Hsub)
    pt = partial_trace(H => Hsub)
    emb = embed(Hsub => H)
    @test pt' == emb
end