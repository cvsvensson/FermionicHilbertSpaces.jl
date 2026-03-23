# function sector(qn::QN, H::SymmetricFockHilbertSpace{B,L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {B,QN,L,IF,FI,I,QNfunc}
#     return hilbert_space(modes(H), H.symmetry.qntofockstates[qn])
# end
# function sector(qn::QN, H::MajoranaHilbertSpace{B,LA,SymmetricFockHilbertSpace{B,L,FockSymmetry{IF,FI,QN,I,QNfunc}}}) where {B,QN,LA,L,IF,FI,I,QNfunc}
#     return MajoranaHilbertSpace(H.majoranaindices, sector(qn, H.parent))
# end
# sector(n::Nothing, H::MajoranaHilbertSpace) = MajoranaHilbertSpace(H.majoranaindices, sector(n, H.parent))
# sector(::Nothing, H::AbstractHilbertSpace) = hilbert_space(modes(H), basisstates(H))
# quantumnumbers(H::SymmetricFockHilbertSpace) = collect(keys(H.symmetry.qntofockstates))
# quantumnumbers(::AbstractHilbertSpace) = (nothing,)
# quantumnumbers(H::MajoranaHilbertSpace) = quantumnumbers(H.parent)
# sectors(H::AbstractHilbertSpace) = map(Base.Fix2(sector, H), quantumnumbers(H))

# indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = map(Base.Fix2(state_index, H), basisstates(Hsub))
# function indices(qn::QN, H::SymmetricFockHilbertSpace{B,L,FockSymmetry{IF,FI,QN,I,QNfunc}}) where {B,QN,L,IF,FI,I,QNfunc}
#     map(Base.Fix2(state_index, H), H.symmetry.qntofockstates[qn])
# end
# function indices(qn::QN, H::MajoranaHilbertSpace{B,LA,SymmetricFockHilbertSpace{B,L,FockSymmetry{IF,FI,QN,I,QNfunc}}}) where {B,QN,LA,L,IF,FI,I,QNfunc}
#     return indices(qn, H.parent)
# end
# indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)
