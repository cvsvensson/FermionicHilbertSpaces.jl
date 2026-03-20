struct BlockHilbertSpace{B,H,Q} <: AbstractHilbertSpace{B}
    parent::H
    blocks::Vector{H}
    qns::Vector{Q}
    qn_to_block_index::Dict{Q,Int}
end
dim(H::BlockHilbertSpace) = sum(dim, H.blocks)
atomic_factors(H::BlockHilbertSpace) = atomic_factors(H.parent)
factors(H::BlockHilbertSpace) = factors(H.parent)
isconstrained(H::BlockHilbertSpace) = true
basisstates(H::BlockHilbertSpace) = Iterators.flatten(Iterators.map(basisstates, H.blocks))
state_index(state, H::BlockHilbertSpace) = begin
    offset = 0
    for block in H.blocks
        idx = state_index(state, block)
        if !ismissing(idx)
            return offset + idx
        end
        offset += dim(block)
    end
    missing
end
sectors(H::BlockHilbertSpace) = H.qns
sector(qn::Q, H::BlockHilbertSpace) = H.blocks[H.qn_to_block_index[qn]]

# function sector(m::AbstractMatrix, qn::Int, H::SymmetricFockHilbertSpace)
#     ls = length.(H.symmetry.qntofockstates)
#     startindex = 1
#     for (n, l) in pairs(ls)
#         if n == qn
#             return m[startindex:startindex+l-1, startindex:startindex+l-1]
#         end
#         startindex += l
#     end
#     throw(ArgumentError("Sector $qn not found in the matrix."))
# end
