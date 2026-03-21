struct BlockHilbertSpace{B,P,Q} <: AbstractHilbertSpace{B}
    parent::P
    ordered_basis_states::Vector{B}
    state_to_index::Dictionary{B,Int}
    qn_to_states::Dictionary{Q,Vector{B}}
end
BlockHilbertSpace(space::P, ordered_basis_states::AbstractVector{B}, state_to_index::Dictionary{B,Int}, qn_to_states::Dictionary{Q,Vector{B}}) where {B,P,Q} = BlockHilbertSpace{B,P,Q}(space, ordered_basis_states, state_to_index, qn_to_states)


function block_space(space, states, qn)
    _qntostates = group(state -> sector(state, qn), states)
    inds = keys(_qntostates)
    filt_inds = filter(!ismissing, inds)
    qn_to_states = map(collect, getindices(_qntostates, filt_inds))
    sortkeys!(qn_to_states)
    ordered_states = reduce(vcat, qn_to_states)
    state_indexdict = Dictionary(ordered_states, 1:length(ordered_states))
    BlockHilbertSpace(space, ordered_states, state_indexdict, qn_to_states)
end

dim(H::BlockHilbertSpace) = length(H.ordered_basis_states)
atomic_factors(H::BlockHilbertSpace) = atomic_factors(H.parent)
factors(H::BlockHilbertSpace) = factors(H.parent)
isconstrained(H::BlockHilbertSpace) = true
basisstates(H::BlockHilbertSpace) = H.ordered_basis_states
Base.parent(H::BlockHilbertSpace) = H.parent
_find_position(Hsub::AbstractHilbertSpace, H::BlockHilbertSpace) = _find_position(Hsub, H.parent)
# cluster_target_subspace(target::BlockHilbertSpace, args...) = cluster_target_subspace(parent(target), args...)
# cluster_target_sub_idx(target::BlockHilbertSpace, catoms, a2t, ti) = cluster_target_sub_idx(parent(target), catoms, a2t, ti)
combine_states(substates, H::BlockHilbertSpace) = combine_states(substates, parent(H))
partial_trace_phase_factor(s1, s2, H::BlockHilbertSpace) = partial_trace_phase_factor(s1, s2, parent(H))
state_splitter(H::BlockHilbertSpace, Hs) = state_splitter(parent(H), Hs)
mode_ordering(H::BlockHilbertSpace) = mode_ordering(parent(H))
operators(H::BlockHilbertSpace) = operators(parent(H))

function basisstate(ind::Int, H::BlockHilbertSpace)
    (ind < 1 || ind > dim(H)) && throw(ArgumentError("Invalid state index $ind"))
    H.ordered_basis_states[ind]
end

state_index(state, H::BlockHilbertSpace) = get(H.state_to_index, state, missing)

quantumnumbers(H::BlockHilbertSpace) = keys(H.qn_to_states)
quantumnumbers(::AbstractHilbertSpace) = (nothing,)

sector(qn, H::BlockHilbertSpace) = constrain_space(parent(H), H.qn_to_states[qn])
sector(::Nothing, H::AbstractHilbertSpace) = H

sectors(H::BlockHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))
sectors(H::AbstractHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))

indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = map(state -> state_index(state, H), basisstates(Hsub))
function indices(qn, H::BlockHilbertSpace)
    states = H.qn_to_states[qn]
    map(state -> state_index(state, H), states)
end
indices(qn, H::AbstractHilbertSpace) = indices(sector(qn, H), H)
indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)



@testitem "BlockHilbertSpace sectors" begin
    @fermions f

    H = hilbert_space(f, 1:4, NumberConservation())
    @test H isa BlockHilbertSpace
    @test quantumnumbers(H) == collect(0:4)

    for n in 0:4
        Hn = sector(n, H)
        @test dim(Hn) == binomial(4, n)
        @test collect(indices(n, H)) == [FermionicHilbertSpaces.state_index(state, H) for state in basisstates(Hn)]
    end

    @test length(sectors(H)) == length(quantumnumbers(H))

    H0 = hilbert_space(f, 1:4)
    @test quantumnumbers(H0) == (nothing,)
    @test sector(nothing, H0) == H0
    @test length(sectors(H0)) == length(quantumnumbers(H0))

    Hprod = hilbert_space(f, 1:4, NumberConservation() * ParityConservation())
    @test Hprod isa BlockHilbertSpace
    qns = quantumnumbers(Hprod)
    @test all(qn -> qn isa Tuple, qns)
    @test all(qn -> dim(sector(qn, Hprod)) > 0, qns)
end
