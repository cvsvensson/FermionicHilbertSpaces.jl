struct BlockHilbertSpace{B,P,Q} <: AbstractHilbertSpace{B}
    parent::P
    ordered_basis_states::Vector{B}
    state_to_index::Dictionary{B,Int}
    qn_to_states::Dictionary{Q,Vector{B}}
end
BlockHilbertSpace(space::P, ordered_basis_states::AbstractVector{B}, state_to_index::Dictionary{B,Int}, qn_to_states::Dictionary{Q,Vector{B}}) where {B,P,Q} = BlockHilbertSpace{B,P,Q}(space, ordered_basis_states, state_to_index, qn_to_states)
Base.hash(H::BlockHilbertSpace, h::UInt) = hash((H.parent, H.ordered_basis_states, H.state_to_index, H.qn_to_states), h)
Base.:(==)(H1::BlockHilbertSpace, H2::BlockHilbertSpace) = H1 === H2 || (H1.parent == H2.parent && H1.ordered_basis_states == H2.ordered_basis_states && H1.state_to_index == H2.state_to_index && H1.qn_to_states == H2.qn_to_states)

function block_space(space, states, sector_function)
    _qntostates = group(state -> sector_function(state), states)
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
isconstrained(H::BlockHilbertSpace) = true
basisstates(H::BlockHilbertSpace) = H.ordered_basis_states
Base.parent(H::BlockHilbertSpace) = H.parent
_find_position(Hsub::AbstractHilbertSpace, H::BlockHilbertSpace) = _find_position(Hsub, H.parent)
clusters(H::BlockHilbertSpace) = clusters(H.parent)
factors(H::BlockHilbertSpace) = factors(H.parent)
combine_states(substates, H::BlockHilbertSpace) = combine_states(substates, parent(H))
partial_trace_phase_factor(s1, s2, H::BlockHilbertSpace) = partial_trace_phase_factor(s1, s2, parent(H))
state_splitter(H::BlockHilbertSpace, Hs) = state_splitter(parent(H), Hs)
mode_ordering(H::BlockHilbertSpace) = mode_ordering(parent(H))
default_sorter(H::BlockHilbertSpace, constraint) = default_sorter(parent(H), constraint)
default_processor(H::BlockHilbertSpace, constraint) = default_processor(parent(H), constraint)

function Base.show(io::IO, H::BlockHilbertSpace)
    if get(io, :compact, false)
        print(io, "BlockHilbertSpace(")
        show(IOContext(io, :compact => true), H.parent)
        print(io, ", $(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional BlockHilbertSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), H.parent)
        qns = collect(keys(H.qn_to_states))
        if !isempty(qns)
            print(io, "\nSectors: ")
            for (i, qn) in enumerate(qns)
                i > 1 && print(io, ", ")
                print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
            end
        end
    end
end

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

function indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace)
    sector_list = collect(sectors(H))
    indexin = findfirst(isequal(Hsub), sector_list)
    # map(state -> state_index(state, H), basisstates(Hsub))
    if indexin === nothing
        throw(ArgumentError("Hilbert space $Hsub is not a sector of $H"))
    end
    qn = collect(quantumnumbers(H))[indexin]
    indices(qn, H)
end
function indices(qn::Q, H::BlockHilbertSpace{B,P,Q}) where {B,P,Q}
    dims = cumsum([length(H.qn_to_states[qn]) for qn in collect(quantumnumbers(H))])
    qn_index = findfirst(isequal(qn), collect(quantumnumbers(H)))
    if qn_index === nothing
        throw(ArgumentError("Quantum number $qn not found in Hilbert space $H"))
    end
    start_index = qn_index == 1 ? 1 : dims[qn_index-1] + 1
    end_index = dims[qn_index]
    start_index:end_index
end
indices(qn, H::AbstractHilbertSpace) = indices(sector(qn, H), H)
# indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)



@testitem "BlockHilbertSpace sectors" begin
    @fermions f

    H = hilbert_space(f, 1:4, NumberConservation())
    @test H isa BlockHilbertSpace
    @test collect(quantumnumbers(H)) == 0:4

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

    ## test fermions on block spaces
    import FermionicHilbertSpaces: fermions
    N = 4
    H = hilbert_space(f, 1:N, NumberConservation(0:N-1))
    @test size(fermions(H)[1], 1) == 2^N - 1
end


@testitem "Sector" begin
    import FermionicHilbertSpaces: sector, sectors, indices, quantumnumbers
    N = 4
    @fermions f
    H = hilbert_space(f, 1:N, NumberConservation())
    @test collect(quantumnumbers(H)) == 0:N
    Hns = sectors(H)
    for (ind, n) in enumerate(quantumnumbers(H))
        Hn = hilbert_space(f, 1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(Hns[n]) # Hn ≠ Hns[ind] since Hn is a SymmetricFockHilbertSpace
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hn, H)]
        @test basisstates(Hn) == basisstates(H)[indices(n, H)]
    end
    # no qns
    @majoranas γ
    Hnoqn = hilbert_space(f, 1:N)
    HMnoqn = majorana_hilbert_space(γ, 1:N)
    @test indices(only(quantumnumbers(Hnoqn)), Hnoqn) == 1:dim(Hnoqn)
    @test indices(only(quantumnumbers(HMnoqn)), HMnoqn) == 1:dim(HMnoqn)
    @test length(sectors(Hnoqn)) == length(sectors(HMnoqn)) == 1
    @test eltype(sectors(HMnoqn)) <: MajoranaHilbertSpace
    # Majorana hilbert spaces
    HM = majorana_hilbert_space(1:N, NumberConservation())
    @test quantumnumbers(HM) == 0:N÷2
    qn = 1
    HMqn = majorana_hilbert_space(1:N, NumberConservation(qn))
    @test basisstates(HMqn) == basisstates(HM)[indices(qn, HM)]
    @test length(sectors(HM)) == N ÷ 2 + 1
    @test eltype(sectors(HM)) <: MajoranaHilbertSpace
end
