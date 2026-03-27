"""
    BlockHilbertSpace

Hilbert space whose basis is grouped into sectors labeled by quantum numbers.
Use `quantumnumbers`, `sector`, and `indices` to access individual blocks.
"""
struct BlockHilbertSpace{B,P,Q} <: AbstractHilbertSpace{B}
    parent::P
    ordered_basis_states::Vector{B}
    state_to_index::Dictionary{B,Int}
    qn_to_states::Dictionary{Q,Vector{B}}
end
BlockHilbertSpace(space::P, ordered_basis_states::AbstractVector{B}, state_to_index::Dictionary{B,Int}, qn_to_states::Dictionary{Q,Vector{B}}) where {B,P,Q} = BlockHilbertSpace{B,P,Q}(space, ordered_basis_states, state_to_index, qn_to_states)
Base.hash(H::BlockHilbertSpace, h::UInt) = hash((H.parent, H.ordered_basis_states, H.state_to_index, H.qn_to_states), h)
Base.:(==)(H1::BlockHilbertSpace, H2::BlockHilbertSpace) = H1 === H2 || (H1.parent == H2.parent && H1.ordered_basis_states == H2.ordered_basis_states && H1.state_to_index == H2.state_to_index && H1.qn_to_states == H2.qn_to_states)
atomic_substate(n, f, space::BlockHilbertSpace) = atomic_substate(n, f, parent(space))

function block_space(space, states, sector_function)
    _qntostates = group(state -> sector_function(state), states)
    _block_space(space, _qntostates)
end
function _block_space(space, _qntostates::Dictionary{Q,<:AbstractVector{B}}) where {Q,B}
    inds = keys(_qntostates)
    filt_inds = filter(!ismissing, inds)
    qn_to_states = map(collect, getindices(_qntostates, filt_inds))
    sortkeys!(qn_to_states)
    ordered_states = reduce(vcat, qn_to_states, init=B[])
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
state_mapper(H::BlockHilbertSpace, Hs) = state_mapper(parent(H), Hs)
mode_ordering(H::BlockHilbertSpace) = mode_ordering(parent(H))

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
        nsectors = length(qns)

        if !isempty(qns)
            max_printed_sectors = 5
            edge_sectors = 1

            if nsectors > max_printed_sectors
                print(io, "\nSectors: $nsectors total [")

                for (i, qn) in enumerate(qns[1:edge_sectors])
                    i > 1 && print(io, ", ")
                    print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
                end

                print(io, ", ..., ")

                for (i, qn) in enumerate(qns[end-edge_sectors+1:end])
                    i > 1 && print(io, ", ")
                    print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
                end

                print(io, "]")
            else
                print(io, "\nSectors: ")
                for (i, qn) in enumerate(qns)
                    i > 1 && print(io, ", ")
                    print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
                end
            end
        end
    end
end

function basisstate(ind::Int, H::BlockHilbertSpace)
    (ind < 1 || ind > dim(H)) && throw(ArgumentError("Invalid state index $ind"))
    H.ordered_basis_states[ind]
end

state_index(state, H::BlockHilbertSpace) = get(H.state_to_index, state, missing)

"""
    quantumnumbers(H)

Return the quantum-number labels that define the sectors of `H`.
For spaces without sector structure, this returns `(nothing,)`.
"""
quantumnumbers(H::BlockHilbertSpace) = keys(H.qn_to_states)
quantumnumbers(::AbstractHilbertSpace) = (nothing,)

"""
    sector(qn, H)

Return the sector of `H` corresponding to quantum number `qn`.
For `qn === nothing`, this returns `H` for non-block spaces.
"""
sector(qn, H::BlockHilbertSpace) = constrain_space(parent(H), H.qn_to_states[qn])
sector(::Nothing, H::AbstractHilbertSpace) = H
sector(::Nothing, ::BlockHilbertSpace) = constrain_space(parent(H), H.qn_to_states[qn])

# sectors(H::BlockHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))
"""
    sectors(H)

Return all sectors of `H` in the same order as `quantumnumbers(H)`.
"""
sectors(H::AbstractHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))

"""
    indices(Hsub, H)
    indices(qn, H)

Return the basis-state indices in `H` belonging to a given sector, specified
either by a sector Hilbert space `Hsub` or by a quantum number `qn`.
"""
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
indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)

_precomputation_before_operator_application(ops, space::BlockHilbertSpace) = _precomputation_before_operator_application(ops, parent(space))

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
        @test basisstates(Hn) == basisstates(Hns[n])
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hns[n], H)]
        @test basisstates(Hn) == basisstates(H)[indices(n, H)]
    end
    # no qns
    @majoranas γ
    Hnoqn = hilbert_space(f, 1:N)
    HMnoqn = hilbert_space(γ, 1:N)
    @test indices(only(quantumnumbers(Hnoqn)), Hnoqn) == 1:dim(Hnoqn)
    @test indices(only(quantumnumbers(HMnoqn)), HMnoqn) == 1:dim(HMnoqn)
    @test length(sectors(Hnoqn)) == length(sectors(HMnoqn)) == 1
    @test eltype(sectors(HMnoqn)) <: FermionicHilbertSpaces.MajoranaHilbertSpace
    # Majorana hilbert spaces
    HM = hilbert_space(γ, 1:N, NumberConservation())
    @test collect(quantumnumbers(HM)) == 0:N÷2
    qn = 1
    HMqn = hilbert_space(γ, 1:N, NumberConservation(qn))
    @test basisstates(HMqn) == basisstates(HM)[indices(qn, HM)]
    @test length(sectors(HM)) == N ÷ 2 + 1
    @test eltype(sectors(HM)) <: FermionicHilbertSpaces.MajoranaHilbertSpace
end


@testitem "Symmetry basisstates" begin
    import FermionicHilbertSpaces: fermionnumber
    @fermions f
    H = hilbert_space(f, 1:5)
    Hcons = constrain_space(H, ParityConservation())
    @test dim(Hcons) == 2^5
    Hcons = constrain_space(H, ParityConservation(1))
    @test dim(Hcons) == 2^4
    odd_focks = basisstates(constrain_space(H, ParityConservation(-1)))
    @test all(isodd ∘ fermionnumber, odd_focks)
    @test dim(constrain_space(H, ParityConservation([-1, 1]))) == 2^5

    ## ProductSymmetry
    qn = ParityConservation([1]) * NumberConservation(1:2, H.modes[1:3])
    states = FermionicHilbertSpaces.generate_states(H.modes, qn)
    H2 = hilbert_space(f, 1:5, qn)
    T = FermionicHilbertSpaces.statetype(H2)
    @test sort(basisstates(H2)) == sort(map(state -> FermionicHilbertSpaces.catenate_fock_states(state, H.modes, T), states))
    @test all(iseven ∘ fermionnumber, basisstates(H2))
end

@testitem "sector" begin
    import FermionicHilbertSpaces: sector, parity
    # Create a Hilbert space with parity symmetry
    @fermions f
    labels = 1:3
    qn = ParityConservation()
    H = hilbert_space(f, labels, qn)
    n = length(basisstates(H))
    m = reshape(1:(n^2), n, n)  # simple test matrix
    # Get the sector for parity = 1
    even_inds = indices(1, H)
    even_sector = m[even_inds, even_inds]
    # The size of the even sector block should match the number of even-parity states
    even_states = [f for f in basisstates(H) if parity(f) == 1]
    @test size(even_sector, 1) == length(even_states)
    # The values should match the corresponding block in m
    # Get the indices of even states in the full basisstates list
    even_inds = findall(f -> parity(f) == 1, basisstates(H))
    @test even_sector == m[even_inds, even_inds]

    # Test with NumberConservation
    import FermionicHilbertSpaces: fermionnumber
    qn_f = NumberConservation([1, 2])
    Hf = hilbert_space(f, labels, qn_f)
    n_f = length(basisstates(Hf))
    m_f = reshape(1:(n_f^2), n_f, n_f)
    # Test sector for fermion number = 1
    sector1_inds = indices(1, Hf)
    sector1 = m_f[sector1_inds, sector1_inds]
    states1 = [f for f in basisstates(Hf) if fermionnumber(f) == 1]
    inds1 = findall(f -> fermionnumber(f) == 1, basisstates(H))
    @test size(sector1, 1) == length(states1)
    @test sector1 == m_f[inds1, inds1]
    # Test sector for fermion number = 2
    sector2_inds = indices(2, Hf)
    sector2 = m_f[sector2_inds, sector2_inds]
    states2 = [f for f in basisstates(Hf) if fermionnumber(f) == 2]
    inds2 = findall(f -> fermionnumber(f) == 2, basisstates(Hf))
    @test size(sector2, 1) == length(states2)
    @test sector2 == m_f[inds2, inds2]
    # Test that an invalid sector throws an error
    @test_throws "Dictionaries.IndexError" sector(99, Hf)
    @test_throws ArgumentError indices(99, Hf)
end

@testitem "No double occupation projection" begin
    @fermions f
    N = 4
    Nup = 2
    Ndn = 1
    spins = (:↑, :↓)
    spatial_labels = 1:N
    labels = vec(collect(Base.product(spatial_labels, spins)))
    spin_up_sites = filter(label -> label[2] == :↑, labels)
    spin_up_conservation = NumberConservation(Nup, hilbert_space(f, spin_up_sites))
    spin_down_sites = filter(label -> label[2] == :↓, labels)
    spin_down_conservation = NumberConservation(Ndn, hilbert_space(f, spin_down_sites))
    no_double_occupation = prod(NumberConservation(0:1, hilbert_space(f, [(k, σ) for σ in spins])) for k in spatial_labels)

    qn = spin_up_conservation * spin_down_conservation * no_double_occupation
    H = hilbert_space(f, labels, qn)
    hopping_symham = sum(zip(spatial_labels, spatial_labels[2:end])) do (i, j)
        sum(spins) do σ
            f[(i, σ)]' * f[(j, σ)] + hc
        end
    end
    @test_throws MethodError matrix_representation(hopping_symham, H)
    @test size(matrix_representation(hopping_symham, H; projection=true), 1) == dim(H)
end

maximum_particles(H::BlockHilbertSpace) = maximum_particles(parent(H))