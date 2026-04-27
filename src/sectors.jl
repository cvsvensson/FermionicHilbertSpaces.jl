"""
    SectorHilbertSpace

Hilbert space whose basis is grouped into sectors labeled by quantum numbers.
Use `quantumnumbers`, `sector`, and `indices` to access individual sectors.
"""
struct SectorHilbertSpace{B,P,Q,C} <: AbstractHilbertSpace{B}
    parent::P
    ordered_basis_states::Vector{B}
    state_to_index::OrderedDict{B,Int}
    qn_to_states::OrderedDict{Q,Vector{B}}
    constraint::C
end
SectorHilbertSpace(space::P, ordered_basis_states::AbstractVector{B}, state_to_index::OrderedDict{B,Int}, qn_to_states::OrderedDict{Q,Vector{B}}) where {B,P,Q} = SectorHilbertSpace{B,P,Q,Nothing}(space, ordered_basis_states, state_to_index, qn_to_states, nothing)
SectorHilbertSpace(space::P, ordered_basis_states::AbstractVector{B}, state_to_index::OrderedDict{B,Int}, qn_to_states::OrderedDict{Q,Vector{B}}, constraint::C) where {B,P,Q,C} = SectorHilbertSpace{B,P,Q,C}(space, ordered_basis_states, state_to_index, qn_to_states, constraint)
Base.hash(H::SectorHilbertSpace, h::UInt) = hash((H.parent, H.ordered_basis_states, H.state_to_index, H.qn_to_states), h)
Base.:(==)(H1::SectorHilbertSpace, H2::SectorHilbertSpace) = H1 === H2 || (H1.parent == H2.parent && H1.ordered_basis_states == H2.ordered_basis_states && H1.state_to_index == H2.state_to_index && H1.qn_to_states == H2.qn_to_states)
atomic_substate(n, f, space::SectorHilbertSpace) = atomic_substate(n, f, parent(space))

sector_space(space, states, ::Missing) = ConstrainedSpace(space, states)
function sector_space(space, states, sector_function, constraint=nothing)
    B = eltype(states)
    sort = Base.hasmethod(isless, Tuple{B,B})
    _qntostates = groupby(sector_function, states; sort)
    _sector_space(space, _qntostates, constraint)
end
function groupby(f::F, itr; sort=false) where F
    V = eltype(itr)
    d = OrderedDict{Any,Vector{V}}()
    for x in itr
        key = f(x)
        ismissing(key) && continue
        push!(get!(d, key) do
                V[]
            end, x)
    end
    ks = sort ? sort!(collect(keys(d))) : collect(keys(d))
    return OrderedDict(k => map(identity, d[k]) for k in ks)
end


function _sector_space(space, qn_to_states::OrderedDict{Q,<:AbstractVector{B}}, constraint=nothing) where {Q,B}
    ordered_states = reduce(vcat, values(qn_to_states), init=B[])
    state_indexdict = OrderedDict(zip(ordered_states, 1:length(ordered_states)))
    SectorHilbertSpace(space, ordered_states, state_indexdict, qn_to_states, constraint)
end

Base.parent(H::SectorHilbertSpace) = H.parent
dim(H::SectorHilbertSpace) = length(H.ordered_basis_states)
atomic_factors(H::SectorHilbertSpace) = atomic_factors(H.parent)
factors(H::SectorHilbertSpace) = factors(parent(H))
groups(H::SectorHilbertSpace) = groups(parent(H))
atomic_id(H::SectorHilbertSpace) = atomic_id(parent(H))
group_id(H::SectorHilbertSpace) = group_id(parent(H))

isconstrained(H::SectorHilbertSpace) = true
basisstates(H::SectorHilbertSpace) = H.ordered_basis_states
_find_position(Hsub::AbstractHilbertSpace, H::SectorHilbertSpace) = _find_position(Hsub, H.parent)
combine_states(substates, H::SectorHilbertSpace) = combine_states(substates, parent(H))
partial_trace_phase_factor(s1, s2, H::SectorHilbertSpace) = partial_trace_phase_factor(s1, s2, parent(H))
state_mapper(H::SectorHilbertSpace, Hs) = state_mapper(parent(H), Hs)
mode_ordering(H::SectorHilbertSpace) = mode_ordering(parent(H))

function basisstate(ind::Int, H::SectorHilbertSpace)
    (ind < 1 || ind > dim(H)) && throw(ArgumentError("Invalid state index $ind"))
    H.ordered_basis_states[ind]
end

state_index(state, H::SectorHilbertSpace) = get(H.state_to_index, state, missing)

"""
    quantumnumbers(H)

Return the quantum-number labels that define the sectors of `H`.
For spaces without sector structure, this returns `(nothing,)`.
"""
quantumnumbers(H::SectorHilbertSpace) = collect(keys(H.qn_to_states))
quantumnumbers(::AbstractHilbertSpace) = (nothing,)

"""
    sector(qn, H)

Return the sector of `H` corresponding to quantum number `qn`.
For `qn === nothing`, this returns `H` for non-sector spaces.
"""
sector(qn, H::SectorHilbertSpace) = constrain_space(parent(H), H.qn_to_states[qn])
sector(::Nothing, H::AbstractHilbertSpace) = H
sector(::Nothing, ::SectorHilbertSpace) = constrain_space(parent(H), H.qn_to_states[qn])

# sectors(H::SectorHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))
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
function indices(Hsub, H::AbstractHilbertSpace)
    sector_list = sectors(H)
    indexin = findfirst(isequal(Hsub), sector_list)
    # map(state -> state_index(state, H), basisstates(Hsub))
    if indexin === nothing
        throw(ArgumentError("Hilbert space $Hsub is not a sector of $H"))
    end
    qn = quantumnumbers(H)[indexin]
    indices(qn, H)
end
function indices(qn::Q, H::SectorHilbertSpace{B,P,Q}) where {B,P,Q}
    dims = cumsum([length(H.qn_to_states[qn]) for qn in collect(quantumnumbers(H))])
    qn_index = findfirst(isequal(qn), collect(quantumnumbers(H)))
    if qn_index === nothing
        throw(ArgumentError("Quantum number $qn not found in Hilbert space $H"))
    end
    start_index = qn_index == 1 ? 1 : dims[qn_index-1] + 1
    end_index = dims[qn_index]
    start_index:end_index
end
# indices(qn, H::AbstractHilbertSpace) = indices(sector(qn, H), H)
indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)

_precomputation_before_operator_application(ops, space::SectorHilbertSpace) = _precomputation_before_operator_application(ops, parent(space))

@testitem "SectorHilbertSpace sectors" begin
    @fermions f

    H = hilbert_space(f, 1:4, NumberConservation())
    @test H isa SectorHilbertSpace
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
    @test Hprod isa SectorHilbertSpace
    qns = quantumnumbers(Hprod)
    @test all(qn -> dim(sector(qn, Hprod)) > 0, qns)

    ## test fermions on sector spaces
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
        @test basisstates(Hn) == basisstates(Hns[ind])
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == basisstates(H)[indices(Hns[ind], H)]
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
    # The size of the even sector should match the number of even-parity states
    even_states = [f for f in basisstates(H) if parity(f) == 1]
    @test size(even_sector, 1) == length(even_states)
    # The values should match the corresponding sector in m
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
    @test_throws KeyError sector(99, Hf)
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
    spin_up_conservation = NumberConservation(Nup, map(l -> f[l], spin_up_sites))
    spin_down_sites = filter(label -> label[2] == :↓, labels)
    spin_down_conservation = NumberConservation(Ndn, map(l -> f[l], spin_down_sites))
    no_double_occupation = prod(NumberConservation(0:1, [f[(k, σ)] for σ in spins]) for k in spatial_labels)

    qn = spin_up_conservation * spin_down_conservation * no_double_occupation
    H = hilbert_space(f, labels, qn)
    hopping_symham = sum(zip(spatial_labels, spatial_labels[2:end])) do (i, j)
        sum(spins) do σ
            f[(i, σ)]' * f[(j, σ)] + hc
        end
    end
    @test_throws ArgumentError matrix_representation(hopping_symham, H)
    @test size(matrix_representation(hopping_symham, H; projection=true), 1) == dim(H)
end

maximum_particles(H::SectorHilbertSpace) = maximum_particles(parent(H))

@testitem "Sector propagation in tensor_product" begin
    @fermions f

    # Both inputs NumberConservation → QNs are [n1, n2] vectors
    k = 2
    H1 = hilbert_space(f, 1:k, NumberConservation())
    H2 = hilbert_space(f, k+1:2k, NumberConservation())
    H = tensor_product(H1, H2)
    @test H isa SectorHilbertSpace
    qns = quantumnumbers(H)
    # should have (k+1)^2 sectors, one per (n1,n2) pair
    @test length(qns) == (k + 1)^2
    for n1 in 0:k, n2 in 0:k
        qn = [n1, n2]
        @test qn in qns
        @test dim(sector(qn, H)) == binomial(k, n1) * binomial(k, n2)
    end
    # states are still correct
    @test dim(H) == 2^(2k)
    # indices match basisstates
    for qn in qns
        Hsec = sector(qn, H)
        @test basisstates(H)[indices(qn, H)] == basisstates(Hsec)
    end

    # One sector input + one plain input → QN type is plain (e.g. Int)
    Hplain = hilbert_space(f, k+1:2k)
    Hmix = tensor_product(H1, Hplain)
    @test Hmix isa SectorHilbertSpace
    for n in 0:k
        @test dim(sector(n, Hmix)) == binomial(k, n) * 2^k
    end

    # ParityConservation input
    H1p = hilbert_space(f, 1:k, ParityConservation())
    H2p = hilbert_space(f, k+1:2k, ParityConservation())
    Hp = tensor_product(H1p, H2p)
    @test Hp isa SectorHilbertSpace
    # combined parity QNs are [p1, p2] with p ∈ {-1, 1}
    @test length(quantumnumbers(Hp)) == 4
end

@testitem "Sector propagation with custom SectorConstraint" begin
    import FermionicHilbertSpaces: SectorConstraint, fermionnumber
    @fermions f
    k = 2

    # Custom sector: label states by whether they contain at least one particle
    sector_fn = state -> fermionnumber(state) >= 1 ? :has_particle : :vacuum
    H1 = hilbert_space(f, 1:k, SectorConstraint(sector_fn))
    @test H1 isa SectorHilbertSpace
    @test sort(quantumnumbers(H1); by=string) == sort([:has_particle, :vacuum]; by=string)
    @test dim(sector(:vacuum, H1)) == 1          # only the all-empty state
    @test dim(sector(:has_particle, H1)) == 2^k - 1

    # Combine with a NumberConservation sector space
    H2 = hilbert_space(f, k+1:2k, NumberConservation())
    H = tensor_product(H1, H2)
    @test H isa SectorHilbertSpace

    # QNs are [custom_label, n2] vectors: 2 labels × (k+1) numbers
    qns = quantumnumbers(H)
    @test length(qns) == 2 * (k + 1)

    # Dimension of each combined sector is correct
    for (label, expected_h1_dim) in [(:vacuum, 1), (:has_particle, 2^k - 1)]
        for n2 in 0:k
            qn = [label, n2]
            @test qn in qns
            @test dim(sector(qn, H)) == expected_h1_dim * binomial(k, n2)
        end
    end
    @test dim(H) == dim(H1) * dim(H2)

    # States are still complete and correctly indexed
    for qn in qns
        @test basisstates(H)[indices(qn, H)] == basisstates(sector(qn, H))
    end
end