"""
    SectorHilbertSpace

Hilbert space whose basis is grouped into sectors labeled by quantum numbers.
Use `quantumnumbers`, `sector`, and `indices` to access individual sectors.
"""
struct SectorHilbertSpace{B,P,Q,C,I} <: AbstractSubspace{B,P}
    parent::P
    ordered_basis_states::Vector{B}
    qn_to_states::OrderedDict{Q,Vector{B}}
    constraint::C
    index::I
    SectorHilbertSpace{B,P,Q,C,I}(parent, ordered_basis_states, qn_to_states, constraint, index) where {B,P,Q,C,I} = new{B,P,Q,C,I}(parent, ordered_basis_states, qn_to_states, constraint, index)
end
SectorHilbertSpace(parent, ordered_basis_states, qn_to_states, constraint, index) = SectorHilbertSpace{eltype(ordered_basis_states),typeof(parent),keytype(qn_to_states),typeof(constraint),typeof(index)}(parent, ordered_basis_states, qn_to_states, constraint, index)
SectorHilbertSpace(::AbstractSubspace, ordered_basis_states, qn_to_states, constraint, index) = throw(nested_subspace_error())
SectorHilbertSpace(space, ordered_basis_states, qn_to_states, constraint; selected_by = constraint) = SectorHilbertSpace(space, ordered_basis_states, qn_to_states, constraint, build_subspace_index(index_constraint(selected_by, space), space, ordered_basis_states))

Base.parent(H::SectorHilbertSpace) = H.parent
subspace_states(H::SectorHilbertSpace) = H.ordered_basis_states
subspace_index(H::SectorHilbertSpace) = H.index
subspace_key(H::SectorHilbertSpace) = (parent(H), H.ordered_basis_states, H.qn_to_states)
rewrap(H::SectorHilbertSpace, parent) = SectorHilbertSpace(parent, H.ordered_basis_states, H.qn_to_states, H.constraint, H.index)

sector_space(space, states, ::Missing) = ConstrainedSpace(space, states)
# `selected_by` is what selected `states`, which can be more than `constraint`, for example in a tensor product of constrained spaces.
function sector_space(space, states, sector_function, constraint; selected_by = constraint)
    B = statetype(space)
    parent, members = unnest(space, states)
    selected_by = coalesce(combine_constraints(own_constraint(space), selected_by), NoSymmetry())
    if members isa Vector{B} && !isempty(members) && allequal(sector_function, members)
        qn = sector_function(first(members))
        ismissing(qn) || return _sector_space(parent, OrderedDict(qn => members), constraint; selected_by)
    end
    _qntostates = groupby(sector_function, members, B)
    return _sector_space(parent, _qntostates, constraint; selected_by)
end
function _group(f::F, itr, ::Type{B}) where {F,B}
    K = nonmissingtype(Base.promote_op(f, eltype(itr)))
    _group!(OrderedDict{isconcretetype(K) ? K : Any,Vector{B}}(), f, itr)
end
function _group!(d, f::F, itr) where {F}
    for x in itr
        key = f(x)
        ismissing(key) && continue
        push!(get!(valtype(d), d, key), x)
    end
    d
end
function groupby(f::F, itr; sortkeys=false, sortvals=false) where {F}
    B = eltype(itr)
    d = _group(f, itr, B)
    _ks = identity.(keys(d))
    K = eltype(_ks)
    ks = sortkeys ? sort!(_ks) : _ks
    newdict = OrderedDict(k => map(identity, d[k]) for k in ks)
    if sortvals
        foreach(sort!, values(newdict))
    end
    newdict
end
function groupby(f::F, itr, ::Type{B}; sortkeys=:auto, sortvals=false) where {F,B}
    d = _group(f, itr, B)
    _ks = identity.(keys(d))
    K = eltype(_ks)
    sortkeys = sortkeys == :auto ? Base.hasmethod(isless, Tuple{K,K}) : false
    ks = sortkeys ? sort!(_ks) : _ks
    if sortvals
        foreach(sort!, values(d))
    end
    return OrderedDict{K,Vector{B}}(k => d[k] for k in ks)
end

function _sector_space(space, qn_to_states::OrderedDict{Q, <:AbstractVector{B}}, constraint; selected_by = constraint) where {Q, B}
    ordered_states::Vector{B} = length(qn_to_states) == 1 ? only(values(qn_to_states)) : vcat(values(qn_to_states)...)
    return SectorHilbertSpace(space, ordered_states, qn_to_states, constraint; selected_by)
end


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
Throws an `ArgumentError` if `qn` is not one of `quantumnumbers(H)`.
"""
sector(qn, H::SectorHilbertSpace, constraint=NoSymmetry()) = constrain_space(parent(H), constraint, _sector_states(qn, H))
sector(::Nothing, H::AbstractHilbertSpace, constraint=NoSymmetry()) = constrain_space(H, constraint)
sector(qn::Nothing, H::SectorHilbertSpace, constraint=NoSymmetry()) = constrain_space(parent(H), constraint, _sector_states(qn, H))

function _sector_states(qn, H::SectorHilbertSpace)
    states = get(H.qn_to_states, qn, nothing)
    isnothing(states) && throw(ArgumentError("Unknown quantum number, got `$qn`; the available quantum numbers are `$(quantumnumbers(H))`"))
    states
end

# sectors(H::SectorHilbertSpace) = map(qn -> sector(qn, H), quantumnumbers(H))
"""
    sectors(H)

Return all sectors of `H` in the same order as `quantumnumbers(H)`.
"""
sectors(H::AbstractHilbertSpace, constraint=NoSymmetry()) = map(qn -> sector(qn, H, constraint), quantumnumbers(H))

function space_indices(Hsub::AbstractHilbertSpace, H::SectorHilbertSpace)
    # `Hsub` is a sector of `H` when it equals `sector(qn, H)` for some `qn`
    qn = Hsub isa ConstrainedSpace && parent(Hsub) == parent(H) ? findfirst(isequal(subspace_states(Hsub)), H.qn_to_states) : nothing
    isnothing(qn) ? space_indices_generic(Hsub, H) : qn_indices(qn, H)
end
space_indices(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = space_indices_generic(Hsub, H)
function space_indices_generic(Hsub, H)
    # fallback for when the sector space isn't exactly in the sectors list (e.g. due to constraints)
    match_same_atoms(Hsub, H)
    map(basisstates(Hsub)) do state
        ind = state_index(state, H)
        if iszero(ind)
            throw(ArgumentError("The state `$state` of the sector space `$Hsub` is not in the parent space `$H`."))
        end
        ind
    end
end
"""
    indices(Hsub, H)
    indices(qn, H)

Return the basis-state indices in `H` belonging to a given sector, specified
either by a sector Hilbert space `Hsub` or by a quantum number `qn`.
Throws an `ArgumentError` if `qn` is not one of `quantumnumbers(H)`.
"""
indices(qn, H) = qn_indices(qn, H)
indices(qn::AbstractHilbertSpace, H::AbstractHilbertSpace) = space_indices(qn, H)
indices(Hsub::AbstractHilbertSpace, H::SectorHilbertSpace) = space_indices(Hsub, H)
indices(qn::Q, H::SectorHilbertSpace{B,P,Q}) where {B,P,Q} = qn_indices(qn, H)
function qn_indices(qn, H::SectorHilbertSpace)
    _sector_states(qn, H)
    dims = cumsum([length(H.qn_to_states[qn]) for qn in collect(quantumnumbers(H))])
    qn_index = findfirst(isequal(qn), collect(quantumnumbers(H)))
    start_index = qn_index == 1 ? 1 : dims[qn_index-1] + 1
    end_index = dims[qn_index]
    start_index:end_index
end
indices(::Nothing, H::AbstractHilbertSpace) = 1:dim(H)

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
    H = hilbert_space(f, 1:N, NumberConservation(0:(N-1)))
    @test size(fermions(H)[1], 1) == 2^N - 1
end


@testitem "Sector" begin
    N = 4
    @fermions f
    H = hilbert_space(f, 1:N, NumberConservation())
    @test collect(quantumnumbers(H)) == 0:N
    Hns = sectors(H)
    for (ind, n) in enumerate(quantumnumbers(H))
        Hn = hilbert_space(f, 1:N, NumberConservation(n))
        @test basisstates(Hn) == basisstates(Hns[ind])
        @test basisstates(Hn) == basisstates(sector(n, H))
        @test basisstates(Hn) == collect(basisstates(H))[indices(Hns[ind], H)]
        @test basisstates(Hn) == collect(basisstates(H))[indices(n, H)]
    end
    # no qns
    @majoranas γ
    Hnoqn = hilbert_space(f, 1:N)
    HMnoqn = hilbert_space(γ, 1:N)
    @test indices(only(quantumnumbers(Hnoqn)), Hnoqn) == 1:dim(Hnoqn)
    @test indices(only(quantumnumbers(HMnoqn)), HMnoqn) == 1:dim(HMnoqn)
    @test length(sectors(Hnoqn)) == length(sectors(HMnoqn)) == 1
    @test all(H -> H isa FermionicHilbertSpaces.MajoranaHilbertSpace, sectors(HMnoqn))
    # Majorana hilbert spaces
    HM = hilbert_space(γ, 1:N, NumberConservation())
    @test collect(quantumnumbers(HM)) == 0:(N÷2)
    qn = 1
    HMqn = hilbert_space(γ, 1:N, NumberConservation(qn))
    @test basisstates(HMqn) == collect(basisstates(HM))[indices(qn, HM)]
    @test length(sectors(HM)) == N ÷ 2 + 1
    @test all(H -> parent(H) isa FermionicHilbertSpaces.MajoranaHilbertSpace, sectors(HM))
end

@testitem "Number-conserving Hamiltonian is block diagonal in sectors" begin
    using LinearAlgebra
    @fermions f
    N = 4
    ham = sum(rand(ComplexF64) * f[i]' * f[j] + hc for i in 1:N, j in 1:N) +
          sum(rand() * f[i]' * f[i] * f[j]' * f[j] for i in 1:N, j in 1:N)
    H = hilbert_space(f, 1:N, NumberConservation())
    M = representation(ham, H, :dense)
    spectrum(m) = eigvals(Hermitian(m))
    sector_spectra = map(quantumnumbers(H)) do n
        inds = indices(n, H)
        rest = setdiff(1:dim(H), inds)
        @test iszero(M[inds, rest])
        Hn = hilbert_space(f, 1:N, NumberConservation(n))
        @test spectrum(M[inds, inds]) ≈ spectrum(representation(ham, Hn, :dense))
        spectrum(M[inds, inds])
    end
    @test sort(reduce(vcat, sector_spectra)) ≈ spectrum(representation(ham, hilbert_space(f, 1:N), :dense))
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
    @test sort(collect(basisstates(H2))) == sort(map(state -> FermionicHilbertSpaces.catenate_fock_states(state, H.modes, T), states))
    @test all(iseven ∘ fermionnumber, basisstates(H2))
end

@testitem "sector" begin
    import FermionicHilbertSpaces: parity
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
    @test_throws "Unknown quantum number, got `99`" sector(99, Hf)
    @test_throws "Unknown quantum number, got `99`" indices(99, Hf)
    @test_throws ArgumentError sector(99, Hf)
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
    @test_throws ArgumentError representation(hopping_symham, H)
    @test size(representation(hopping_symham, H; projection=true), 1) == dim(H)
end

@testitem "Sector propagation in tensor_product" begin
    @fermions f

    # Both inputs NumberConservation → QNs are [n1, n2] vectors
    k = 2
    H1 = hilbert_space(f, 1:k, NumberConservation())
    H2 = hilbert_space(f, (k+1):2k, NumberConservation())
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
        @test collect(basisstates(H))[indices(qn, H)] == collect(basisstates(Hsec))
    end

    # One sector input + one plain input → QN type is plain (e.g. Int)
    Hplain = hilbert_space(f, (k+1):2k)
    Hmix = tensor_product(H1, Hplain)
    @test Hmix isa SectorHilbertSpace
    for n in 0:k
        @test dim(sector(n, Hmix)) == binomial(k, n) * 2^k
    end

    # ParityConservation input
    H1p = hilbert_space(f, 1:k, ParityConservation())
    H2p = hilbert_space(f, (k+1):2k, ParityConservation())
    Hp = tensor_product(H1p, H2p)
    @test Hp isa SectorHilbertSpace
    # combined parity QNs are [p1, p2] with p ∈ {-1, 1}
    @test length(quantumnumbers(Hp)) == 4
end

@testitem "Sector propagation with custom SectorConstraint" begin
    import FermionicHilbertSpaces: fermionnumber
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
    H2 = hilbert_space(f, (k+1):2k, NumberConservation())
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
        @test collect(basisstates(H))[indices(qn, H)] == collect(basisstates(sector(qn, H)))
    end

    ##
    subconstraint = NumberConservation(missing, [H1, H2], [-1, 1])
    for (label, expected_h1_dim) in [(:vacuum, 1), (:has_particle, 2^k - 1)]
        for n2 in 0:k
            qn = [label, n2]
            @test qn in qns
            @test dim(sector(qn, H, subconstraint)) == expected_h1_dim * binomial(k, n2)
        end
    end
    @test dim(H) == dim(H1) * dim(H2)
    for qn in qns
        @test Set(collect(basisstates(H))[indices(qn, H)]) == Set(basisstates(sector(qn, H, subconstraint)))
    end

end

@testitem "Generic sectors fallback" begin
    @fermions f
    H = hilbert_space(f, 1:2, ParityConservation())
    Hsecs = sectors(H, NumberConservation(1))
    inds = map(Hsec -> indices(Hsec, H), Hsecs)
    @test Set(inds[1]) == Set(1:2)
    @test length(inds[2]) == 0
end

@testitem "A sector of a subspace keeps only the states of the subspace" begin
    @fermions f
    H = hilbert_space(f, 1:3)
    states = collect(basisstates(H))
    Hc = constrain_space(H, states[1:4])
    for constraint in (NumberConservation(), ParityConservation())
        @test Set(basisstates(constrain_space(Hc, constraint, states))) == Set(states[1:4])
    end
end

@testitem "state_index of a subspace takes only states" begin
    @fermions f
    @test_throws MethodError state_index(1, hilbert_space(f, 1:2, NumberConservation()))
    Hg = constrain_space(GenericHilbertSpace(:A, [:a, :b, :c]), [:a, :c])
    @test state_index(:c, Hg) == 2
    @test state_index(:b, Hg) == 0
end
