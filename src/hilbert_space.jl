struct GenericHilbertSpace{L,S} <: AbstractHilbertSpace
    label::L
    basisstates::S
end
Base.:(==)(H1::GenericHilbertSpace, H2::GenericHilbertSpace) = H1 === H2 || (H1.label == H2.label && H1.basisstates == H2.basisstates)
Base.hash(H::GenericHilbertSpace, h::UInt) = hash((H.label, H.basisstates), h)
basisstates(H::GenericHilbertSpace) = H.basisstates
Base.keys(H::GenericHilbertSpace) = (H.label,)
state_index(state, H::GenericHilbertSpace) = findfirst(==(state), H.basisstates)

struct ProductState{F,O} <: AbstractBasisState
    fock_state::F
    other_states::O
end
Base.:(==)(s1::ProductState, s2::ProductState) = s1 === s2 || (s1.fock_state == s2.fock_state && s1.other_states == s2.other_states)
Base.hash(s::ProductState, h::UInt) = hash((s.fock_state, s.other_states), h)
ProductState{Nothing}(other_states) = ProductState{Nothing,typeof(other_states)}(nothing, other_states)
Base.show(io::IO, s::ProductState) = print(io, "ProductState(", s.fock_state, ", ", s.other_states, ")")
Base.show(io::IO, s::ProductState{Nothing}) = print(io, "ProductState(", s.other_states, ")")

struct ProductSpace{HF,HS} <: AbstractHilbertSpace
    fock_space::HF
    other_spaces::HS
    function ProductSpace(fspace::HF, ospaces::HS) where {HF<:Union{<:AbstractFockHilbertSpace,Nothing},HS}
        length(ospaces) == 0 && return fspace
        if isnothing(fspace) && length(ospaces) == 1
            return only(ospaces)
        end
        new{HF,HS}(fspace, ospaces)
    end
end
Base.keys(H::ProductSpace{Nothing}) = mapreduce(keys, TupleTools.vcat, H.other_spaces)
Base.keys(H::ProductSpace) = vcat(keys(H.fock_space), [mapreduce(keys, TupleTools.vcat, H.other_spaces)...])
Base.:(==)(H1::ProductSpace, H2::ProductSpace) = H1 === H2 || (H1.fock_space == H2.fock_space && H1.other_spaces == H2.other_spaces)
Base.hash(H::ProductSpace, h::UInt) = hash((H.fock_space, H.other_spaces), h)
ProductSpace{Nothing}(other_spaces) = ProductSpace(nothing, other_spaces)

# label(H::ProductSpace) = (label(H.fock_space), map(label, H.other_spaces)...)
# label(H::ProductSpace{Nothing}) = map(label, H.other_spaces)
# label(H::AbstractFockHilbertSpace) = keys(H)
basisstates(H::ProductSpace) = Iterators.map(s -> ProductState(s...), Iterators.product(basisstates(H.fock_space), Iterators.product(map(basisstates, H.other_spaces)...)))
basisstates(H::ProductSpace{Nothing}) = Iterators.map(ProductState{Nothing}, Iterators.product(map(basisstates, H.other_spaces)...))

function basisstate(n::Int, H::ProductSpace{Nothing})
    I = CartesianIndices(map(dim, H.other_spaces))[n]
    ProductState(nothing, ntuple(i -> basisstate(I[i], H.other_spaces[i]), length(H.other_spaces)))
end
function basisstate(n::Int, H::ProductSpace)
    I = CartesianIndices((dim(H.fock_space), map(dim, H.other_spaces)...))[n]
    ProductState(basisstate(I[1], H.fock_space), ntuple(i -> basisstate(I[i+1], H.other_spaces[i]), length(H.other_spaces)))
end
dim(H::ProductSpace{Nothing}) = prod(dim, H.other_spaces; init=1)
dim(H::ProductSpace) = dim(H.fock_space) * prod(dim, H.other_spaces; init=1)
function state_index(state::ProductState, H::ProductSpace)
    ind = state_index(state.fock_state, H.fock_space)
    n = ind
    dimprod = dim(H.fock_space)
    for (state, space) in zip(state.other_states, H.other_spaces)
        n += (state_index(state, space) - 1) * dimprod
        dimprod *= dim(space)
    end
    return n
end
function state_index(state::ProductState{Nothing}, H::ProductSpace{Nothing})
    n = 1
    dimprod = 1
    for (state, space) in zip(state.other_states, H.other_spaces)
        n += (state_index(state, space) - 1) * dimprod
        dimprod *= dim(space)
    end
    return n
end

simple_complementary_subsystem(H::ProductSpace, Hsub::AbstractHilbertSpace) = complementary_subsystem(H, Hsub)
function complementary_subsystem(H::ProductSpace, Hsub::AbstractHilbertSpace)
    spaces = [H.other_spaces[i] for i in eachindex(H.other_spaces) if H.other_spaces[i] != Hsub]
    length(spaces) + 1 == length(H.other_spaces) || throw(ArgumentError("Hsub must be one of the spaces in the product space H"))
    ProductSpace(H.fock_space, spaces)
end
function complementary_subsystem(H::ProductSpace, Hsub::AbstractFockHilbertSpace)
    fcomp = complementary_subsystem(H.fock_space, Hsub)
    ProductSpace(fcomp, H.other_spaces)
end
function complementary_subsystem(H::ProductSpace, Hsub::ProductSpace)
    other_spaces = setdiff(H.other_spaces, Hsub.other_spaces)
    fcomp = complementary_subsystem(H.fock_space, Hsub.fock_space)
    if length(other_spaces) == 0
        return fcomp
    end
    ProductSpace(fcomp, other_spaces)
end
function Base.show(io::IO, H::ProductSpace{Nothing})
    d = dim(H)
    dimstring = ""
    for (n, H) in enumerate(H.other_spaces)
        n > 1 && (dimstring *= "×")
        dimstring *= "$(dim(H))"
    end
    println(io, "$(d)-dimensional ProductSpace($dimstring)")
    n = length(H.other_spaces)
    print(io, "$n spaces: ", keys(H))
end
function Base.show(io::IO, H::ProductSpace)
    d = dim(H)
    dimstring = "$(dim(H.fock_space))"
    for (n, H) in enumerate(H.other_spaces)
        (dimstring *= "×")
        dimstring *= "$(dim(H))"
    end
    println(io, "$(d)-dimensional ProductSpace($dimstring)")
    N = length(keys(H.fock_space))
    println(io, "$N fermions: ", keys(H.fock_space))
    n = length(H.other_spaces)
    print(io, "$n other spaces: ", map(keys, H.other_spaces))
end

flat_non_fock_spaces(Hs) = foldl(_flat_non_fock_spaces, Hs; init=())
_flat_non_fock_spaces(acc, H::AbstractHilbertSpace) = (acc..., H)
_flat_non_fock_spaces(acc, ::AbstractFockHilbertSpace) = acc
_flat_non_fock_spaces(acc, P::ProductSpace) = (acc..., P.other_spaces...)
flat_fock_spaces(Hs) = foldl(_flat_fock_spaces, Hs; init=())
_flat_fock_spaces(acc, H::AbstractFockHilbertSpace) = (acc..., H)
_flat_fock_spaces(acc, ::AbstractHilbertSpace) = acc
_flat_fock_spaces(acc, P::ProductSpace) = (acc..., P.fock_space)
_flat_fock_spaces(acc, ::ProductSpace{Nothing}) = acc
flat_fock_states(states) = foldl(_flat_fock_states, states; init=())
_flat_fock_states(acc, s::AbstractFockState) = (acc..., s)
_flat_fock_states(acc, ::Any) = acc
_flat_fock_states(acc, s::ProductState) = (acc..., s.fock_state)
_flat_fock_states(acc, ::ProductState{Nothing}) = acc
flat_non_fock_states(states) = foldl(_flat_non_fock_states, states; init=())
_flat_non_fock_states(acc, ::AbstractFockState) = acc
_flat_non_fock_states(acc, s::Any) = (acc..., s)
_flat_non_fock_states(acc, s::ProductState) = (acc..., s.other_states...)


function StateExtender(Hs, H::ProductSpace{Nothing})
    ospaces = flat_non_fock_spaces(Hs)
    sublabels = map(keys, ospaces)
    labels = map(keys, H.other_spaces)
    perm = map(l -> findfirst(==(l), sublabels)::Int, labels)
    function extender(states)
        ostates = flat_non_fock_states(states)
        ProductState{Nothing}(TupleTools.permute(ostates, perm))
    end
end
function StateExtender(Hs, H::ProductSpace)
    fock_spaces = flat_fock_spaces(Hs)
    fockstateextender = length(fock_spaces) > 1 ? StateExtender(fock_spaces, H.fock_space) : only
    ospaces = flat_non_fock_spaces(Hs)
    sublabels = map(keys, ospaces)
    labels = map(keys, H.other_spaces)
    perm = map(l -> findfirst(==(l), sublabels)::Int, labels)
    function extender(states)
        fockstates = flat_fock_states(states)
        ostates = flat_non_fock_states(states)
        fstate = fockstateextender(fockstates)
        ProductState(fstate, TupleTools.permute(ostates, perm))
    end
end
fock_part(H::AbstractFockHilbertSpace) = H
fock_part(::AbstractHilbertSpace) = nothing
fock_part(P::ProductSpace) = P.fock_space
fock_part(P::ProductState) = P.fock_state
fock_part(::Any) = nothing
fock_part(s::AbstractFockState) = s
non_fock_part(::AbstractFockHilbertSpace) = nothing
non_fock_part(H::AbstractHilbertSpace) = H
non_fock_part(P::ProductSpace) = P.other_spaces
non_fock_part(P::ProductState) = P.other_states
non_fock_part(s::Any) = s
non_fock_part(::AbstractFockState) = nothing

phase_factor_h(f1, f2, Hs, H::ProductSpace{Nothing}) = 1
# phase_factor_h(f1, f2, Hs, H::AbstractHilbertSpace) = 1
phase_factor_h(f1, f2, Hs, H::ProductSpace) = phase_factor_h(f1, f2, Hs, H.fock_space)

phase_factor_h(f1::ProductState, f2::ProductState, partition, jw::JordanWignerOrdering) = phase_factor_h(f1.fock_state, f2.fock_state, partition, jw)
phase_factor_h(f1::ProductState, f2::AbstractFockState, partition, jw::JordanWignerOrdering) = phase_factor_h(f1.fock_state, f2, partition, jw)
phase_factor_h(f1::AbstractFockState, f2::ProductState, partition, jw::JordanWignerOrdering) = phase_factor_h(f1, f2.fock_state, partition, jw)

@testitem "GenericHilbertSpace, ProductSpace" begin
    using FermionicHilbertSpaces: GenericHilbertSpace
    using LinearAlgebra
    H1 = GenericHilbertSpace(:A, [:a, :b])
    H2 = GenericHilbertSpace(:B, [:c, :d])
    P = tensor_product(H1, H2)
    @test dim(P) == dim(H1) * dim(H2)
    @test 2 * I(2) == partial_trace(1.0 * I(4), P => H1)

    Hf = hilbert_space(1:2)
    P2 = tensor_product(Hf, H1, H2)
    @test dim(P2) / dim(H1) * I(2) == partial_trace(1.0 * I(dim(P2)), P2 => H1)

    P3 = tensor_product(Hf, P)
    @test P3 == P2
    @test_throws ArgumentError tensor_product(Hf, P2)
end


function Base.show(io::IO, H::Htype) where Htype<:AbstractFockHilbertSpace
    d = dim(H)
    N = length(keys(H))
    println(io, "$(d)-dimensional $(Htype.name.name):")
    print(io, "$N fermions: ")
    print(IOContext(io, :compact => true), modes(H))
end
Base.show(io::IO, ::MIME"text/plain", H::AbstractHilbertSpace) = show(io, H)

dim(H::AbstractHilbertSpace) = Int(length(basisstates(H)))
isorderedpartition(Hs, H::AbstractFockHilbertSpace) = isorderedpartition(map(modes, Hs), H.jw)
isorderedsubsystem(Hsub::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = isorderedsubsystem(Hsub.jw, H.jw)
isorderedsubsystem(Hsub::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = isorderedsubsystem(Hsub.jw, jw)
# issubsystem(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = issubsystem(subsystem.jw, jw)
# issubsystem(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = issubsystem(subsystem.jw, H.jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = consistent_ordering(subsystem.jw, jw)
consistent_ordering(subsystem::AbstractFockHilbertSpace, H::AbstractFockHilbertSpace) = consistent_ordering(subsystem.jw, H.jw)
focknbr_from_site_labels(H::AbstractFockHilbertSpace, jw::JordanWignerOrdering) = focknbr_from_site_labels(keys(H), jw)
ispartition(Hs, H::AbstractFockHilbertSpace) = ispartition(map(modes, Hs), H.jw)

mode_ordering(H::AbstractFockHilbertSpace) = H.jw
# mode_ordering(jw::JordanWignerOrdering) = jw
# mode_ordering(v::AbstractVector) = JordanWignerOrdering(v)
modes(H::AbstractFockHilbertSpace) = keys(H)
modes(jw::JordanWignerOrdering) = jw.ordering.keys
modes(v::AbstractVector) = v
embedding_unitary(partition, H::AbstractFockHilbertSpace) = embedding_unitary(partition, basisstates(H), H.jw)
bipartite_embedding_unitary(X, Xbar, H::AbstractFockHilbertSpace) = bipartite_embedding_unitary(X, Xbar, basisstates(H), H.jw)


"""
    SimpleFockHilbertSpace
A type representing a simple Fock Hilbert space with all fock states included.
"""
struct SimpleFockHilbertSpace{F,L} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    function SimpleFockHilbertSpace(labels, ::Type{F}=FockNumber{default_fock_representation(length(labels))}) where F
        jw = JordanWignerOrdering(labels)
        new{F,eltype(jw)}(jw)
    end
end
Base.keys(H::SimpleFockHilbertSpace) = keys(H.jw)
"""
    basisstates(H)
Return an iterator over all basis states for the given Hilbert space `H`.
"""
basisstates(H::SimpleFockHilbertSpace{F}) where F = Iterators.map(F ∘ FockNumber, UnitRange{UInt64}(0, 2^length(H.jw) - 1))
basisstate(ind, ::SimpleFockHilbertSpace{F}) where F = (F ∘ FockNumber)(ind - 1)
state_index(state::FockNumber, ::SimpleFockHilbertSpace) = state.f + 1
function Base.:(==)(H1::SimpleFockHilbertSpace, H2::SimpleFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    return true
end

"""
    FockHilbertSpace
A type representing a Fock Hilbert space with a given set of modes and Fock states.
"""
struct FockHilbertSpace{L,F,I} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    basisstates::F
    state_index::I
    function FockHilbertSpace(labels, basisstates::F=map(FockNumber, UnitRange{UInt64}(0, 2^length(labels) - 1))) where F
        jw = JordanWignerOrdering(labels)
        state_index = Dict(reverse(pair) for pair in enumerate(basisstates))
        new{eltype(jw),F,typeof(state_index)}(jw, basisstates, state_index)
    end
end
Base.keys(H::FockHilbertSpace) = keys(H.jw)
basisstates(H::FockHilbertSpace) = H.basisstates
basisstate(ind, H::FockHilbertSpace) = basisstates(H)[ind]
state_index(fockstate::AbstractFockState, H::FockHilbertSpace) = get(H.state_index, fockstate, missing)
function Base.:(==)(H1::FockHilbertSpace, H2::FockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.basisstates != H2.basisstates
        return false
    end
    if H1.state_index != H2.state_index
        return false
    end
    return true
end


"""
    SymmetricFockHilbertSpace
A type representing a Fock Hilbert space with fockstates organized by their quantum number.
"""
struct SymmetricFockHilbertSpace{L,S} <: AbstractFockHilbertSpace
    jw::JordanWignerOrdering{L}
    symmetry::S
end
function SymmetricFockHilbertSpace(labels, qn::AbstractSymmetry)
    SymmetricFockHilbertSpace(JordanWignerOrdering(labels), qn)
end
function SymmetricFockHilbertSpace(jw::JordanWignerOrdering, qn::AbstractSymmetry)
    labelled_symmetry, basisstates = instantiate_and_get_basisstates(jw, qn)
    sym_concrete = focksymmetry(basisstates, labelled_symmetry)
    SymmetricFockHilbertSpace{eltype(jw),typeof(sym_concrete)}(jw, sym_concrete)
end
SymmetricFockHilbertSpace(labels, qn::AbstractSymmetry, basisstates) = SymmetricFockHilbertSpace(JordanWignerOrdering(labels), qn, basisstates)
function SymmetricFockHilbertSpace(jw::JordanWignerOrdering, qn::AbstractSymmetry, basisstates)
    labelled_symmetry = instantiate(qn, jw)
    sym_concrete = focksymmetry(basisstates, labelled_symmetry)
    SymmetricFockHilbertSpace{eltype(jw),typeof(sym_concrete)}(jw, sym_concrete)
end


function Base.show(io::IO, H::SymmetricFockHilbertSpace)
    d = dim(H)
    N = length(keys(H))
    println(io, "$(d)-dimensional SymmetricFockHilbertSpace:")
    print(io, "$N fermions: ")
    println(IOContext(io, :compact => true), modes(H))
    show(io, H.symmetry)
end
Base.show(io::IO, sym::FockSymmetry) = print(io, sym.conserved_quantity)

Base.keys(H::SymmetricFockHilbertSpace) = keys(H.jw)
basisstate(ind, H::SymmetricFockHilbertSpace) = basisstate(ind, H.symmetry)
state_index(f::AbstractFockState, H::SymmetricFockHilbertSpace) = state_index(f, H.symmetry)
basisstates(H::SymmetricFockHilbertSpace) = basisstates(H.symmetry)
basisstates(H::SymmetricFockHilbertSpace{<:Any,NoSymmetry}) = Iterators.map(FockNumber, UnitRange{UInt64}(0, 2^length(H.jw) - 1))

function Base.:(==)(H1::SymmetricFockHilbertSpace, H2::SymmetricFockHilbertSpace)
    if H1 === H2
        return true
    end
    if H1.jw != H2.jw
        return false
    end
    if H1.symmetry != H2.symmetry
        return false
    end
    return true
end

"""
    hilbert_space(labels[, symmetry, basisstates])
Construct a Hilbert space from a set of labels, with optional symmetry and Fock number specification.
"""
hilbert_space(labels) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, basisstates) = FockHilbertSpace(labels, basisstates)
hilbert_space(labels, ::NoSymmetry) = SimpleFockHilbertSpace(labels)
hilbert_space(labels, ::NoSymmetry, basisstates) = FockHilbertSpace(labels, basisstates)
hilbert_space(labels, qn::AbstractSymmetry) = SymmetricFockHilbertSpace(labels, qn)
hilbert_space(labels, qn::AbstractSymmetry, basisstates) = SymmetricFockHilbertSpace(labels, qn, basisstates)

#= Tests for isorderedsubsystem, issubsystem, and consistent_ordering for Hilbert spaces =#
@testitem "Hilbert space subsystem and ordering" begin
    import FermionicHilbertSpaces: isorderedsubsystem, issubsystem, consistent_ordering
    # Simple Hilbert spaces
    H = hilbert_space([1, 2, 3])
    Hsub1 = hilbert_space([1, 2])
    Hsub2 = hilbert_space([2, 3])
    Hsub3 = hilbert_space([2, 1])
    Hsub4 = hilbert_space([3, 2])
    Hsub5 = hilbert_space([4])

    # issubsystem
    @test issubsystem(Hsub1, H)
    @test issubsystem(Hsub2, H)
    @test !issubsystem(Hsub5, H)
    @test issubsystem(Hsub1, H.jw)
    @test !issubsystem(Hsub5, H.jw)

    # consistent_ordering
    @test consistent_ordering(Hsub1, H)
    @test consistent_ordering(Hsub2, H.jw)
    @test !consistent_ordering(Hsub3, H)
    @test !consistent_ordering(Hsub4, H.jw)

    # isorderedsubsystem
    @test isorderedsubsystem(Hsub1, H)
    @test isorderedsubsystem(Hsub2, H.jw)
    @test !isorderedsubsystem(Hsub3, H)
    @test !isorderedsubsystem(Hsub4, H)
    @test !isorderedsubsystem(Hsub5, H)

    # ispartition and isorderedpartition
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    # Partition of Hilbert spaces (as required by isorderedpartition(Hs, H))
    H = hilbert_space([1, 2, 3])
    Hs1 = [hilbert_space([1]), hilbert_space([2, 3])]
    Hs2 = [hilbert_space([2]), hilbert_space([1, 3])]
    Hs3 = [hilbert_space([1, 2, 3])]
    Hs4 = [hilbert_space([1]), hilbert_space([2])]
    @test ispartition(Hs1, H)
    @test ispartition(map(keys, Hs2), H.jw)
    @test !ispartition(Hs4, H)
    @test isorderedpartition(Hs1, H)
    @test isorderedpartition(map(keys, Hs3), H.jw)
    @test !isorderedpartition(Hs4, H)
end

"""
    subregion(modes, H::AbstractHilbertSpace)

Return a subregion of the Hilbert space `H` that is spanned by the modes in `modes`. Only substates in `H` are included.
"""
function subregion(modes, H::SimpleFockHilbertSpace{F}) where F
    if !isorderedsubsystem(modes, H.jw)
        throw(ArgumentError("The modes $(modes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    SimpleFockHilbertSpace(modes, F)
end

function subregion(submodes, H::AbstractFockHilbertSpace)
    if !isorderedsubsystem(submodes, mode_ordering(H))
        throw(ArgumentError("The modes $(submodes) are not an ordered subsystem of the Hilbert space $(H)"))
    end
    states = substates(submodes, H)
    unique!(states)
    FockHilbertSpace(submodes, states)
end

function substates(modes, H::AbstractHilbertSpace)
    subsites = getindices(H, modes)
    substates = map(f -> substate(subsites, f), basisstates(H))
end
function substate(siteindices, f::FockNumber)
    subbits = Iterators.map(i -> _bit(f, i), siteindices)
    return focknbr_from_bits(subbits)
end

# complementary_subsystem(H::SimpleFockHilbertSpace, Hsub::AbstractFockHilbertSpace) = SimpleFockHilbertSpace(setdiff(collect(keys(H)), collect(keys(Hsub))))
# function complementary_subsystem(H::SingleParticleHilbertSpace, Hsub::SingleParticleHilbertSpace)
#     single_particle_hilbert_space(setdiff(collect(keys(H)), collect(keys(Hsub))))
# end
statetype(::SimpleFockHilbertSpace{F}) where F = F
statetype(::FockHilbertSpace{<:Any,<:V}) where V = eltype(V)
statetype(::SymmetricFockHilbertSpace{<:Any,S}) where S = statetype(S)
statetype(::FockSymmetry{V}) where V = eltype(v)
statetype(::Type{<:FockSymmetry{V}}) where V = eltype(V)
function simple_complementary_subsystem(H::AbstractFockHilbertSpace, Hsub::AbstractFockHilbertSpace)
    F = promote_type(statetype(H), statetype(Hsub))
    SimpleFockHilbertSpace(setdiff(collect(keys(H)), collect(keys(Hsub))), F)
end
function complementary_subsystem(H::AbstractFockHilbertSpace, Hsub::AbstractFockHilbertSpace)
    F = promote_type(statetype(H), statetype(Hsub))
    _Hbar = SimpleFockHilbertSpace(setdiff(collect(keys(H)), collect(keys(Hsub))), F)
    split = StateSplitter(H, (Hsub, _Hbar))
    states = unique!(map(basisstates(H)) do f
        fsub, fbar = split(f)
        fbar
    end)
    FockHilbertSpace(modes(_Hbar), states)
end
complementary_subsystem(H::AbstractFockHilbertSpace, ::Nothing) = H
complementary_subsystem(::Nothing, Hsub::AbstractHilbertSpace) = nothing

@testitem "subregion function" begin
    using FermionicHilbertSpaces
    # SimpleFockHilbertSpace
    H = hilbert_space([1, 2, 3])
    Hsub = subregion([1, 2], H)
    @test Hsub isa SimpleFockHilbertSpace
    @test keys(Hsub) == [1, 2]
    # FockHilbertSpace
    focks = [FockNumber(1), FockNumber(3)]
    HF = FockHilbertSpace([1, 2], focks)
    HFsub = subregion([1], HF)
    @test HFsub isa FockHilbertSpace
    @test keys(HFsub) == [1]
    basisstates(HFsub) == [FockNumber(1)]
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    HS = hilbert_space([1, 2], qn)
    HSsub = subregion([1], HS)
    @test HSsub isa FockHilbertSpace
    @test keys(HSsub) == [1]
    basisstates(HSsub) == [FockNumber(0), FockNumber(1)]
    # Error on non-subsystem
    @test_throws ArgumentError subregion([4], H)
end



function sector(m::AbstractMatrix, qn::Int, H::SymmetricFockHilbertSpace)
    ls = length.(H.symmetry.qntofockstates)
    startindex = 1
    for (n, l) in pairs(ls)
        if n == qn
            return m[startindex:startindex+l-1, startindex:startindex+l-1]
        end
        startindex += l
    end
    throw(ArgumentError("Sector $qn not found in the matrix."))
end

@testitem "Hilbert space printing" begin
    # Check that printing of Hilbert spaces doesn't error
    using FermionicHilbertSpaces
    io = IOBuffer()
    # SimpleFockHilbertSpace
    H_simple = hilbert_space(1:3)
    @test begin
        show(io, H_simple)
        true
    end
    # FockHilbertSpace
    focks = [FockNumber(0), FockNumber(1), FockNumber(2)]
    H_fock = FockHilbertSpace(1:2, focks)
    @test begin
        show(io, H_fock)
        true
    end
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    H_sym = hilbert_space(1:2, qn)
    @test begin
        show(io, H_sym)
        true
    end
end
