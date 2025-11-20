
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

basisstates(H::ProductSpace) = Iterators.map(s -> ProductState(s...), Iterators.product(basisstates(H.fock_space), Iterators.product(map(basisstates, H.other_spaces)...)))
basisstates(H::ProductSpace{Nothing}) = Iterators.map(ProductState{Nothing}, Iterators.product(map(basisstates, H.other_spaces)...))

function basisstate(n::Int, H::ProductSpace{Nothing})
    I = CartesianIndices(map(dim, H.other_spaces))[n]
    states = [basisstate(I[i], H.other_spaces[i]) for i in 1:length(H.other_spaces)]
    ProductState(nothing, states)
end
function basisstate(n::Int, H::ProductSpace)
    I = CartesianIndices((dim(H.fock_space), map(dim, H.other_spaces)...))[n]
    states = [basisstate(I[i+1], H.other_spaces[i]) for i in 1:length(H.other_spaces)]
    ProductState(basisstate(I[1], H.fock_space), states)
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
    spaces = filter(Hn -> Hn != Hsub, H.other_spaces)
    length(spaces) + 1 == length(H.other_spaces) || throw(ArgumentError("Hsub must be one of the spaces in the product space H"))
    ProductSpace(H.fock_space, spaces)
end
function complementary_subsystem(H::ProductSpace, Hsub::AbstractFockHilbertSpace)
    fcomp = complementary_subsystem(H.fock_space, Hsub)
    ProductSpace(fcomp, H.other_spaces)
end
complementary_subsystem(::Nothing, ::Nothing) = nothing
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

_num_non_fock_spaces(H::AbstractHilbertSpace) = 1
_num_non_fock_spaces(::AbstractFockHilbertSpace) = 0
_num_non_fock_spaces(P::ProductSpace) = length(P.other_spaces)
_substate(s::AbstractBasisState, n::Int) = n == 1 ? s : error("Substate index out of bounds")
_substate(s::Any, n::Int) = n == 1 ? s : error("Substate index out of bounds")
_substate(s::ProductState, n::Int) = s.other_states[n]

function StateExtender(Hs, H::ProductSpace{Nothing})
    ospaces = flat_non_fock_spaces(Hs)
    sublabels = map(keys, ospaces)
    labels = map(keys, H.other_spaces)
    perm = (map(l -> findfirst(==(l), sublabels)::Int, labels))
    spacelengths = map(_num_non_fock_spaces, Hs)
    cumulative_spacelengths = (0, cumsum(spacelengths)...)
    accessors = map(perm) do n
        which_space = findfirst(i -> cumulative_spacelengths[i] < n <= cumulative_spacelengths[i+1], 1:length(Hs))
        which_subspace = n - cumulative_spacelengths[which_space]
        states -> _substate(states[which_space], which_subspace)
    end
    function extender(states)
        ostates = map(acc -> acc(states), accessors)
        ProductState{Nothing}(ostates)
    end
end
function StateExtender(Hs, H::ProductSpace)
    fock_spaces = flat_fock_spaces(Hs)
    fockstateextender = length(fock_spaces) > 1 ? StateExtender(fock_spaces, H.fock_space) : only
    ospaces = flat_non_fock_spaces(Hs)
    non_fock_space = ProductSpace{Nothing}(H.other_spaces)
    non_fock_extender = StateExtender(ospaces, non_fock_space)
    function extender(states)
        fockstates = flat_fock_states(states)
        ostates = flat_non_fock_states(states)
        eostates = non_fock_extender(ostates).other_states
        fstate = fockstateextender(fockstates)
        ProductState(fstate, eostates) 
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
