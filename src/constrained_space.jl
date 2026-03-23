
struct ConstrainedSpace{B,H,S} <: AbstractHilbertSpace{B}
    parent::H
    states::S
    state_index::Dict{B,Int}
end
_find_position(Hsub::AbstractHilbertSpace, H::ConstrainedSpace) = _find_position(Hsub, H.parent)
Base.:(==)(H1::ConstrainedSpace, H2::ConstrainedSpace) = H1.parent == H2.parent && H1.states == H2.states && H1.state_index == H2.state_index
Base.hash(H::ConstrainedSpace, h::UInt) = hash(H.parent, hash(H.states, hash(H.state_index, h)))
Base.parent(H::ConstrainedSpace) = H.parent
function ConstrainedSpace(space::ConstrainedSpace, states)
    intersect(states, space.states)
    ConstrainedSpace(parent(space), states)
end

function ConstrainedSpace(space::H, states::S) where {H,S}
    B = eltype(states)
    state_index = Dict{B,Int}(s => i for (i, s) in enumerate(states))
    ConstrainedSpace{B,H,S}(space, states, state_index)
end
dim(H::ConstrainedSpace) = length(H.states)
basisstates(H::ConstrainedSpace) = H.states
basisstate(n::Int, H::ConstrainedSpace) = H.states[n]
state_index(state, H::ConstrainedSpace) = get(H.state_index, state, missing)
atomic_factors(H::ConstrainedSpace) = atomic_factors(parent(H))
clusters(H::ConstrainedSpace) = clusters(parent(H))
factors(H::ConstrainedSpace) = (parent(H),)
isconstrained(H::ConstrainedSpace) = true
combine_states(substates, sp::ConstrainedSpace) = combine_states(substates, parent(sp))
partial_trace_phase_factor(s1, s2, sp::ConstrainedSpace) = partial_trace_phase_factor(s1, s2, parent(sp))

function constrain_space(H::ConstrainedSpace, constraint::AbstractConstraint; kwargs...)
    space = constrain_space(parent(H), constraint; kwargs...)
    states = filter(state -> haskey(H.state_index, state), basisstates(space))
    ConstrainedSpace(parent(H), states)
end


function Base.show(io::IO, H::ConstrainedSpace)
    if get(io, :compact, false)
        print(io, "ConstrainedSpace(")
        show(IOContext(io, :compact => true), parent(H))
        print(io, ", $(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional ConstrainedSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
    end
end
constrain_space(H, ::NoSymmetry; kwargs...) = H

function constrain_space(H::AbstractHilbertSpace, states::AbstractVector{B}) where B<:AbstractBasisState
    ConstrainedSpace(H, states)
end

state_splitter(H::ConstrainedSpace, Hs) = state_splitter(parent(H), Hs)
default_sorter(H::ConstrainedSpace, constraint) = default_sorter(parent(H), constraint)
default_processor(H::ConstrainedSpace, constraint) = default_processor(parent(H), constraint)

mode_ordering(H::ConstrainedSpace) = mode_ordering(parent(H))

apply_local_operators(ops::Vector{<:NCMul}, state::ProductState, space::ConstrainedSpace, precomp) = apply_local_operators(ops, state, space.parent, precomp)


@testitem "Constrained space" begin
    import FermionicHilbertSpaces: constrain_space, CombineFockNumbersProcessor, unweighted_number_branch_constraint, subregion, FermionicMode
    N = 5
    @fermions f
    H = hilbert_space(f, 1:N)
    sym = sum(f[k]' * f[k] for k in 1:N)
    m = matrix_representation(sym, H)
    @test dim(H) == 2^N
    @test size(m) == (2^N, 2^N)

    #total number of particles is 1
    constraint = unweighted_number_branch_constraint([1], H.modes, H.modes)
    combiner = CombineFockNumbersProcessor{FockNumber{Int}}()
    Hc = constrain_space(H, constraint; leaf_processor=combiner)
    @test dim(Hc) == N
    mc = matrix_representation(sym, Hc)
    @test size(mc) == (N, N)

    @test m == numberoperator(H)

    H2 = hilbert_space(f, 1:N, NumberConservation(1))
    @test numberoperator(H2) == matrix_representation(sym, H2)

    Nsub = 3
    Hsub = subregion(H.modes[1:Nsub], H2)
    @test dim(Hsub) == Nsub + 1
end

