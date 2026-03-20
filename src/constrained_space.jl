
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
    # B = statetype(space)
    B = eltype(states)
    # B == eltype(S) || throw(ArgumentError("State type of constrained space must match underlying space, got $B and $(eltype(S))"))
    state_index = Dict{B,Int}(s => i for (i, s) in enumerate(states))
    ConstrainedSpace{B,H,S}(space, states, state_index)
end
dim(H::ConstrainedSpace) = length(H.states)
basisstates(H::ConstrainedSpace) = H.states
basisstate(n::Int, H::ConstrainedSpace) = H.states[n]
state_index(state, H::ConstrainedSpace) = get(H.state_index, state, missing)
atomic_factors(H::ConstrainedSpace) = atomic_factors(parent(H))
factors(H::ConstrainedSpace) = (parent(H),)
isconstrained(H::ConstrainedSpace) = true
cluster_target_subspace(target::ConstrainedSpace, args...) = cluster_target_subspace(parent(target), args...)
cluster_target_sub_idx(target::ConstrainedSpace, catoms, a2t, ti) = cluster_target_sub_idx(parent(target), catoms, a2t, ti)
combine_states(substates::Tuple, sp::ConstrainedSpace) = combine_states(substates, parent(sp))
partial_trace_phase_factor(s1, s2, sp::ConstrainedSpace) = partial_trace_phase_factor(s1, s2, parent(sp))

function constrain_space(H::ConstrainedSpace, constraint; kwargs...)
    space = constrain_space(parent(H), constraint; kwargs...)
    states = filter(state -> haskey(H.state_index, state), space.states)
    ConstrainedSpace(parent(H), states)
end


function Base.show(io::IO, H::ConstrainedSpace)
    println(io, "$(dim(H))-dimensional ConstrainedSpace:")
    print(io, "Underlying space: ")
    show(IOContext(io, :compact => true), parent(H))
end
constrain_space(H,::NoSymmetry) = H
function constrain_space(H, constraint; sortby=nothing, kwargs...)
    states = generate_states(H, constraint; kwargs...)
    isnothing(sortby) || sort!(states, by=sortby)
    ConstrainedSpace(H, states)
end
function constrain_space(H::AbstractHilbertSpace, states::AbstractVector{B}) where B<:AbstractBasisState
    ConstrainedSpace(H, states)
end

state_splitter(H::ConstrainedSpace, Hs) = state_splitter(parent(H), Hs)


mode_ordering(H::ConstrainedSpace) = mode_ordering(parent(H))
operators(H::ConstrainedSpace) = operators(parent(H))
@testitem "Constrained space" begin
    import FermionicHilbertSpaces: constrained_space, CombineFockNumbersProcessor, unweighted_number_branch_constraint, subregion, FermionicMode
    N = 5
    @fermions f
    Hs = [hilbert_space(a[n]) for n in 1:N]
    H = ProductSpace(Hs)
    m = matrix_representation(sym, H)
    @test dim(H) == 2^N
    @test size(m) == (2^N, 2^N)

    #total number of particles is 1
    constraint = unweighted_number_branch_constraint([1], Hs, Hs)
    combiner = CombineFockNumbersProcessor()
    Hc = constrained_space(H.modes, constraint; leaf_processor=combiner)
    @test dim(Hc) == N
    sym = sum(a[k]' * a[k] for k in 1:N)
    mc = matrix_representation(sym, Hc)
    @test size(mc) == (N, N)


    @test m == numberoperator(H)
    H2 = hilbert_space(1:N, NumberConservation(1))
    m2 = matrix_representation(sym, [f => H2])

    @fermions f
    sym = sum(f[k]' * f[k] for k in 1:N)
    @test_throws ArgumentError matrix_representation(sym, H)

    Nsub = 3
    Hsub = subregion(Hs[1:Nsub], H)
    @test dim(Hsub) == Nsub + 1

    embed(rand(2, 2), Hs[1] => H)
    msub = partial_trace(m, H => Hsub)
    embed(msub, Hsub => H)
end

