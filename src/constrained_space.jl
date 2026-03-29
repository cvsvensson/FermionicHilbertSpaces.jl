
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
# factors(H::ConstrainedSpace) = (parent(H),)
isconstrained(H::ConstrainedSpace) = true
combine_states(substates, sp::ConstrainedSpace) = combine_states(substates, parent(sp))
partial_trace_phase_factor(s1, s2, sp::ConstrainedSpace) = partial_trace_phase_factor(s1, s2, parent(sp))
atomic_substate(n, f, space::ConstrainedSpace) = atomic_substate(n, f, parent(space))
constrain_space(space::AbstractHilbertSpace, constraint::NoSymmetry) = space
constrain_space(space::AbstractHilbertSpace, states::AbstractVector{B}) where B<:AbstractBasisState = ConstrainedSpace(space, states)

function constrain_space(space, constraint::AbstractConstraint)
    allowed = in(Set(allowed_values(constraint, space)))
    qn_func = sector_function(constraint, space, missing)
    qn(state) = begin
        n = qn_func(state)
        allowed(n) ? n : missing
    end
    block_space(space, basisstates(space), qn)
end
function constrain_space(space, constraint::ProductConstraint)
    alloweds = map(c -> in(Set(allowed_values(c, space))), constraint.constraints)
    qn_funcs = map(c -> sector_function(c, space, missing), constraint.constraints)
    qn(state) = begin
        qns = map(qn_func -> qn_func(state), qn_funcs)
        all(aq -> aq[1](aq[2]), zip(alloweds, qns)) ? qns : missing
    end
    block_space(space, basisstates(space), qn)
end

allowed_values(::NumberConservation{Missing}, space) = 0:maximum_particles(space)
allowed_values(constraint::NumberConservation{T}, space) where T = constraint.total
allowed_values(p::ParityConservation, space) = p.allowed_parities

@testitem "constrain_space" begin
    @fermions f
    H = hilbert_space(f, 1:2)
    @test dim(constrain_space(H, NumberConservation())) == 4
    @test dim(constrain_space(H, NumberConservation(1))) == 2
    @test dim(constrain_space(H, NumberConservation(0:1))) == 3

    @test dim(constrain_space(H, ParityConservation())) == 4
    @test dim(constrain_space(H, ParityConservation(1))) == 2

    H1 = constrain_space(H, NumberConservation())
    H2 = constrain_space(H1, NumberConservation(0:1))
    H3 = constrain_space(H2, NumberConservation(1:2))
    @test basisstates(H3) == basisstates(constrain_space(H, NumberConservation(1)))

    #repeat for bosons
    @bosons b 1:2
    Hs = hilbert_space.(values(b), 2)
    H = tensor_product(Hs...)
    @test dim(constrain_space(H, NumberConservation())) == 4
    @test dim(constrain_space(H, NumberConservation(1))) == 2
    @test dim(constrain_space(H, NumberConservation(0:1))) == 3

    @test dim(constrain_space(H, ParityConservation())) == 4
    @test dim(constrain_space(H, ParityConservation(1))) == 2

    H1 = constrain_space(H, NumberConservation())
    H2 = constrain_space(H1, NumberConservation(0:1))
    H3 = constrain_space(H2, NumberConservation(1:2))
    @test basisstates(H3) == basisstates(constrain_space(H, NumberConservation(1)))
end

state_mapper(H::ConstrainedSpace, Hs) = state_mapper(parent(H), Hs)
mode_ordering(H::ConstrainedSpace) = mode_ordering(parent(H))

apply_local_operators(ops::Vector{<:NCMul}, state::ProductState, space::ConstrainedSpace, precomp) = apply_local_operators(ops, state, space.parent, precomp)
_precomputation_before_operator_application(ops::Union{<:Any,<:NCMul}, space::ConstrainedSpace) = _precomputation_before_operator_application(ops, parent(space))

@testitem "Constrained space" begin
    import FermionicHilbertSpaces: constrain_space, CombineFockNumbersProcessor, subregion
    N = 5
    @fermions f
    H = hilbert_space(f, 1:N)
    sym = sum(f[k]' * f[k] for k in 1:N)
    m = matrix_representation(sym, H)
    @test dim(H) == 2^N
    @test size(m) == (2^N, 2^N)

    #total number of particles is 1
    Hc = constrain_space(H, NumberConservation(1))
    Hc2 = tensor_product(H.modes, NumberConservation(1))
    @test Set(basisstates(Hc)) == Set(basisstates(Hc2))
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

apply_local_operators(ops::Vector{<:NCMul}, state, space::BlockHilbertSpace, precomp) = apply_local_operators(ops, state, space.parent, precomp)
