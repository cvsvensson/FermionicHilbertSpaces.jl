abstract type AbstractBranchConstraint <: AbstractConstraint end
struct BranchConstraint{F} <: AbstractBranchConstraint
    f::F
end
branch_constraint(constraint::BranchConstraint, space) = constraint
has_sectors(::AbstractConstraint) = false
"""
    valid_branch(constraint, partial_state, remaining_spaces) -> Bool
    
    Return `true` if the branch should be explored, `false` to prune. By default this calls `constraint.f(partial_state, remaining_spaces)`.
"""
valid_branch(constraint::BranchConstraint, partial_state, depth, spaces) = constraint.f(partial_state, depth, spaces)
valid_branch(constraint::ProductConstraint, partial_state, depth, spaces) = all(c -> valid_branch(c, partial_state, depth, spaces), constraint.constraints)

process_partial(::Nothing, partial_state, depth, spaces) = nothing
process_partial(processor, partial_state, depth, spaces) = processor(partial_state, depth, spaces)

"""
    generate_states(spaces, constraints; partial_processor=nothing, process_result=(state, space) -> state)

Generate all tensor product states from `spaces` satisfying `constraint`.
Uses backtracking with pruning via `valid_branch`.

`partial_processor(partial_state, depth, spaces)` is called whenever a branch is accepted.
`process_result(full_state, spaces)` can transform each completed state before storing it.
"""
function generate_states(space::AbstractHilbertSpace{B}, constraint; kwargs...) where B
    mapper = state_mapper(space, atomic_factors(space))
    process_result = (full_state, spaces) -> only(first(combine_states(full_state, mapper)))
    generate_states(atomic_factors(space), constraint, B; process_result, kwargs...)
end
function generate_states(spaces, _constraint, ::Type{B}=Any; partial_processor=nothing, process_result=(state, spaces) -> copy(state)) where B
    constraint = branch_constraint(_constraint, spaces)
    all_statetypes = statetype.(spaces)
    partial = Vector{Union{all_statetypes...}}(undef, length(spaces))
    results = B[]
    backtrack!(results, partial, spaces, 1, constraint, partial_processor, process_result)
    return results
end

function backtrack!(results, partial, spaces, depth, constraint, partial_processor, process_result)
    n = length(spaces)
    if depth > n
        # All spaces assigned, add to results
        push!(results, process_result(partial, spaces))
        return
    end

    for state in basisstates(spaces[depth])
        partial[depth] = state
        # Check if this branch is worth exploring
        if valid_branch(constraint, partial, depth, spaces)
            process_partial(partial_processor, partial, depth, spaces)
            backtrack!(results, partial, spaces, depth + 1, constraint, partial_processor, process_result)
        end
    end
end

function catenate_fock_states(full_state, spaces, T)
    num = zero(T)
    shift = 0
    for (state, space) in zip(full_state, spaces)
        num |= state << shift
        shift += nbr_of_modes(space)
    end
    num
end
unweighted_number_branch_constraint(allowed_sums, allspaces) = unweighted_number_branch_constraint(allowed_sums, missing, allspaces)
function unweighted_number_branch_constraint(allowed_sums, _subspaces, _allspaces)
    allspaces = _allspaces isa AbstractHilbertSpace ? atomic_factors(_allspaces) : collect(Iterators.flatten(Iterators.map(atomic_factors, _allspaces)))
    subspaces = ismissing(_subspaces) ? allspaces : collect(Iterators.flatten(Iterators.map(atomic_factors, _subspaces)))
    return _unweighted_number_branch_constraint(allowed_sums, subspaces, allspaces)
end
function _unweighted_number_branch_constraint(allowed_numbers, subspaces, allspaces)
    issub = BitVector(map(s -> s in subspaces, allspaces))
    remaining_max_particles = Int[]
    for depth in 0:length(allspaces)
        remaining_spaces = allspaces[depth+1:end][issub[depth+1:end]]
        max_particles = sum(maximum_particles, remaining_spaces, init=0)
        push!(remaining_max_particles, max_particles)
    end
    BranchConstraint((partial, depth, spaces) -> begin
        current = sum(n -> issub[n] ? particle_number(partial[n]) : 0, 1:depth)
        remaining = remaining_max_particles[depth+1]
        feasible = any(allowed -> current <= allowed <= current + remaining, allowed_numbers)
        return feasible
    end)
end

weighted_number_branch_constraint(allowed_sums, weights, allspaces) = weighted_number_branch_constraint(allowed_sums, weights, missing, allspaces)
function weighted_number_branch_constraint(allowed_sums, _weights, _subspaces, _allspaces)
    allspaces = _allspaces isa AbstractHilbertSpace ? atomic_factors(_allspaces) : collect(Iterators.flatten(Iterators.map(atomic_factors, _allspaces)))
    subspaces = ismissing(_subspaces) ? allspaces : collect(Iterators.flatten(Iterators.map(atomic_factors, _subspaces)))
    ismissing(_weights) && return unweighted_number_branch_constraint(allowed_sums, subspaces, allspaces)
    return _weighted_number_branch_constraint(allowed_sums, _weights, subspaces, allspaces)
end
function _weighted_number_branch_constraint(allowed_sums, _weights, subspaces, allspaces)
    issub = BitVector(map(s -> s in subspaces, allspaces))
    #extend weights to all spaces, filling non-subspaces with zeros
    weights = zeros(eltype(_weights), length(allspaces))
    weights[issub] .= _weights
    n = length(allspaces)
    length(weights) == n || error("weights must have same length as allspaces")

    # Precompute suffix min/max of weighted particle contributions
    remaining_min = zeros(eltype(weights), n + 1)
    remaining_max = zeros(eltype(weights), n + 1)
    for i in n:-1:1
        w = weights[i]
        if w != 0
            maxp = maximum_particles(allspaces[i])
            remaining_min[i] = remaining_min[i+1] + min(zero(w), w * maxp)
            remaining_max[i] = remaining_max[i+1] + max(zero(w), w * maxp)
        else
            remaining_min[i] = remaining_min[i+1]
            remaining_max[i] = remaining_max[i+1]
        end
    end

    BranchConstraint((partial, depth, spaces) -> begin
        current = zero(eltype(weights))
        for i in 1:depth
            weights[i] != 0 && (current += weights[i] * particle_number(partial[i]))
        end
        min_possible = current + remaining_min[depth+1]
        max_possible = current + remaining_max[depth+1]
        return any(target -> min_possible <= target <= max_possible, allowed_sums)
    end)
end


@testitem "generate_states with BranchConstraint" begin
    using FermionicHilbertSpaces: generate_states, BranchConstraint, basisstate, hilbert_space, _bit, unweighted_number_branch_constraint, weighted_number_branch_constraint, CombineFockNumbersProcessor

    # Define a simple constraint: only allow states where the first space is in its first basis state
    @fermions f
    H1 = hilbert_space(f, 1:1)  # Basis states: |0>, |1>
    H2 = hilbert_space(f, 2:2)  # Basis states: |0>, |1>
    Hs = (H1, H2)

    constraint = BranchConstraint((partial, depth, spaces) -> partial[1] == basisstate(1, H1))
    states = generate_states((H1, H2), constraint)

    # Should only get states where the first space is in its first basis state (|0>)
    expected = [[basisstate(1, H1), basisstate(1, H2)], [basisstate(1, H1), basisstate(2, H2)]]
    @test sort(states) == sort(expected)

    # Partial and full processors are both applied
    visited_depths = Int[]
    process_result = CombineFockNumbersProcessor{FockNumber{Int}}()
    states_as_int = map(f -> f.f, generate_states(
        Hs,
        BranchConstraint((partial, depth, spaces) -> true);
        partial_processor=(partial, depth, spaces) -> push!(visited_depths, depth),
        process_result))
    @test count(==(1), visited_depths) == 2
    @test count(==(2), visited_depths) == 4
    @test sort(states_as_int) == [0x00, 0x01, 0x02, 0x03]


    masks = [0b1010, 0b0101]
    allowed_ones = [[0], [2]]
    Hs = [hilbert_space(f, n:n) for n in 1:5]
    c1 = unweighted_number_branch_constraint(allowed_ones[1], [Hs[2], Hs[4]], Hs)
    c2 = unweighted_number_branch_constraint(allowed_ones[2], [Hs[1], Hs[3]], Hs)
    constraint = c1 * c2
    states = generate_states(Hs, constraint; process_result)
    # verify particle numbers 
    for state in states
        @test count_ones(state.f & masks[1]) in allowed_ones[1]
        @test count_ones(state.f & masks[2]) in allowed_ones[2]
    end

    cons2 = NumberConservation(allowed_ones[1], [Hs[2], Hs[4]]) * NumberConservation(allowed_ones[2], [Hs[1], Hs[3]])
    @test states == generate_states(Hs, cons2; process_result)

    @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed
        c1 = unweighted_number_branch_constraint(allowed[1], [Hs[2], Hs[4]], Hs)
        c2 = unweighted_number_branch_constraint(allowed[2], [Hs[1], Hs[3]], Hs)
        constraint = c1 * c2
        states = generate_states(Hs, constraint; process_result)
        for state in states
            count_ones(state.f & masks[1]) in allowed[1] || return false
            count_ones(state.f & masks[2]) in allowed[2] || return false
        end
        true
    end


    # Test that large number of modes works
    N = 80
    H = hilbert_space(f, 1:N)
    allowed_ones = [[0, 1], [-1, 0], [2]]
    constraint = prod(unweighted_number_branch_constraint(allowed, H) for allowed in allowed_ones)
    states = generate_states(H, constraint)
    @test all(s -> s.f >= 0, states)

    weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
    constraint = prod(weighted_number_branch_constraint(allowed, w, H) for (allowed, w) in zip(allowed_ones, weights))
    states = generate_states(H, constraint)
    @test all(s -> s.f >= 0, states)

end