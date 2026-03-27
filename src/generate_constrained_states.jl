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
function generate_states(spaces, constraint, full_space::AbstractHilbertSpace{B}; kwargs...) where B
    mapper = state_mapper(full_space, spaces)
    process_result = (state, spaces) -> only(first(combine_states(state, mapper)))
    valid_final_state = if isconstrained(full_space)
        state -> !ismissing(state_index(state, full_space)) # This might be redundant, if full_space is always constructed to only include valid states
    else
        state -> true
    end
    generate_states(spaces, constraint, B; process_result, valid_final_state, kwargs...)
end
default_process_result(state, spaces) = copy(state)
function generate_states(spaces, _constraint, ::Type{B}=Any; partial_processor=nothing, process_result=default_process_result, valid_final_state=state -> true) where B
    constraint = branch_constraint(_constraint, spaces)
    all_statetypes = statetype.(spaces)
    partial = Vector{Union{all_statetypes...}}(undef, length(spaces))
    results = B[]
    backtrack!(results, partial, spaces, 1, constraint, partial_processor, process_result, valid_final_state)
    return results
end

function backtrack!(results, partial, spaces, depth, constraint, partial_processor, process_result, valid_final_state)
    n = length(spaces)
    if depth > n
        # All spaces assigned, add to results
        state = process_result(partial, spaces)
        if valid_final_state(state)
            push!(results, state)
        end
        return
    end

    for state in basisstates(spaces[depth])
        partial[depth] = state
        # Check if this branch is worth exploring
        if valid_branch(constraint, partial, depth, spaces)
            process_partial(partial_processor, partial, depth, spaces)
            backtrack!(results, partial, spaces, depth + 1, constraint, partial_processor, process_result, valid_final_state)
        end
    end
end

_normalize_constraint_values(values::AbstractVector) = collect(values)
_normalize_constraint_values(values::Tuple) = collect(values)
_normalize_constraint_values(values::AbstractRange) = collect(values)
_normalize_constraint_values(value) = [value]

# _normalize_constraint_functions(f::Function, n) = ntuple(n -> f, n)
# _normalize_constraint_functions(functions, n) = functions
struct WeightedFunction{F,W}
    func::F
    weights::W
end
function _contribution_values(subspaces, functions)
    map(__contribution_values, subspaces, functions)
end
function _contribution_values(subspaces, func::Function)
    map(Base.Fix2(__contribution_values, func), subspaces)
end
function _contribution_values(subspaces, fw::WeightedFunction)
    map((space, weight) -> __contribution_values(space, s -> weight * fw.func(s)), subspaces, fw.weights)
end
function __contribution_values(space, func)
    values = unique(Iterators.map(func, basisstates(space)))
    isempty(values) && throw(ArgumentError("Cannot build AdditiveConstraint from a space with no basis states"))
    values
end

additive_branch_constraint(allowed_sums, functions, allspaces) = additive_branch_constraint(allowed_sums, functions, missing, allspaces)
function additive_branch_constraint(allowed_sums, functions, _subspaces, allspaces)
    subspaces = ismissing(_subspaces) ? allspaces : _subspaces
    allowed_values = _normalize_constraint_values(allowed_sums)
    _additive_branch_constraint(allowed_values, functions, subspaces, allspaces)
end
function _additive_branch_constraint(allowed_values, functions, subspaces, allspaces)
    positions = map(subspaces) do subspace
        pos = findfirst(isequal(subspace), allspaces)
        isnothing(pos) && throw(ArgumentError("All AdditiveConstraint subspaces must be present in the generated space"))
        pos
    end
    contribution_values = _contribution_values(subspaces, functions)
    T = typeof(sum(first, contribution_values))
    # T = mapreduce(eltype, promote_type, contribution_values; init=eltype(allowed_values))
    n = length(allspaces)
    contribution_min = zeros(T, n)
    contribution_max = zeros(T, n)
    for (pos, vals) in zip(positions, contribution_values)
        contribution_min[pos] += minimum(vals)
        contribution_max[pos] += maximum(vals)
    end

    remaining_min = zeros(T, n + 1)
    remaining_max = zeros(T, n + 1)
    for i in n:-1:1
        remaining_min[i] = remaining_min[i+1] + contribution_min[i]
        remaining_max[i] = remaining_max[i+1] + contribution_max[i]
    end
    BranchConstraint((partial, depth, _) -> begin
        current = _additive_function_application(partial, positions, functions, depth, T)
        # current = sum(f(partial[pos]) for (pos, f) in zip(positions, functions) if pos <= depth; init=zero(T))
        min_possible = current + remaining_min[depth+1]
        max_possible = current + remaining_max[depth+1]
        any(t -> min_possible <= t <= max_possible, allowed_values)
    end)
end
function _additive_function_application(substates, functions, ::Type{T}=Int) where T
    sum(f(substate) for (substate, f) in zip(substates, functions); init=zero(T))
end
function _additive_function_application(partial, positions, functions, depth, ::Type{T}=Int) where T
    substates = (partial[pos] for pos in positions if pos <= depth)
    _additive_function_application(substates, functions, T)
end
_additive_function_application(substates, func::Function, ::Type{T}=Int) where T = sum(func, substates; init=zero(T))
_additive_function_application(substates, fw::WeightedFunction, ::Type{T}=Int) where T = sum(w * fw.func(substate) for (substate, w) in zip(substates, fw.weights); init=zero(T))

branch_constraint(constraint::AdditiveConstraint, spaces) = additive_branch_constraint(constraint.allowed_values, constraint.functions, constraint.subspaces, spaces)


@testitem "generate_states with BranchConstraint" begin
    using FermionicHilbertSpaces: generate_states, BranchConstraint, AdditiveConstraint, basisstate, hilbert_space, _bit, CombineFockNumbersProcessor, constrain_space, quantumnumbers, particle_number

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
    c1 = NumberConservation(allowed_ones[1], [Hs[2], Hs[4]])
    c2 = NumberConservation(allowed_ones[2], [Hs[1], Hs[3]])
    constraint = c1 * c2
    states = generate_states(Hs, constraint)
    # verify particle numbers 
    for state in states
        @test count_ones(state[2].f) + count_ones(state[4].f) in allowed_ones[1]
        @test count_ones(state[1].f) + count_ones(state[3].f) in allowed_ones[2]
    end
    states = generate_states(Hs, constraint; process_result) #Now for states combined into focknumbers
    for state in states
        @test count_ones(state.f & masks[1]) in allowed_ones[1]
        @test count_ones(state.f & masks[2]) in allowed_ones[2]
    end

    cons2 = NumberConservation(allowed_ones[1], [Hs[2], Hs[4]]) * NumberConservation(allowed_ones[2], [Hs[1], Hs[3]])
    @test states == generate_states(Hs, cons2; process_result)

    @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed
        c1 = NumberConservation(allowed[1], [Hs[2], Hs[4]])
        c2 = NumberConservation(allowed[2], [Hs[1], Hs[3]])
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
    constraint = prod(NumberConservation(allowed) for allowed in allowed_ones)
    states = generate_states(H.modes, constraint, H)
    @test all(s -> s.f >= 0, states)

    weights = [Int.(floor.(2sin.(1:N))), Int.(sign.((1:N) .- div(N, 2))), ones(Int, N)]
    constraint = prod(NumberConservation(allowed, missing, w) for (allowed, w) in zip(allowed_ones, weights))
    states = generate_states(H.modes, constraint, H)
    @test all(s -> s.f >= 0, states)

    H = tensor_product(Hs)
    additive = AdditiveConstraint([1], (Hs[1], Hs[2], Hs[3]), (
        s -> 2 * particle_number(s),
        s -> -particle_number(s),
        s -> particle_number(s),
    ))
    numcon = NumberConservation([1], (Hs[1], Hs[2], Hs[3]), (2, -1, 1))
    Hnum = tensor_product(Hs, numcon)
    Hadd = tensor_product(Hs, additive)
    @test Hnum == Hadd
    # states = generate_states(Hs, additive, H; process_result)
    @test all(state -> begin
            n1 = count_ones(state.f & 0b00001)
            n2 = count_ones(state.f & 0b00010)
            n3 = count_ones(state.f & 0b00100)
            2n1 - n2 + n3 == 1
        end, basisstates(Hadd))

    additive_with_shared_function = AdditiveConstraint([0, 2], (Hs[4], Hs[5]), particle_number)
    states = generate_states(Hs, additive_with_shared_function; process_result)
    @test all(state -> begin
            n4 = count_ones(state.f & 0b01000)
            n5 = count_ones(state.f & 0b10000)
            n4 + n5 in (0, 2)
        end, states)

    Hc = tensor_product(Hs, additive)
    @test collect(quantumnumbers(Hc)) == [1]

    compound = additive * AdditiveConstraint([1], (Hs[4], Hs[5]), particle_number)
    states = generate_states(Hs, compound; process_result)
    @test all(state -> begin
            n1 = count_ones(state.f & 0b00001)
            n2 = count_ones(state.f & 0b00010)
            n3 = count_ones(state.f & 0b00100)
            n4 = count_ones(state.f & 0b01000)
            n5 = count_ones(state.f & 0b10000)
            2n1 - n2 + n3 == 1 && n4 + n5 == 1
        end, states)

    # Test with clusters and mixed spaces
    H1 = hilbert_space(f, 1:2)
    H2 = hilbert_space(f, 3:3)
    @boson b
    Hb = hilbert_space(b, 3)
    H = tensor_product((H1, H2, Hb))
    mapper = state_mapper(H, (H1, H2, Hb))
    constraint = NumberConservation(1)
    _states = generate_states((H1, H2, Hb), constraint)
    states = map(s -> only(first(combine_states(s, mapper))), _states)
    states2 = basisstates(constrain_space(H, constraint))
    @test Set(states) == Set(states2)
end