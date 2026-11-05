abstract type AbstractBranchConstraint <: AbstractConstraint end
"""
    BranchConstraint(f)

Constraint for pruning branches while generating basis states.
`f(partial_state, depth, spaces) -> Bool` is evaluated during
backtracking by [`generate_states`](@ref), after the state at position `depth`
has been assigned. Entries after `depth` in `partial_state` have not yet been
assigned.

Returning `true` continues exploring the branch, while returning `false`
prunes it. The callback should therefore accept partial states and reject a
branch only when no valid completion can satisfy the constraint.

`BranchConstraint` is intended for use with `tensor_product` or more directly with `generate_states`.
"""
struct BranchConstraint{F} <: AbstractBranchConstraint
    f::F
end
supports_branch_pruning(::BranchConstraint) = true
supports_filtering(::BranchConstraint) = false
supports_sector_grouping(::BranchConstraint) = false
supply_missing_constraint_info(constraint::BranchConstraint, space, spaces) = constraint
branch_constraint(constraint::BranchConstraint, space) = constraint
"""
    valid_branch(constraint, partial_state, depth, spaces) -> Bool

Return `true` if the branch should be explored, `false` to prune.
For a `BranchConstraint` this calls `constraint.f(partial_state, depth, spaces)`.
"""
valid_branch(constraint::BranchConstraint, partial_state, depth, spaces) = constraint.f(partial_state, depth, spaces)
valid_branch(constraint::ProductConstraint, partial_state, depth, spaces) = all(Iterators.map(c -> valid_branch(c, partial_state, depth, spaces), constraint.constraints))

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
        state -> !iszero(state_index(state, full_space)) # This might be redundant, if full_space is always constructed to only include valid states
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

function filter_function(constraint::BranchConstraint, space::AbstractHilbertSpace)
    state -> true
end

_normalize_constraint_values(values::AbstractVector) = values
_normalize_constraint_values(values::Tuple) = values
_normalize_constraint_values(values::AbstractRange) = values
_normalize_constraint_values(value) = [value]
_normalize_constraint_values(::Missing) = missing


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
    isempty(values) && throw(ArgumentError("Cannot build an `AdditiveConstraint` from a space with no basis states."))
    values
end
hilbert_space(space::AbstractHilbertSpace) = space
additive_branch_constraint(allowed_sums, functions, allspaces) = additive_branch_constraint(allowed_sums, functions, missing, allspaces)
function additive_branch_constraint(allowed_sums, functions, _subspaces, allspaces)
    allowed_values = _normalize_constraint_values(allowed_sums)
    subspaces = map(hilbert_space, ismissing(_subspaces) ? allspaces : _subspaces)
    positions = ismissing(_subspaces) ? (1:length(allspaces)) : _subspace_positions(subspaces, allspaces)
    _additive_branch_constraint(allowed_values, functions, subspaces, positions, length(allspaces))
end
function _subspace_positions(subspaces, allspaces)
    space_ids = map(collect ∘ atom_ids, allspaces) # collect, since spaces may be given as symbols
    return map(subspaces) do subspace
        pos = findfirst(==(collect(atom_ids(subspace))), space_ids)
        isnothing(pos) && throw(ArgumentError("All subspaces of an `AdditiveConstraint` must be present in the generated space, but `$subspace` is not."))
        pos
    end
end
function _additive_branch_constraint(allowed_values, functions, subspaces, positions, n)
    ismissing(allowed_values) && return BranchConstraint((partial, depth, spaces) -> true)
    contribution_values = _contribution_values(subspaces, functions)
    T0 = mapreduce(vals -> mapreduce(typeof, promote_type, vals), promote_type, contribution_values)
    T = promote_type(T0, typeof(sum(first, contribution_values))) # `sum` widens small integer types
    contribution_min = zeros(T, n)
    contribution_max = zeros(T, n)
    position_starts = zeros(Int, n + 1)
    position_starts[1] = 1
    for (pos, vals) in zip(positions, contribution_values)
        contribution_min[pos] += minimum(vals)
        contribution_max[pos] += maximum(vals)
        position_starts[pos+1] += 1
    end
    cumsum!(position_starts, position_starts)
    subspace_order = sortperm(collect(positions))

    remaining_min = zeros(T, n + 1)
    remaining_max = zeros(T, n + 1)
    for i in n:-1:1
        remaining_min[i] = remaining_min[i+1] + contribution_min[i]
        remaining_max[i] = remaining_max[i+1] + contribution_max[i]
    end
    # `prefix[d]` is the contribution of positions `1:d-1` on the current branch. This relies on
    # `generate_states` validating depth `d` only after accepting depths `1:d-1` of the same branch.
    prefix = zeros(T, n + 1)
    BranchConstraint((partial, depth, _) -> begin
        current = prefix[depth]
        for i in position_starts[depth]:position_starts[depth+1]-1
            current += _contribution(functions, subspace_order[i], partial[depth])
        end
        prefix[depth+1] = current
        min_possible = current + remaining_min[depth+1]
        max_possible = current + remaining_max[depth+1]
        any(t -> min_possible <= t <= max_possible, allowed_values)
    end)
end
_contribution(func::Function, k, substate) = func(substate)
_contribution(fw::WeightedFunction, k, substate) = fw.weights[k] * fw.func(substate)
_contribution(functions, k, substate) = functions[k](substate)

branch_constraint(constraint::AdditiveConstraint, spaces) = additive_branch_constraint(constraint.allowed_values, constraint.maps, constraint.spaces, spaces)


@testitem "generate_states with BranchConstraint" begin
    using FermionicHilbertSpaces: generate_states, BranchConstraint, AdditiveConstraint, basisstate, hilbert_space, _bit, CombineFockNumbersProcessor, constrain_space, quantumnumbers, particle_number, combine_states

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
    additive = AdditiveConstraint([1];
        spaces=(Hs[1], Hs[2], Hs[3]),
        maps=(s -> 2 * particle_number(s),
            s -> -particle_number(s),
            s -> particle_number(s))
    )
    numcon = NumberConservation([1], (Hs[1], Hs[2], Hs[3]), (2, -1, 1))
    Hnum = tensor_product(Hs; constraint=numcon)
    Hadd = tensor_product(Hs; constraint=additive)
    @test Hnum == Hadd
    # states = generate_states(Hs, additive, H; process_result)
    @test all(state -> begin
        n1 = count_ones(state.f & 0b00001)
        n2 = count_ones(state.f & 0b00010)
        n3 = count_ones(state.f & 0b00100)
        2n1 - n2 + n3 == 1
    end, basisstates(Hadd))

    additive_with_shared_function = AdditiveConstraint([0, 2]; spaces=(Hs[4], Hs[5]), maps=particle_number)
    states = generate_states(Hs, additive_with_shared_function; process_result)
    @test all(state -> begin
        n4 = count_ones(state.f & 0b01000)
        n5 = count_ones(state.f & 0b10000)
        n4 + n5 in (0, 2)
    end, states)

    Hc = tensor_product(Hs; constraint=additive)
    @test collect(quantumnumbers(Hc)) == [1]

    compound = additive * AdditiveConstraint([1]; spaces=(Hs[4], Hs[5]), maps=particle_number)
    states = generate_states(Hs, compound; process_result)
    @test all(state -> begin
        n1 = count_ones(state.f & 0b00001)
        n2 = count_ones(state.f & 0b00010)
        n3 = count_ones(state.f & 0b00100)
        n4 = count_ones(state.f & 0b01000)
        n5 = count_ones(state.f & 0b10000)
        2n1 - n2 + n3 == 1 && n4 + n5 == 1
    end, states)

    # Test with composite and mixed spaces
    H1 = hilbert_space(f, 1:2)
    H2 = hilbert_space(f, 3:3)
    @boson b
    Hb = hilbert_space(b, 3)
    H = tensor_product((H1, H2, Hb))
    mapper = FermionicHilbertSpaces.state_mapper(H, (H1, H2, Hb))
    constraint = NumberConservation(1)
    _states = generate_states((H1, H2, Hb), constraint)
    states = map(s -> only(first(combine_states(s, mapper))), _states)
    states2 = basisstates(constrain_space(H, constraint))
    @test Set(states) == Set(states2)
end
@testitem "Additive constraints on subsets of mixed spaces" begin
    using FermionicHilbertSpaces: AdditiveConstraint
    @fermions f
    @spin s1 1
    @spin s2 1 // 2
    @boson b
    Hs = (hilbert_space(f[1]), hilbert_space(s1), hilbert_space(b, 3), hilbert_space(f[2]), hilbert_space(s2))
    H = tensor_product(Hs)
    spin_sum = AdditiveConstraint([1 // 2, 3 // 2]; spaces=(Hs[5], Hs[2]), maps=(s -> s.m, s -> 2 * s.m))
    weighted = NumberConservation([1, 2], (Hs[4], Hs[3]), (1, 2))
    bool_count = AdditiveConstraint([2]; spaces=(Hs[1], Hs[4]), maps=s -> isodd(s.f))
    for constraint in (spin_sum, weighted, bool_count, spin_sum * weighted)
        expected = basisstates(constrain_space(H, constraint))
        @test !isempty(expected)
        @test Set(basisstates(tensor_product(Hs; constraint))) == Set(expected)
    end
    Hfs = map(hilbert_space, (f[1], f[2], f[3]))
    weighted_all = NumberConservation([2]; weights=(1, 2, 1))
    @test Set(basisstates(tensor_product(Hfs; constraint=weighted_all))) == Set(FockNumber.([0b010, 0b101]))
end
