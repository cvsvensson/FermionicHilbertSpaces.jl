

# mask_region_size(weights::Vector) = count(!=(0), weights)
# mask_region_size(mask::FockNumber) = mask_region_size(mask.f)
# mask_region_size(mask::Integer) = count_ones(mask)
# set_bit(num::FockNumber{T}, pos::Int, value::Bool) where T = FockNumber{T}(set_bit(num.f, pos, value))
# function set_bit(num::T, pos::Int, value::Bool) where T
#     mask = one(T) << (pos - 1)          # single-bit mask
#     newf = value ? (num | mask) : (num & ~mask)
#     T(newf)
# end


##
# Abstract type for constraints
abstract type AbstractConstraint end
abstract type AbstractBranchConstraint <: AbstractConstraint end
struct NoSymmetry <: AbstractConstraint end

struct ProductConstraint{C} <: AbstractConstraint
    constraints::C
end

instantiate(constraints::ProductConstraint, H) = prod(instantiate(cons, H) for cons in constraints.constraints)
Base.:*(sym1::AbstractConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1, sym2))
Base.:*(sym1::AbstractConstraint, sym2::ProductConstraint) = ProductConstraint((sym1, sym2.constraints...))
Base.:*(sym1::ProductConstraint, sym2::AbstractConstraint) = ProductConstraint((sym1.constraints..., sym2))
Base.:*(sym1::ProductConstraint, sym2::ProductConstraint) = ProductConstraint((sym1.constraints..., sym2.constraints...))
generate_states(space, constraint::ProductConstraint; kwargs...) = generate_states(space, constraint.constraints; kwargs...)

struct BranchConstraint{F} <: AbstractBranchConstraint
    f::F
end
valid_branch(constraint::BranchConstraint, partial_state, depth, spaces) = constraint.f(partial_state, depth, spaces)

"""
    valid_branch(constraint, partial_state, remaining_spaces) -> Bool

Return `true` if the branch should be explored, `false` to prune.
By default this calls `constraint.f(partial_state, remaining_spaces)`.
"""
valid_branch(constraints, partial_state, depth, spaces) = all(valid_branch(constraint, partial_state, depth, spaces) for constraint in constraints)

process_partial(::Nothing, partial_state, depth, spaces) = nothing
process_partial(processor, partial_state, depth, spaces) = processor(partial_state, depth, spaces)

process_leaf(::Nothing, full_state, spaces) = Tuple(full_state)
process_leaf(::typeof(identity), full_state, spaces) = Tuple(full_state)
process_leaf(processor, full_state, spaces) = processor(full_state, spaces)

_init_results(spaces, ::typeof(identity)) = Tuple{statetype.(spaces)...}[]
_init_results(spaces, leaf_processor) = Any[]

"""
    generate_states(spaces, constraints; partial_processor=nothing, leaf_processor=identity)

Generate all tensor product states from `spaces` satisfying `constraint`.
Uses backtracking with pruning via `valid_branch`.

`partial_processor(partial_state, depth, spaces)` is called whenever a branch is accepted.
`leaf_processor(full_state, spaces)` can transform each completed state before storing it.
"""
generate_states(space, constraint::AbstractConstraint; kwargs...) = generate_states(space, (constraint,); kwargs...)
function generate_states(space, _constraints; partial_processor=nothing, leaf_processor=identity)
    spaces = factors(space)
    constraints = map(c -> branch_constraint(c, space), _constraints)
    results = _init_results(spaces, leaf_processor)
    all_statetypes = statetype.(spaces)
    # Start backtracking
    partial = Vector{Union{all_statetypes...}}(undef, length(spaces))
    backtrack!(results, partial, spaces, 1, constraints, partial_processor, leaf_processor)

    return results
end

function backtrack!(results, partial, spaces, depth, constraints, partial_processor, leaf_processor)
    n = length(spaces)

    if depth > n
        # All spaces assigned, add to results
        push!(results, process_leaf(leaf_processor, partial, spaces))
        return
    end

    for state in basisstates(spaces[depth])
        partial[depth] = state
        # Check if this branch is worth exploring
        if valid_branch(constraints, partial, depth, spaces)
            process_partial(partial_processor, partial, depth, spaces)
            backtrack!(results, partial, spaces, depth + 1, constraints, partial_processor, leaf_processor)
        end
    end
end

function catenate_fock_states(full_state, spaces)
    T = default_fock_representation(sum(nbr_of_modes, spaces))
    num = FockNumber(zero(T))
    shift = 0
    for (state, space) in zip(full_state, spaces)
        num |= state << shift
        shift += nbr_of_modes(space)
    end
    num
end

struct CombineFockNumbersProcessor end
function (processor::CombineFockNumbersProcessor)(full_state, spaces)
    catenate_fock_states(full_state, spaces)
end
_init_results(spaces, ::CombineFockNumbersProcessor) = FockNumber{default_fock_representation(sum(nbr_of_modes, spaces))}[]
unweighted_number_branch_constraint(allowed_numbers, ::Nothing, allspaces) = unweighted_number_branch_constraint(allowed_numbers, allspaces, allspaces)

function unweighted_number_branch_constraint(allowed_numbers, subspaces, allspaces)
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
function weighted_number_branch_constraint(allowed_sums, weights, allspaces)
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

maximum_particles(H::AbstractFockHilbertSpace) = nbr_of_modes(H)
particle_number(s::FockNumber) = count_ones(s.f)
particle_number(s::FockNumber{Bool}) = s.f ? 1 : 0


@testitem "generate_states with BranchConstraint" begin
    using FermionicHilbertSpaces: generate_states, BranchConstraint, basisstate, hilbert_space, _bit, unweighted_number_branch_constraint

    # Define a simple constraint: only allow states where the first space is in its first basis state
    @fermions f
    H1 = hilbert_space(f, 1:1)  # Basis states: |0>, |1>
    H2 = hilbert_space(f, 2:2)  # Basis states: |0>, |1>
    H = tensor_product((H1, H2))

    constraint = BranchConstraint((partial, depth, spaces) -> partial[1] == basisstate(1, H1))
    states = generate_states(H, constraint)

    # Should only get states where the first space is in its first basis state (|0>)
    expected = [(basisstate(1, H1), basisstate(1, H2)), (basisstate(1, H1), basisstate(2, H2))]
    @test sort(states) == sort(expected)

    # Partial and leaf processors are both applied
    visited_depths = Int[]
    leaf_processor = FermionicHilbertSpaces.CombineFockNumbersProcessor()
    states_as_int = map(f -> f.f, generate_states(
        H,
        BranchConstraint((partial, depth, spaces) -> true);
        partial_processor=(partial, depth, spaces) -> push!(visited_depths, depth),
        leaf_processor))
    @test count(==(1), visited_depths) == 2
    @test count(==(2), visited_depths) == 4
    @test sort(states_as_int) == [0x00, 0x01, 0x02, 0x03]


    masks = [0b1010, 0b0101]
    allowed_ones = [[0], [2]]
    Hs = [hilbert_space(f, n:n) for n in 1:5]
    H = tensor_product(Hs)
    c1 = unweighted_number_branch_constraint(allowed_ones[1], [Hs[2], Hs[4]], Hs)
    c2 = unweighted_number_branch_constraint(allowed_ones[2], [Hs[1], Hs[3]], Hs)
    constraint = c1 * c2
    states = generate_states(H, constraint; leaf_processor)
    # verify particle numbers 
    for state in states
        @test count_ones(state.f & masks[1]) in allowed_ones[1]
        @test count_ones(state.f & masks[2]) in allowed_ones[2]
    end

    cons2 = NumberConservation(allowed_ones[1], [Hs[2], Hs[4]]) * NumberConservation(allowed_ones[2], [Hs[1], Hs[3]])
    @test states == generate_states(H, FermionicHilbertSpaces.instantiate(cons2, H); leaf_processor)

    @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed
        c1 = unweighted_number_branch_constraint(allowed[1], [Hs[2], Hs[4]], Hs)
        c2 = unweighted_number_branch_constraint(allowed[2], [Hs[1], Hs[3]], Hs)
        constraint = c1 * c2
        states = generate_states(H, constraint; leaf_processor)
        for state in states
            count_ones(state.f & masks[1]) in allowed[1] || return false
            count_ones(state.f & masks[2]) in allowed[2] || return false
        end
        true
    end

end