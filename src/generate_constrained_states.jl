
mask_region_size(weights::Vector) = count(!=(0), weights)
mask_region_size(mask::FockNumber) = mask_region_size(mask.f)
mask_region_size(mask::Integer) = count_ones(mask)
set_bit(num::FockNumber{T}, pos::Int, value::Bool) where T = FockNumber{T}(set_bit(num.f, pos, value))
function set_bit(num::T, pos::Int, value::Bool) where T
    mask = one(T) << (pos - 1)          # single-bit mask
    newf = value ? (num | mask) : (num & ~mask)
    T(newf)
end
# function generate_states(masks::Union{Vector{<:K},NTuple{N,<:K}}, allowed_ones, max_bits, ::Type{T}=default_fock_representation(max_bits)) where {N,T,K<:Integer}
#     region_lengths = map(mask_region_size, masks)
#     any(rl > max_bits for rl in region_lengths) && error("Constraint mask exceeds max_bits")
#     filled_ones = [0 for _ in masks]
#     #filled_zeros = [0 for _ in masks]
#     remaining_bits::Vector{Int} = collect(region_lengths)
#     states = T[]
#     num = zero(T)
#     if max_bits == 0
#         return [T(0)]
#     end
#     if max_bits < 0
#         error("max_bits must be non-negative")
#     end
#     bit_position = 1
#     # Build dependency mapping
#     affected_constraints = [Int[] for _ in 1:max_bits]
#     for (k, mask) in enumerate(masks)
#         for bit_pos in 1:(max_bits)
#             if _bit(mask, bit_pos)
#                 push!(affected_constraints[bit_pos], k)
#             end
#         end
#     end

#     operation_stack = [:put_one, :put_zero] # Stack to keep track of operations
#     sizehint!(operation_stack, max_bits * 3)

#     # count = 0
#     while !isempty(operation_stack)
#         # count += 1
#         op = pop!(operation_stack)

#         if op == :revert_zero
#             # Revert putting a zero at bit_position - 1
#             for k in affected_constraints[bit_position-1]
#                 #filled_zeros[k] -= 1
#                 remaining_bits[k] += 1
#             end
#             bit_position -= 1
#             continue
#         end

#         if op == :revert_one
#             # Revert putting a one at bit_position - 1
#             for k in affected_constraints[bit_position-1]
#                 filled_ones[k] -= 1
#                 remaining_bits[k] += 1
#             end
#             bit_position -= 1
#             continue
#         end

#         if op == :put_zero
#             feasible = affected_constraints_can_be_satisfied(false, affected_constraints[bit_position], allowed_ones, filled_ones, remaining_bits)
#             if feasible
#                 # Put a zero at bit_position
#                 num = set_bit(num, bit_position, false)
#                 if bit_position == max_bits
#                     push!(states, num)
#                     continue
#                 end
#                 push!(operation_stack, :revert_zero)
#                 for k in affected_constraints[bit_position]
#                     #filled_zeros[k] += 1
#                     remaining_bits[k] -= 1
#                 end
#                 bit_position += 1
#                 push!(operation_stack, :put_one)
#                 push!(operation_stack, :put_zero)
#             end
#             continue
#         end

#         if op == :put_one
#             feasible = affected_constraints_can_be_satisfied(true, affected_constraints[bit_position], allowed_ones, filled_ones, remaining_bits)

#             if feasible
#                 # Put a one at bit_position
#                 num = set_bit(num, bit_position, true)
#                 if bit_position == max_bits
#                     push!(states, num)
#                     continue
#                 end
#                 push!(operation_stack, :revert_one)
#                 for k in affected_constraints[bit_position]
#                     filled_ones[k] += 1
#                     remaining_bits[k] -= 1
#                 end
#                 bit_position += 1
#                 push!(operation_stack, :put_one)
#                 push!(operation_stack, :put_zero)
#             end
#         end
#     end
#     return states
# end

# @inline function affected_constraints_can_be_satisfied(testbit, ks, allowed_ones, filled_ones, remaining_bits)
#     for k in ks
#         feasible = false
#         newones = filled_ones[k] + testbit
#         remaining = remaining_bits[k] - 1
#         for target_ones in allowed_ones[k]
#             newones <= target_ones <= newones + remaining && (feasible = true) && break
#         end
#         !feasible && return false
#     end
#     return true
# end

# function get_mask(weights, T=default_fock_representation(length(weights)))
#     mask = zero(T)
#     for (i, w) in enumerate(weights)
#         w != 0 && (mask |= (one(T) << (i - 1)))
#     end
#     return mask
# end
# function generate_states(weights::Union{<:Vector,<:Tuple}, allowed_sums, max_bits, T=default_fock_representation(max_bits))
#     generate_states_weighted_constraints(weights, allowed_sums, max_bits, T)
# end
# function generate_states_weighted_constraints(weights, allowed_sums, max_bits, T=default_fock_representation(max_bits))
#     # Validate inputs
#     length(weights) == length(allowed_sums) || error("weights and allowed_sums must have same length")

#     # Derive masks from weights (non-zero weights indicate masked positions)
#     masks = [get_mask(weight_vec, T) for weight_vec in weights]
#     for (k, weight_vec) in enumerate(weights)
#         length(weight_vec) == max_bits || error("Weight vector $k must have length max_bits")
#         for i in 1:max_bits
#             if weight_vec[i] != 0
#                 masks[k] |= (one(T) << (i - 1))
#             end
#         end
#     end

#     region_lengths = map(count_ones, masks)
#     any(rl > max_bits for rl in region_lengths) && error("Constraint region exceeds max_bits")

#     current_sums = [0 for _ in weights]
#     # remaining_bits = collect(region_lengths)

#     # Precompute min/max possible contributions from remaining bits
#     remaining_min = [zeros(Int, max_bits + 1) for _ in weights]
#     remaining_max = [zeros(Int, max_bits + 1) for _ in weights]

#     for (k, weight_vec) in enumerate(weights)
#         for pos in max_bits:-1:1
#             if weight_vec[pos] != 0  # Position is in mask if weight is non-zero
#                 w = weight_vec[pos]
#                 next_min = pos < max_bits ? remaining_min[k][pos+1] : 0
#                 next_max = pos < max_bits ? remaining_max[k][pos+1] : 0
#                 remaining_min[k][pos] = next_min + min(0, w)
#                 remaining_max[k][pos] = next_max + max(0, w)
#             else
#                 remaining_min[k][pos] = pos < max_bits ? remaining_min[k][pos+1] : 0
#                 remaining_max[k][pos] = pos < max_bits ? remaining_max[k][pos+1] : 0
#             end
#         end
#     end

#     states = T[]
#     num = zero(T)
#     if max_bits == 0
#         return [T(0)]
#     end
#     if max_bits < 0
#         error("max_bits must be non-negative")
#     end
#     bit_position = 1

#     # Build dependency mapping (which constraints are affected by each bit position)
#     affected_constraints = [Int[] for _ in 1:max_bits]
#     for (k, weight_vec) in enumerate(weights)
#         for bit_pos in 1:max_bits
#             if weight_vec[bit_pos] != 0
#                 push!(affected_constraints[bit_pos], k)
#             end
#         end
#     end

#     operation_stack = [:put_one, :put_zero]
#     sizehint!(operation_stack, max_bits * 3)

#     while !isempty(operation_stack)
#         op = pop!(operation_stack)

#         if op == :revert_zero
#             for k in affected_constraints[bit_position-1]
#                 # remaining_bits[k] += 1
#             end
#             bit_position -= 1
#             continue
#         end

#         if op == :revert_one
#             for k in affected_constraints[bit_position-1]
#                 current_sums[k] -= weights[k][bit_position-1]
#                 # remaining_bits[k] += 1
#             end
#             bit_position -= 1
#             continue
#         end

#         if op == :put_zero
#             feasible = weighted_constraints_can_be_satisfied(
#                 false, bit_position, affected_constraints[bit_position],
#                 allowed_sums, weights, current_sums, remaining_min, remaining_max
#             )
#             if feasible
#                 num = set_bit(num, bit_position, false)
#                 if bit_position == max_bits
#                     push!(states, num)
#                     continue
#                 end
#                 push!(operation_stack, :revert_zero)
#                 for k in affected_constraints[bit_position]
#                     # remaining_bits[k] -= 1
#                 end
#                 bit_position += 1
#                 push!(operation_stack, :put_one)
#                 push!(operation_stack, :put_zero)
#             end
#             continue
#         end

#         if op == :put_one
#             feasible = weighted_constraints_can_be_satisfied(
#                 true, bit_position, affected_constraints[bit_position],
#                 allowed_sums, weights, current_sums, remaining_min, remaining_max
#             )
#             if feasible
#                 num = set_bit(num, bit_position, true)
#                 if bit_position == max_bits
#                     push!(states, num)
#                     continue
#                 end
#                 push!(operation_stack, :revert_one)
#                 for k in affected_constraints[bit_position]
#                     current_sums[k] += weights[k][bit_position]
#                     # remaining_bits[k] -= 1
#                 end
#                 bit_position += 1
#                 push!(operation_stack, :put_one)
#                 push!(operation_stack, :put_zero)
#             end
#         end
#     end
#     return states
# end

# @inline function weighted_constraints_can_be_satisfied(
#     testbit, bit_position, ks, allowed_sums, weights,
#     current_sums, remaining_min, remaining_max
# )
#     for k in ks
#         feasible = false
#         new_sum = current_sums[k] + (testbit ? weights[k][bit_position] : 0)

#         # Get min/max possible sum from remaining bits after this position
#         min_possible = new_sum + remaining_min[k][bit_position+1]
#         max_possible = new_sum + remaining_max[k][bit_position+1]

#         for target_sum in allowed_sums[k]
#             min_possible <= target_sum <= max_possible && (feasible = true) && break
#         end
#         !feasible && return false
#     end
#     return true
# end


# @testitem "generate_states" begin
#     using FermionicHilbertSpaces: generate_states
#     # Test 1: Simple constraint - exactly 2 ones in positions 1-4
#     masks = [0b1111]  # Mask for positions 1-4
#     allowed_ones = [[2]]  # Exactly 2 ones
#     max_bits = 4
#     states = generate_states(masks, allowed_ones, max_bits, UInt8)

#     # Should get all 4-bit numbers with exactly 2 ones
#     expected = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
#     @test sort(states) == expected

#     # Test 2: Multiple constraints
#     masks = [0b0111, 0b1110]  # Positions 1-3 and 2-4
#     allowed_ones = [[1, 2], [1, 2]]  # 1 or 2 ones in each region
#     max_bits = 4
#     states = generate_states(masks, allowed_ones, max_bits, UInt8)

#     # States must have 1-2 ones in positions 1-3 AND 1-2 ones in positions 2-4
#     valid_states = []
#     for i in 0:15
#         ones_123 = count_ones(i & 0b0111)
#         ones_234 = count_ones(i & 0b1110)
#         if (ones_123 in [1, 2]) && (ones_234 in [1, 2])
#             push!(valid_states, i)
#         end
#     end
#     @test sort(states) == valid_states

# end

# @testitem "generate_states_weighted_constraints" begin
#     using FermionicHilbertSpaces: _bit, generate_states, generate_states_weighted_constraints

#     # Test 1: Simple weighted sum
#     weights = [[1, 2, 3, 4]]  # Weights for positions 1-4
#     allowed_sums = [[5, 6]]  # Sum must be 5 or 6
#     max_bits = 4
#     states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

#     # Check that all generated states have the correct weighted sum
#     @test all(states) do state
#         weighted_sum = sum(i -> _bit(state, i) ? weights[1][i] : 0, 1:4)
#         weighted_sum in [5, 6]
#     end

#     # Test 2: Negative weights
#     masks = [0b0111]  # Positions 1-3
#     weights = [[-2, 3, -1, 0]]  # Mix of positive and negative weights
#     allowed_sums = [[0, 1]]  # Sum must be 0 or 1
#     max_bits = 4
#     states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

#     @test all(states) do state
#         weighted_sum = sum(i -> (_bit(state, i) && _bit(masks[1], i)) ? weights[1][i] : 0, 1:4)
#         weighted_sum in [0, 1]
#     end

#     # Test 3: Multiple weighted constraints
#     masks = [0b0011, 0b1100]  # Positions 1-2 and 3-4
#     weights = [[2, 3, 0, 0], [0, 0, 1, 4]]  # Weights for each constraint
#     allowed_sums = [[2, 3], [4, 5]]  # Different sum requirements
#     max_bits = 4
#     states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

#     @test all(states) do state
#         sum1 = sum(i -> (_bit(state, i) && _bit(masks[1], i)) ? weights[1][i] : 0, 1:4)
#         sum2 = sum(i -> (_bit(state, i) && _bit(masks[2], i)) ? weights[2][i] : 0, 1:4)
#         sum1 in [2, 3] && sum2 in [4, 5]
#     end

#     # Test 4: Verify weighted version can replicate counting behavior
#     masks = [0b1111]
#     weights = [[1, 1, 1, 1]]  # All weights = 1 (equivalent to counting)
#     allowed_sums = [[2]]  # Sum = 2 (equivalent to 2 ones)
#     max_bits = 4
#     weighted_states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

#     # Compare with regular generate_states
#     allowed_ones = [[2]]
#     regular_states = generate_states(masks, allowed_ones, max_bits, UInt8)

#     @test sort(weighted_states) == sort(regular_states)


#     mask = [0b1010, 0b0101]
#     allowed_sums1 = [[0], [2]]
#     weights = [[-1, 1, -1, 1, 0], [1, 1, 1, 1, 0]] #note different order than mask1
#     allowed_sums2 = [allowed_sums1[1] .- allowed_sums1[2], allowed_sums1[1] .+ allowed_sums1[2]]
#     max_bits = 5
#     states1 = generate_states(mask, allowed_sums1, max_bits, UInt8)
#     states2 = generate_states_weighted_constraints(weights, allowed_sums2, max_bits, UInt8)
#     @test sort(states1) == sort(states2)

#     @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed_sums
#         states = generate_states(mask, allowed_sums, max_bits, UInt8)
#         allowed_sums2 = [allowed_sums[1] - allowed_sums[2] allowed_sums[1] + allowed_sums[2]]
#         states2 = generate_states_weighted_constraints(weights, allowed_sums2, max_bits, UInt8)
#         sort(states) == sort(states2)
#     end

#     ## Tests for weighted_number_branch_constraint (BranchConstraint-based path)
#     using FermionicHilbertSpaces: weighted_number_branch_constraint, hilbert_space
#     leaf_processor = FermionicHilbertSpaces.CombineFockNumbersProcessor()

#     # Test 5: unit weights reproduce the unweighted (counting) result
#     Hs4 = [hilbert_space(n:n) for n in 1:4]
#     c_unit = weighted_number_branch_constraint([2], ones(Int, 4), Hs4)
#     states_unit = map(f -> f.f, generate_states(Hs4, c_unit; leaf_processor))
#     states_legacy = generate_states([0b1111], [[2]], 4, UInt8)
#     @test sort(states_unit) == sort(states_legacy)

#     # Test 6: non-unit positive weights, verified by brute force
#     Hs5 = [hilbert_space(n:n) for n in 1:5]
#     w_pos = [1, 2, 3, 4, 5]
#     c_pos = weighted_number_branch_constraint([6, 7], w_pos, Hs5)
#     states_pos = map(f -> f.f, generate_states(Hs5, c_pos; leaf_processor))
#     brute_pos = UInt8[s for s in UInt8(0):UInt8(31) if sum(i -> _bit(s, i) ? w_pos[i] : 0, 1:5) in [6, 7]]
#     @test sort(states_pos) == sort(brute_pos)

#     # Test 7: negative weights, verified by brute force
#     w_neg = [-1, 2, -1, 3]
#     c_neg = weighted_number_branch_constraint([1, 2], w_neg, Hs4)
#     states_neg = map(f -> f.f, generate_states(Hs4, c_neg; leaf_processor))
#     brute_neg = UInt8[s for s in UInt8(0):UInt8(15) if sum(i -> _bit(s, i) ? w_neg[i] : 0, 1:4) in [1, 2]]
#     @test sort(states_neg) == sort(brute_neg)

#     # Test 8: two constraints composed via *, verified by brute force
#     w1 = [1, 0, 1, 0, 1]   # odd sites only
#     w2 = [0, 1, 0, 1, 0]   # even sites only
#     c1 = weighted_number_branch_constraint([1], w1, Hs5)
#     c2 = weighted_number_branch_constraint([2], w2, Hs5)
#     states_combined = map(f -> f.f, generate_states(Hs5, c1 * c2; leaf_processor))
#     brute_combined = UInt8[s for s in UInt8(0):UInt8(31)
#                            if sum(i -> _bit(s, i) ? w1[i] : 0, 1:5) == 1 &&
#                            sum(i -> _bit(s, i) ? w2[i] : 0, 1:5) == 2]
#     @test sort(states_combined) == sort(brute_combined)
# end


##
# Abstract type for constraints
abstract type AbstractBranchConstraint end
struct BranchConstraint{F} <: AbstractBranchConstraint
    f::F
end

"""
    valid_branch(constraint, partial_state, remaining_spaces) -> Bool

Return `true` if the branch should be explored, `false` to prune.
By default this calls `constraint.f(partial_state, remaining_spaces)`.
"""
valid_branch(constraint::AbstractBranchConstraint, partial_state, depth, spaces) = constraint.f(partial_state, depth, spaces)

process_partial(::Nothing, partial_state, depth, spaces) = nothing
process_partial(processor, partial_state, depth, spaces) = processor(partial_state, depth, spaces)

process_leaf(::Nothing, full_state, spaces) = Tuple(full_state)
process_leaf(::typeof(identity), full_state, spaces) = Tuple(full_state)
process_leaf(processor, full_state, spaces) = processor(full_state, spaces)

_init_results(spaces, ::typeof(identity)) = Tuple{statetype.(spaces)...}[]
_init_results(spaces, leaf_processor) = Any[]

"""
    generate_states(spaces, constraint::AbstractBranchConstraint; partial_processor=nothing, leaf_processor=identity)

Generate all tensor product states from `spaces` satisfying `constraint`.
Uses backtracking with pruning via `valid_branch`.

`partial_processor(partial_state, depth, spaces)` is called whenever a branch is accepted.
`leaf_processor(full_state, spaces)` can transform each completed state before storing it.
"""
function generate_states(space, constraint::AbstractBranchConstraint; partial_processor=nothing, leaf_processor=identity)
    spaces = factors(space)
    results = _init_results(spaces, leaf_processor)

    if isempty(spaces)
        push!(results, process_leaf(leaf_processor, (), spaces))
        return results
    end
    all_statetypes = statetype.(spaces)
    # Start backtracking
    partial = Vector{Union{all_statetypes...}}(undef, length(spaces))
    backtrack!(results, partial, spaces, 1, constraint, partial_processor, leaf_processor)

    return results
end

function backtrack!(results, partial, spaces, depth, constraint, partial_processor, leaf_processor)
    n = length(spaces)

    if depth > n
        # All spaces assigned, add to results
        push!(results, process_leaf(leaf_processor, partial, spaces))
        return
    end

    for state in basisstates(spaces[depth])
        partial[depth] = state
        # Check if this branch is worth exploring
        if valid_branch(constraint, partial, depth, spaces)
            process_partial(partial_processor, partial, depth, spaces)
            backtrack!(results, partial, spaces, depth + 1, constraint, partial_processor, leaf_processor)
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

struct ProductBranchConstraint{C} <: AbstractBranchConstraint
    constraints::C
end
Base.:*(constraint1::ProductBranchConstraint, constraint2::ProductBranchConstraint) = ProductBranchConstraint((constraint1.constraints..., constraint2.constraints...))
function valid_branch(constraint::ProductBranchConstraint, partial_state, depth, spaces)
    for c in constraint.constraints
        !valid_branch(c, partial_state, depth, spaces) && return false
    end
    return true
end
Base.:*(constraint1::BranchConstraint, constraint2::ProductBranchConstraint) = ProductBranchConstraint((constraint1, constraint2.constraints...))
Base.:*(constraint1::ProductBranchConstraint, constraint2::BranchConstraint) = ProductBranchConstraint((constraint1.constraints..., constraint2))
Base.:*(constraint1::BranchConstraint, constraint2::BranchConstraint) = ProductBranchConstraint((constraint1, constraint2))



@testitem "generate_states with BranchConstraint" begin
    using FermionicHilbertSpaces: generate_states, BranchConstraint, basisstate, hilbert_space, _bit, unweighted_number_branch_constraint

    # Define a simple constraint: only allow states where the first space is in its first basis state
    @fermions f
    H1 = hilbert_space(f, 1:1)  # Basis states: |0>, |1>
    H2 = hilbert_space(f, 2:2)  # Basis states: |0>, |1>
    spaces = (H1, H2)

    constraint = BranchConstraint((partial, depth, spaces) -> partial[1] == basisstate(1, H1))
    states = generate_states(spaces, constraint)

    # Should only get states where the first space is in its first basis state (|0>)
    expected = [(basisstate(1, H1), basisstate(1, H2)), (basisstate(1, H1), basisstate(2, H2))]
    @test sort(states) == sort(expected)

    # Empty product has one state: the empty tuple
    @test generate_states((), BranchConstraint((partial, depth, spaces) -> true)) == [()]

    # Partial and leaf processors are both applied
    visited_depths = Int[]
    leaf_processor = FermionicHilbertSpaces.CombineFockNumbersProcessor()
    states_as_int = map(f -> f.f, generate_states(
        spaces,
        BranchConstraint((partial, depth, spaces) -> true);
        partial_processor=(partial, depth, spaces) -> push!(visited_depths, depth),
        leaf_processor))
    @test count(==(1), visited_depths) == 2
    @test count(==(2), visited_depths) == 4
    @test sort(states_as_int) == [0x00, 0x01, 0x02, 0x03]

    ## Check consistency with generate_states for particle numbers

    masks = [0b1010, 0b0101]
    allowed_ones = [[0], [2]]
    Hs = [hilbert_space(n:n) for n in 1:5]

    max_bits = 5
    legacy_states = generate_states(masks, allowed_ones, max_bits, UInt8)

    c1 = unweighted_number_branch_constraint(allowed_ones[1], [Hs[2], Hs[4]], Hs)
    c2 = unweighted_number_branch_constraint(allowed_ones[2], [Hs[1], Hs[3]], Hs)
    constraint = c1 * c2
    generic_states = map(f -> f.f, generate_states(Hs, constraint; leaf_processor))
    @test sort(generic_states) == sort(legacy_states)

    @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed
        legacy_states = generate_states(masks, allowed, max_bits, UInt8)
        constraint = unweighted_number_branch_constraint(masks, allowed, max_bits)
        generic_states = generate_states(Hs, constraint; leaf_processor)
        sort(generic_states) == sort(legacy_states)
    end


end