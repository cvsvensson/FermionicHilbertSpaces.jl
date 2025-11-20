
mask_region_size(weights::Vector) = count(!=(0), weights)
mask_region_size(mask::FockNumber) = mask_region_size(mask.f)
mask_region_size(mask::Integer) = count_ones(mask)
set_bit!(num::FockNumber{T}, pos::Int, value::Bool) where T = FockNumber{T}(set_bit!(num.f, pos, value))
function set_bit!(num::T, pos::Int, value::Bool) where T
    mask = one(T) << (pos - 1)          # single-bit mask
    newf = value ? (num | mask) : (num & ~mask)
    T(newf)
end
function generate_states(masks::Union{Vector{<:K},NTuple{N,<:K}}, allowed_ones, max_bits, ::Type{T}=default_fock_representation(max_bits)) where {N,T,K<:Integer}
    region_lengths = map(mask_region_size, masks)
    any(rl > max_bits for rl in region_lengths) && error("Constraint mask exceeds max_bits")
    filled_ones = [0 for _ in masks]
    filled_zeros = [0 for _ in masks]
    remaining_bits::Vector{Int} = collect(region_lengths)
    states = T[]
    num = zero(T)
    if max_bits == 0
        return [T(0)]
    end
    if max_bits < 0
        error("max_bits must be non-negative")
    end
    bit_position = 1
    # Build dependency mapping
    affected_constraints = [Int[] for _ in 1:max_bits]
    for (k, mask) in enumerate(masks)
        for bit_pos in 1:(max_bits)
            if _bit(mask, bit_pos)
                push!(affected_constraints[bit_pos], k)
            end
        end
    end

    operation_stack = [:put_one, :put_zero] # Stack to keep track of operations
    sizehint!(operation_stack, max_bits * 3)

    # count = 0
    while !isempty(operation_stack)
        # count += 1
        op = pop!(operation_stack)

        if op == :revert_zero
            # Revert putting a zero at bit_position - 1
            for k in affected_constraints[bit_position-1]
                filled_zeros[k] -= 1
                remaining_bits[k] += 1
            end
            bit_position -= 1
            continue
        end

        if op == :revert_one
            # Revert putting a one at bit_position - 1
            for k in affected_constraints[bit_position-1]
                filled_ones[k] -= 1
                remaining_bits[k] += 1
            end
            bit_position -= 1
            continue
        end

        if op == :put_zero
            feasible = affected_constraints_can_be_satisfied(false, affected_constraints[bit_position], allowed_ones, region_lengths, filled_ones, filled_zeros, remaining_bits)
            if feasible
                # Put a zero at bit_position
                num = set_bit!(num, bit_position, false)
                if bit_position == max_bits
                    push!(states, num)
                    continue
                end
                push!(operation_stack, :revert_zero)
                for k in affected_constraints[bit_position]
                    filled_zeros[k] += 1
                    remaining_bits[k] -= 1
                end
                bit_position += 1
                push!(operation_stack, :put_one)
                push!(operation_stack, :put_zero)
            end
            continue
        end

        if op == :put_one
            feasible = affected_constraints_can_be_satisfied(true, affected_constraints[bit_position], allowed_ones, region_lengths, filled_ones, filled_zeros, remaining_bits)

            if feasible
                # Put a one at bit_position
                num = set_bit!(num, bit_position, true)
                if bit_position == max_bits
                    push!(states, num)
                    continue
                end
                push!(operation_stack, :revert_one)
                for k in affected_constraints[bit_position]
                    filled_ones[k] += 1
                    remaining_bits[k] -= 1
                end
                bit_position += 1
                push!(operation_stack, :put_one)
                push!(operation_stack, :put_zero)
            end
        end
    end
    return states
end

@inline function affected_constraints_can_be_satisfied(testbit, ks, allowed_ones, region_lengths, filled_ones, filled_zeros, remaining_bits)
    for k in ks
        feasible = false
        newones = filled_ones[k] + testbit
        newzeros = filled_zeros[k] + !testbit
        remaining = remaining_bits[k] - 1
        rl = region_lengths[k]
        for target_ones in allowed_ones[k]
            newones <= target_ones <= newones + remaining && (feasible = true) && break
        end
        !feasible && return false
    end
    return true
end

function get_mask(weights, T=default_fock_representation(length(weights)))
    mask = zero(T)
    for (i, w) in enumerate(weights)
        w != 0 && (mask |= (one(T) << (i - 1)))
    end
    return mask
end
function generate_states(weights::Union{<:Vector,<:Tuple}, allowed_sums, max_bits, T=default_fock_representation(max_bits))
    generate_states_weighted_constraints(weights, allowed_sums, max_bits, T)
end
function generate_states_weighted_constraints(weights, allowed_sums, max_bits, T=default_fock_representation(max_bits))
    # Validate inputs
    length(weights) == length(allowed_sums) || error("weights and allowed_sums must have same length")

    # Derive masks from weights (non-zero weights indicate masked positions)
    masks = [get_mask(weight_vec, T) for weight_vec in weights]
    for (k, weight_vec) in enumerate(weights)
        length(weight_vec) == max_bits || error("Weight vector $k must have length max_bits")
        for i in 1:max_bits
            if weight_vec[i] != 0
                masks[k] |= (one(T) << (i - 1))
            end
        end
    end

    region_lengths = map(count_ones, masks)
    any(rl > max_bits for rl in region_lengths) && error("Constraint region exceeds max_bits")

    current_sums = [0 for _ in weights]
    remaining_bits = collect(region_lengths)

    # Precompute min/max possible contributions from remaining bits
    remaining_min = [zeros(Int, max_bits + 1) for _ in weights]
    remaining_max = [zeros(Int, max_bits + 1) for _ in weights]

    for (k, weight_vec) in enumerate(weights)
        for pos in max_bits:-1:1
            if weight_vec[pos] != 0  # Position is in mask if weight is non-zero
                w = weight_vec[pos]
                next_min = pos < max_bits ? remaining_min[k][pos+1] : 0
                next_max = pos < max_bits ? remaining_max[k][pos+1] : 0
                remaining_min[k][pos] = next_min + min(0, w)
                remaining_max[k][pos] = next_max + max(0, w)
            else
                remaining_min[k][pos] = pos < max_bits ? remaining_min[k][pos+1] : 0
                remaining_max[k][pos] = pos < max_bits ? remaining_max[k][pos+1] : 0
            end
        end
    end

    states = T[]
    num = zero(T)
    if max_bits == 0
        return [T(0)]
    end
    if max_bits < 0
        error("max_bits must be non-negative")
    end
    bit_position = 1

    # Build dependency mapping (which constraints are affected by each bit position)
    affected_constraints = [Int[] for _ in 1:max_bits]
    for (k, weight_vec) in enumerate(weights)
        for bit_pos in 1:max_bits
            if weight_vec[bit_pos] != 0
                push!(affected_constraints[bit_pos], k)
            end
        end
    end

    operation_stack = [:put_one, :put_zero]
    sizehint!(operation_stack, max_bits * 3)

    while !isempty(operation_stack)
        op = pop!(operation_stack)

        if op == :revert_zero
            for k in affected_constraints[bit_position-1]
                remaining_bits[k] += 1
            end
            bit_position -= 1
            continue
        end

        if op == :revert_one
            for k in affected_constraints[bit_position-1]
                current_sums[k] -= weights[k][bit_position-1]
                remaining_bits[k] += 1
            end
            bit_position -= 1
            continue
        end

        if op == :put_zero
            feasible = weighted_constraints_can_be_satisfied(
                false, bit_position, affected_constraints[bit_position],
                allowed_sums, weights, current_sums, remaining_min, remaining_max
            )
            if feasible
                num = set_bit!(num, bit_position, false)
                if bit_position == max_bits
                    push!(states, num)
                    continue
                end
                push!(operation_stack, :revert_zero)
                for k in affected_constraints[bit_position]
                    remaining_bits[k] -= 1
                end
                bit_position += 1
                push!(operation_stack, :put_one)
                push!(operation_stack, :put_zero)
            end
            continue
        end

        if op == :put_one
            feasible = weighted_constraints_can_be_satisfied(
                true, bit_position, affected_constraints[bit_position],
                allowed_sums, weights, current_sums, remaining_min, remaining_max
            )
            if feasible
                num = set_bit!(num, bit_position, true)
                if bit_position == max_bits
                    push!(states, num)
                    continue
                end
                push!(operation_stack, :revert_one)
                for k in affected_constraints[bit_position]
                    current_sums[k] += weights[k][bit_position]
                    remaining_bits[k] -= 1
                end
                bit_position += 1
                push!(operation_stack, :put_one)
                push!(operation_stack, :put_zero)
            end
        end
    end
    return states
end

@inline function weighted_constraints_can_be_satisfied(
    testbit, bit_position, ks, allowed_sums, weights,
    current_sums, remaining_min, remaining_max
)
    for k in ks
        feasible = false
        new_sum = current_sums[k] + (testbit ? weights[k][bit_position] : 0)

        # Get min/max possible sum from remaining bits after this position
        min_additional = bit_position < length(remaining_min[k]) - 1 ? remaining_min[k][bit_position+1] : 0
        max_additional = bit_position < length(remaining_max[k]) - 1 ? remaining_max[k][bit_position+1] : 0

        min_possible = new_sum + min_additional
        max_possible = new_sum + max_additional

        for target_sum in allowed_sums[k]
            min_possible <= target_sum <= max_possible && (feasible = true) && break
        end
        !feasible && return false
    end
    return true
end


@testitem "generate_states" begin
    using FermionicHilbertSpaces: generate_states
    # Test 1: Simple constraint - exactly 2 ones in positions 1-4
    masks = [0b1111]  # Mask for positions 1-4
    allowed_ones = [[2]]  # Exactly 2 ones
    max_bits = 4
    states = generate_states(masks, allowed_ones, max_bits, UInt8)

    # Should get all 4-bit numbers with exactly 2 ones
    expected = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
    @test sort(states) == expected

    # Test 2: Multiple constraints
    masks = [0b0111, 0b1110]  # Positions 1-3 and 2-4
    allowed_ones = [[1, 2], [1, 2]]  # 1 or 2 ones in each region
    max_bits = 4
    states = generate_states(masks, allowed_ones, max_bits, UInt8)

    # States must have 1-2 ones in positions 1-3 AND 1-2 ones in positions 2-4
    valid_states = []
    for i in 0:15
        ones_123 = count_ones(i & 0b0111)
        ones_234 = count_ones(i & 0b1110)
        if (ones_123 in [1, 2]) && (ones_234 in [1, 2])
            push!(valid_states, i)
        end
    end
    @test sort(states) == valid_states

end

@testitem "generate_states_weighted_constraints" begin
    using FermionicHilbertSpaces: _bit, generate_states, generate_states_weighted_constraints

    # Test 1: Simple weighted sum
    weights = [[1, 2, 3, 4]]  # Weights for positions 1-4
    allowed_sums = [[5, 6]]  # Sum must be 5 or 6
    max_bits = 4
    states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

    # Check that all generated states have the correct weighted sum
    @test all(states) do state
        weighted_sum = sum(i -> _bit(state, i) ? weights[1][i] : 0, 1:4)
        weighted_sum in [5, 6]
    end

    # Test 2: Negative weights
    masks = [0b0111]  # Positions 1-3
    weights = [[-2, 3, -1, 0]]  # Mix of positive and negative weights
    allowed_sums = [[0, 1]]  # Sum must be 0 or 1
    max_bits = 4
    states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

    @test all(states) do state
        weighted_sum = sum(i -> (_bit(state, i) && _bit(masks[1], i)) ? weights[1][i] : 0, 1:4)
        weighted_sum in [0, 1]
    end

    # Test 3: Multiple weighted constraints
    masks = [0b0011, 0b1100]  # Positions 1-2 and 3-4
    weights = [[2, 3, 0, 0], [0, 0, 1, 4]]  # Weights for each constraint
    allowed_sums = [[2, 3], [4, 5]]  # Different sum requirements
    max_bits = 4
    states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

    @test all(states) do state
        sum1 = sum(i -> (_bit(state, i) && _bit(masks[1], i)) ? weights[1][i] : 0, 1:4)
        sum2 = sum(i -> (_bit(state, i) && _bit(masks[2], i)) ? weights[2][i] : 0, 1:4)
        sum1 in [2, 3] && sum2 in [4, 5]
    end

    # Test 4: Verify weighted version can replicate counting behavior
    masks = [0b1111]
    weights = [[1, 1, 1, 1]]  # All weights = 1 (equivalent to counting)
    allowed_sums = [[2]]  # Sum = 2 (equivalent to 2 ones)
    max_bits = 4
    weighted_states = generate_states_weighted_constraints(weights, allowed_sums, max_bits, UInt8)

    # Compare with regular generate_states
    allowed_ones = [[2]]
    regular_states = generate_states(masks, allowed_ones, max_bits, UInt8)

    @test sort(weighted_states) == sort(regular_states)


    mask = [0b1010, 0b0101]
    allowed_sums1 = [[0], [2]]
    weights = [[-1, 1, -1, 1, 0], [1, 1, 1, 1, 0]] #note different order than mask1
    allowed_sums2 = [allowed_sums1[1] .- allowed_sums1[2], allowed_sums1[1] .+ allowed_sums1[2]]
    max_bits = 5
    states1 = generate_states(mask, allowed_sums1, max_bits, UInt8)
    states2 = generate_states_weighted_constraints(weights, allowed_sums2, max_bits, UInt8)
    @test sort(states1) == sort(states2)

    @test all([[[0], [2]], [[1], [1]], [[0], [0]], [[2], [2]]]) do allowed_sums
        states = generate_states(mask, allowed_sums, max_bits, UInt8)
        allowed_sums2 = [allowed_sums[1] - allowed_sums[2] allowed_sums[1] + allowed_sums[2]]
        states2 = generate_states_weighted_constraints(weights, allowed_sums2, max_bits, UInt8)
        sort(states) == sort(states2)
    end
end

