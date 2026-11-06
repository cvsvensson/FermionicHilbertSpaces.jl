# Sums that vary at a common depth are correlated, such as a charge and a dipole moment or a charge and the bonds of a
# blockade, so the bounds of each remaining sum alone accept branches that no completion satisfies. `_joint_cuts` lets
# `_sums_branch_constraint` also prune such sums by their joint values: `cuts[d]` holds, for each set of linked sums, the
# sorted keys of the joint values after depth `d` that the valid states reach, which makes the pruning exact. A single
# sum is pruned this way when the values of one of its groups have gaps, as for weights `5` and `1`. A sum is in the key
# only while it varies both up to the cut and after it, so that local constraints do not enlarge the sets, and the key
# also holds the states of the depths up to the cut that a later group reads, as for `nᵢnᵢ₊₁`. The keys are found by a
# forward pass over those that the prefixes reach, narrowed by the bounds, and a backward pass that keeps those that a
# suffix completes. Sums with values that are not exact, or with more than `MAX_JOINT_ENTRIES` keys, keep the bounds.
const MAX_JOINT_ENTRIES = 2^22

function _joint_cuts(p)
    n, T = length(p.sizes), eltype(p.constants)
    Cut = @NamedTuple{terms::Vector{Tuple{Int, Int, T, T, Int}}, opens::Vector{Tuple{Int, Int}}, allowed::Vector{Int}}
    cuts = [Cut[] for _ in 1:n]
    T <: Union{Integer, Rational} || return cuts
    varying = [any(g -> !allequal(p.tables[g]), gs) for gs in p.step_groups]
    linked = collect(eachindex(p.constants))
    root(k) = linked[k] == k ? k : root(linked[k])
    for d in 1:n
        ks = [p.step_owner[s] for s in searchsorted(p.step_depth, d) if varying[s]]
        foreach(k -> linked[root(k)] = root(first(ks)), ks)
    end
    for r in unique(root.(eachindex(p.constants)))
        steps = filter(s -> varying[s] && root(p.step_owner[s]) == r, eachindex(varying))
        gapped = any(s -> any(g -> !all(isone, diff(sort!(unique(p.tables[g])))), p.step_groups[s]), steps)
        isempty(steps) || (gapped || !allequal(p.step_owner[steps])) && _add_joint_cuts!(cuts, p, steps)
    end
    return cuts
end

# The key at the cut after depth `d` is `Σₖ stride[k, d + 1] * Kₖ` over the active sums, where the value of the sum `k`
# is `offset[k, d + 1] + Kₖ * scale[k]`, with `Kₖ` modulo the modulus, if any, plus `stride * (x - 1)` for the state
# `x` at each open depth.
function _add_joint_cuts!(cuts, p, steps)
    n, T, K = length(p.sizes), eltype(p.constants), length(p.constants)
    owners = unique(p.step_owner[steps])
    first_depth, last_depth = fill(n + 1, K), zeros(Int, K)
    scale = [iszero(m) ? zero(T) : one(T) for m in p.moduli]
    last_read = zeros(Int, n)
    for s in steps, g in p.step_groups[s]
        k, d, t = p.step_owner[s], p.step_depth[s], p.tables[g]
        first_depth[k], last_depth[k] = min(first_depth[k], d), max(last_depth[k], d)
        iszero(p.moduli[k]) && (scale[k] = gcd(scale[k], mapreduce(v -> v - first(t), gcd, t)))
        foreach(j -> last_read[j] = max(last_read[j], d), p.positions[g])
    end
    lo, width = zeros(T, K, n + 1), zeros(Int, K, n + 1)
    lo[:, 1] = p.constants
    for (s, gs) in enumerate(p.step_groups), g in gs
        k, t = p.step_owner[s], p.tables[g]
        k in owners || continue
        lo[k, p.step_depth[s] + 1] += minimum(t)
        width[k, p.step_depth[s] + 1] += Int(div(maximum(t) - minimum(t), scale[k]))
    end
    offset, range = cumsum(lo; dims = 2), 1 .+ cumsum(width; dims = 2)
    foreach(k -> iszero(p.moduli[k]) || (range[k, :] .= min.(range[k, :], p.moduli[k])), owners)
    active = [filter(k -> first_depth[k] <= cut < last_depth[k], owners) for cut in 0:n]
    opens = [filter(j -> last_read[j] > cut, 1:cut) for cut in 0:n]
    stride, open_stride = zeros(Int, K, n + 1), [zeros(Int, length(o)) for o in opens]
    for cut in 1:(n + 1)
        total = 1
        for k in active[cut]
            stride[k, cut] = total
            total, overflow = Base.mul_with_overflow(total, range[k, cut])
            overflow && return
        end
        for (m, j) in enumerate(opens[cut])
            open_stride[cut][m] = total
            total, overflow = Base.mul_with_overflow(total, p.sizes[j])
            overflow && return
        end
    end
    steps_at = [steps[searchsorted(p.step_depth[steps], d)] for d in 1:n]
    index, coordinate, value, stepped = zeros(Int, n), zeros(Int, K), zeros(T, K), zeros(Int, K)
    # Whether `f` holds for the key of an extension of `key`, at the cut before depth `d`, by a state at `d` that the
    # bounds of each sum allow.
    function extends(f, key, d)
        foreach(k -> coordinate[k] = mod(div(key, stride[k, d]), range[k, d]), active[d])
        foreach((j, st) -> index[j] = mod(div(key, st), p.sizes[j]) + 1, opens[d], open_stride[d])
        for x in 1:p.sizes[d]
            index[d] = x
            ok = true
            for s in steps_at[d]
                k = p.step_owner[s]
                v = offset[k, d] + coordinate[k] * scale[k] * (first_depth[k] < d)
                for g in p.step_groups[s]
                    v += p.tables[g][_table_index(p.strides[g], index, p.positions[g])]
                end
                ok = p.reachable(v, s)
                ok || break
                value[k], stepped[k] = v, d
            end
            ok || continue
            new = 0
            for k in active[d + 1]
                c = stepped[k] == d ? _joint_coordinate(value[k], offset[k, d + 1], scale[k], p.moduli[k]) : coordinate[k]
                new += stride[k, d + 1] * c
            end
            for (j, st) in zip(opens[d + 1], open_stride[d + 1])
                new += st * (index[j] - 1)
            end
            f(new) && return true
        end
        return false
    end
    reached = [[0]]
    for d in 1:n
        next = Int[]
        foreach(key -> extends(new -> (push!(next, new); false), key, d), reached[d])
        push!(reached, unique!(sort!(next)))
        sum(length, reached) > MAX_JOINT_ENTRIES && return
    end
    for d in n:-1:1
        filter!(key -> extends(new -> insorted(new, reached[d + 1]), key, d), reached[d])
    end
    for d in 1:(n - 1)
        isempty(active[d + 1]) && isempty(opens[d + 1]) && continue
        terms = map(active[d + 1]) do k
            s = findlast(s -> p.step_owner[s] == k && p.step_depth[s] <= d, eachindex(p.step_depth))
            (s, stride[k, d + 1], offset[k, d + 1], scale[k], p.moduli[k])
        end
        push!(cuts[d], (; terms, opens = collect(zip(opens[d + 1], open_stride[d + 1])), allowed = reached[d + 1]))
    end
    return
end

function _joint_key(cut, values, index)
    key = 0
    for (s, stride, offset, scale, m) in cut.terms
        key += stride * _joint_coordinate(values[s], offset, scale, m)
    end
    for (j, stride) in cut.opens
        key += stride * (index[j] - 1)
    end
    return key
end
_joint_coordinate(value, offset, scale, m) = (c = Int(div(value - offset, scale)); iszero(m) ? c : mod(c, m))

@testitem "Pruning by joint values keeps the states of filtering" begin
    using FermionicHilbertSpaces: AdditiveConstraint, particle_number
    # `constrain_space` with candidate states filters them without pruning.
    function samespace(H, constraint)
        pruned, filtered = constrain_space(H, constraint), constrain_space(H, constraint, basisstates(H))
        basisstates(pruned) == basisstates(filtered) || return false
        return !(pruned isa SectorHilbertSpace) || quantumnumbers(pruned) == quantumnumbers(filtered)
    end
    @fermions f
    @spins S 1 // 2
    n(i) = f[i]'f[i]
    up(i) = S[i][:z] + 1 // 2
    N = 12
    Hf, Hs = hilbert_space(f, 1:N), hilbert_space(S, 1:N)
    halfway = AdditiveConstraint([1.0]; spaces = [hilbert_space(f[i]) for i in 1:N], maps = s -> 0.5 * particle_number(s))
    cases = [
        (Hf, (sum(n(i) for i in 1:N) => 6, sum(i * n(i) for i in 1:N) => 30:45)),
        (Hf, NumberConservation(5) * ParityConservation(-1)),
        (Hf, (sum(n(i) for i in 1:8) => 3, sum(n(i) for i in 5:12) => 2)),
        (Hf, sum(5n(i) for i in 1:(N - 1)) + n(N) => 21),
        (Hs, (sum(up(i) for i in 1:N) => 3, [up(i) * up(i + 1) => 0 for i in 1:(N - 1)]...)),
        (Hs, (sum(up(i) for i in 1:N) => 3, sum(up(i) * up(i + 1) for i in 1:(N - 1)) => 1)),
        (Hf, NumberConservation(2) * halfway),
    ]
    @test all(((H, c),) -> samespace(H, c), cases)
end
