# Sums that vary at a common depth are correlated, such as a charge and a dipole moment, so the bounds of each sum alone
# accept branches that no completion satisfies. Pruning by the joint values that valid states reach is exact. Sums with
# inexact values, or with more than `MAX_JOINT_ENTRIES` joint values, keep the bounds alone. A single sum is joined
# when its values have gaps, as for weights `5` and `1`.
const MAX_JOINT_ENTRIES = 2^20

function _joint_cuts(p)
    n, T = length(p.sizes), eltype(p.constants)
    Term = @NamedTuple{step::Int, stride::Int, offset::T, scale::T, modulus::Int}
    Cut = @NamedTuple{terms::Vector{Term}, opens::Vector{Tuple{Int, Int}}, allowed::Vector{Int}}
    cuts = [Cut[] for _ in 1:n]
    T <: Union{Integer, Rational} || return cuts
    steps = findall(gs -> any(g -> !allequal(p.tables[g]), gs), p.step_groups)
    isempty(steps) && return cuts
    gapped = any(s -> any(g -> !all(isone, diff(sort!(unique(p.tables[g])))), p.step_groups[s]), steps)
    !gapped && allequal(p.step_owner[steps]) && return cuts
    l = _joint_layout(p, steps)
    reached = isnothing(l) ? nothing : _joint_keys(p, l)
    isnothing(reached) && return cuts
    for d in 1:(n - 1)
        isempty(l.active[d + 1]) && isempty(l.opens[d + 1]) && continue
        terms = map(l.active[d + 1]) do k
            step = findlast(s -> p.step_owner[s] == k && p.step_depth[s] <= d, eachindex(p.step_depth))
            (; step, stride = l.stride[k, d + 1], offset = l.offset[k, d + 1], scale = l.scale[k], modulus = p.moduli[k])
        end
        push!(cuts[d], (; terms, opens = collect(zip(l.opens[d + 1], l.open_stride[d + 1])), allowed = reached[d + 1]))
    end
    return cuts
end

# A cut after depth `d` has the key `Σₖ stride[k, d + 1] * Kₖ` over the active sums, which vary on both sides of the
# cut, where the value of the sum `k` is `offset[k, d + 1] + Kₖ * scale[k]`, with `Kₖ` modulo the modulus, if any, plus
# `stride * (x - 1)` for the state `x` at each open depth, a depth before the cut that a later step reads. `allowed`
# holds the keys that valid states reach.
function _joint_layout(p, steps)
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
    stride, open_stride = zeros(Int, K, n + 1), Vector{Vector{Int}}(undef, n + 1)
    for cut in 1:(n + 1)
        factors = [range[active[cut], cut]; [p.sizes[j] for j in opens[cut]]]
        prod(big, factors; init = big(1)) > typemax(Int) && return
        strides = cumprod([1; factors])
        stride[active[cut], cut] = strides[eachindex(active[cut])]
        open_stride[cut] = strides[length(active[cut]) .+ eachindex(opens[cut])]
    end
    steps_at = [steps[searchsorted(p.step_depth[steps], d)] for d in 1:n]
    buffers = (index = zeros(Int, n), coordinate = zeros(Int, K), value = zeros(T, K), stepped = zeros(Int, K))
    return (; scale, offset, range, first_depth, active, opens, stride, open_stride, steps_at, buffers...)
end

# The keys that the prefixes reach, then only those that a suffix completes.
function _joint_keys(p, l)
    reached, total = [[0]], 1
    for d in eachindex(p.sizes)
        next = Int[]
        for key in reached[d]
            _joint_extends(new -> (push!(next, new); false), key, d, p, l)
            total + length(next) > MAX_JOINT_ENTRIES && return
        end
        push!(reached, unique!(sort!(next)))
        total += length(next)
    end
    for d in reverse(eachindex(p.sizes))
        filter!(key -> _joint_extends(new -> insorted(new, reached[d + 1]), key, d, p, l), reached[d])
    end
    return reached
end

# Whether `f` holds for the key of an extension of `key`, at the cut before depth `d`, by a state at `d` that the bounds
# of each sum allow.
function _joint_extends(f, key, d, p, l)
    (; index, coordinate, value, stepped) = l
    foreach(k -> coordinate[k] = mod(div(key, l.stride[k, d]), l.range[k, d]), l.active[d])
    foreach((j, st) -> index[j] = mod(div(key, st), p.sizes[j]) + 1, l.opens[d], l.open_stride[d])
    for x in 1:p.sizes[d]
        index[d] = x
        ok = true
        for s in l.steps_at[d]
            k = p.step_owner[s]
            v = l.offset[k, d] + (l.first_depth[k] < d ? coordinate[k] * l.scale[k] : zero(eltype(value)))
            for g in p.step_groups[s]
                v += p.tables[g][_table_index(p.strides[g], index, p.positions[g])]
            end
            ok = p.reachable(v, s)
            ok || break
            value[k], stepped[k] = v, d
        end
        ok || continue
        new = 0
        for k in l.active[d + 1]
            c = stepped[k] == d ? _joint_coordinate(value[k], l.offset[k, d + 1], l.scale[k], p.moduli[k]) : coordinate[k]
            new += l.stride[k, d + 1] * c
        end
        for (j, st) in zip(l.opens[d + 1], l.open_stride[d + 1])
            new += st * (index[j] - 1)
        end
        f(new) && return true
    end
    return false
end

function _joint_key(cut, values, index)
    key = 0
    for t in cut.terms
        key += t.stride * _joint_coordinate(values[t.step], t.offset, t.scale, t.modulus)
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
        (Hs, (sum(S[i][:z] for i in 1:N) => -3, sum(up(i) * up(i + 1) for i in 1:(N - 1)) => 1)),
        (Hf, NumberConservation(2) * halfway),
    ]
    @test all(((H, c),) -> samespace(H, c), cases)
end
