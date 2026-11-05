# Ranked summary of an AirspeedVelocity run in results/, for the benchmark PR comment:
#     julia benchmark/ci_report.jl <base-rev> <head-rev>
using AirspeedVelocity: load_results
include(joinpath(@__DIR__, "report.jl"))

# Time thresholds from runs that compare main with itself. Most keys stay within 1.15×; NOISY keys
# moved by up to 1.4×, and JUMPY keys, which run at one of a few speeds per process, by up to 1.9×.
# Allocated bytes are nearly deterministic.
const TIME_RATIO, NOISY_RATIO, JUMPY_RATIO, MEMORY_RATIO = 1.15, 1.5, 2.5, 1.02
const NOISY = [
    "bosons/matrix_representation", "embed/precomputed/apply_inplace", "matrix_application/lazy liouvillian/mul sparse",
    "matrix_application/lazy liouvillian/mul!", "partial_trace/product space/map", "partial_trace/pure_state_full_space",
    "permutations/few_large", "reshape/repeat", "spin/matrix_representation_heisenberg", "symbolic/sum",
    "symbolic_states/projected_op_on_state",
]
const JUMPY = [
    "hubbard/constrain_no_double_occupation_N30", "partial_trace_algorithms/Full space/Dense/FullAlg",
    "partial_trace_algorithms/Sparse space/Dense/FullAlg",
]
time_ratio(name) = name in JUMPY ? JUMPY_RATIO : name in NOISY ? NOISY_RATIO : TIME_RATIO

ratio_beyond(a, b, threshold) = max(a / b, b / a) > threshold

function rows(base, head, key, scale, threshold)
    names = sort!([name for name in intersect(keys(base), keys(head)) if !isnothing(get(base[name], key, nothing))])
    return map(names) do name
        baseline, target = scale * base[name][key], scale * head[name][key]
        (; name, baseline, target, changed = ratio_beyond(target, baseline, threshold(name)), threshold = threshold(name))
    end
end

base, head = ARGS
results = load_results("FermionicHilbertSpaces", [base, head]; input_dir = "results")
println("## Benchmarks\n")
print_changes(stdout, "Time", rows(results[base], results[head], "median", 1.0e-9, time_ratio), format_time)
print_changes(stdout, "Memory", rows(results[base], results[head], "memory", 1, _ -> MEMORY_RATIO), format_bytes)
println(
    "A median time counts as changed beyond $(TIME_RATIO)×, or $(NOISY_RATIO)× and $(JUMPY_RATIO)× for two short lists of noisy keys in `ci_report.jl`; ",
    "memory beyond $(MEMORY_RATIO)×. Ratios are PR / `$base`.\n"
)
