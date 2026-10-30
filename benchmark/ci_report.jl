# Ranked summary of an AirspeedVelocity run in results/, for the benchmark PR comment:
#     julia benchmark/ci_report.jl <base-rev> <head-rev>
using AirspeedVelocity: load_results
include(joinpath(@__DIR__, "report.jl"))

# A time changes when the medians differ by more than TIME_FLOOR and by more than SPREAD_FACTOR
# times the combined relative interquartile range of the samples. The IQR is not divided by √n:
# one process per revision cannot see the variation between processes, so the margin is wide,
# and TIME_FLOOR covers the few percent by which tight loops vary between processes.
const TIME_FLOOR, SPREAD_FACTOR, MEMORY_RATIO = 1.1, 3, 1.02
# Tight loops whose speed depends on where the JIT places them in memory, which varies between
# processes: identical code measures up to ~2× apart while the spread within a run stays small.
const NOISY = r"^partial_trace_algorithms/.*/Dense/FullAlg$|^embed/precomputed/apply_inplace$|^matrix_application/lazy liouvillian/"
const NOISY_RATIO = 2.5

ratio_beyond(a, b, threshold) = max(a / b, b / a) > threshold
spread(d) = haskey(d, "25") ? (d["75"] - d["25"]) / d["median"] : Inf
function time_changed(name, b, t)
    occursin(NOISY, name) && return ratio_beyond(t["median"], b["median"], NOISY_RATIO)
    return ratio_beyond(t["median"], b["median"], TIME_FLOOR) && abs(log(t["median"] / b["median"])) > SPREAD_FACTOR * hypot(spread(b), spread(t))
end
memory_changed(name, b, t) = ratio_beyond(t["memory"], b["memory"], MEMORY_RATIO)

function rows(base, head, key, scale, changed)
    names = sort!([name for name in intersect(keys(base), keys(head)) if !isnothing(get(base[name], key, nothing))])
    return map(names) do name
        (; name, baseline = scale * base[name][key], target = scale * head[name][key], changed = changed(name, base[name], head[name]))
    end
end

base, head = ARGS
results = load_results("FermionicHilbertSpaces", [base, head]; input_dir = "results")
println("## Benchmarks\n")
print_changes(stdout, "Time", rows(results[base], results[head], "median", 1.0e-9, time_changed), format_time)
print_changes(stdout, "Memory", rows(results[base], results[head], "memory", 1, memory_changed), format_bytes)
println(
    "A time counts as changed when the medians differ by more than $(TIME_FLOOR)× and by more than $(SPREAD_FACTOR)× the combined relative IQR of both sides ",
    "($(NOISY_RATIO)× for a few keys that jump between processes); memory when it differs by more than $(MEMORY_RATIO)×. Ratios are PR / `$base`.\n"
)
