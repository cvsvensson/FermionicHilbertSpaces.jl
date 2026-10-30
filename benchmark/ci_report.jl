# Ranked summary of an AirspeedVelocity run in results/, for the benchmark PR comment:
#     julia benchmark/ci_report.jl <base-rev> <head-rev>
using AirspeedVelocity: load_results
using TOML: parsefile
include(joinpath(@__DIR__, "report.jl"))

# Time thresholds per key, from ci_thresholds.jl; keys without one, such as new benchmarks, get
# DEFAULT_TIME_RATIO. Allocated bytes are nearly deterministic, so one small MEMORY_RATIO serves all.
const TIME_RATIOS = parsefile(joinpath(@__DIR__, "ci_thresholds.toml"))
const DEFAULT_TIME_RATIO, MEMORY_RATIO = 1.15, 1.02

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
print_changes(stdout, "Time", rows(results[base], results[head], "median", 1.0e-9, name -> get(TIME_RATIOS, name, DEFAULT_TIME_RATIO)), format_time)
print_changes(stdout, "Memory", rows(results[base], results[head], "memory", 1, _ -> MEMORY_RATIO), format_bytes)
println(
    "A median time counts as changed beyond its key's threshold, measured from runs that compare main with itself ($(DEFAULT_TIME_RATIO)× for keys without one); ",
    "memory beyond $(MEMORY_RATIO)×. Ratios are PR / `$base`.\n"
)
