# Ranked summary of an AirspeedVelocity run in results/, for the benchmark PR comment:
#     julia benchmark/ci_report.jl <base-rev> <head-rev>
using AirspeedVelocity: load_results
include(joinpath(@__DIR__, "report.jl"))

const TIME_RATIO, MEMORY_RATIO = 1.5, 1.02
# The dense FullAlg partial traces are a tight loop whose speed depends on where the JIT places
# it in memory, which varies between processes: identical code measures up to ~2× apart.
const NOISY, NOISY_RATIO = r"^partial_trace_algorithms/.*/Dense/FullAlg$", 2.5

function rows(base, head, key, threshold, scale)
    names = sort!([name for name in intersect(keys(base), keys(head)) if !isnothing(get(base[name], key, nothing))])
    return map(names) do name
        baseline, target = scale * base[name][key], scale * head[name][key]
        (; name, baseline, target, changed = max(target / baseline, baseline / target) > threshold(name))
    end
end

base, head = ARGS
results = load_results("FermionicHilbertSpaces", [base, head]; input_dir = "results")
println("## Benchmarks\n")
print_changes(stdout, "Time", rows(results[base], results[head], "median", name -> occursin(NOISY, name) ? NOISY_RATIO : TIME_RATIO, 1.0e-9), format_time)
print_changes(stdout, "Memory", rows(results[base], results[head], "memory", _ -> MEMORY_RATIO, 1), format_bytes)
println("Changed means a median time ratio above $TIME_RATIO ($NOISY_RATIO for the dense FullAlg partial traces) or a memory ratio above $MEMORY_RATIO, PR / `$base`.\n")
