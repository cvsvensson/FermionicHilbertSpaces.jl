# Ranked summary of benchmark changes, shared by local/compare.jl and the benchmark workflow.

format_time(t) = t < 1.0e-6 ? "$(round(t * 1.0e9, sigdigits = 3)) ns" : t < 1.0e-3 ? "$(round(t * 1.0e6, sigdigits = 3)) µs" :
    t < 1 ? "$(round(t * 1.0e3, sigdigits = 3)) ms" : "$(round(t, sigdigits = 3)) s"
format_bytes(b) = b < 2^10 ? "$(round(Int, b)) B" : b < 2^20 ? "$(round(b / 2^10, sigdigits = 3)) KiB" :
    b < 2^30 ? "$(round(b / 2^20, sigdigits = 3)) MiB" : "$(round(b / 2^30, sigdigits = 3)) GiB"

# Each row is `(; name, baseline, target, changed)`, optionally with a `threshold`. Prints a count of
# the regressed, improved and unchanged rows, then a Markdown table of the changed ones: regressions
# first, each group ranked by ratio.
function print_changes(io, metric, rows, format)
    ratio(r) = r.target / r.baseline
    changed = filter(r -> r.changed, rows)
    regressions = sort!(filter(r -> ratio(r) > 1, changed); by = ratio, rev = true)
    improvements = sort!(filter(r -> ratio(r) < 1, changed); by = ratio)
    println(io, "**$metric**: $(length(regressions)) regressed, $(length(improvements)) improved, $(length(rows) - length(changed)) unchanged\n")
    isempty(changed) && return
    columns = ["ratio" => r -> "$(round(ratio(r), sigdigits = 3))×", "benchmark" => r -> "`$(r.name)`", "baseline" => r -> format(r.baseline), "target" => r -> format(r.target)]
    hasproperty(first(changed), :threshold) && insert!(columns, 2, "threshold" => r -> "$(r.threshold)×")
    cells = [[first.(columns)]; [[f(r) for f in last.(columns)] for r in [regressions; improvements]]]
    widths = [maximum(c -> textwidth(c[j]), cells) for j in eachindex(columns)]
    line(c) = println(io, "| ", join(rpad.(c, widths), " | "), " |")
    line(cells[1])
    line(["-"^w for w in widths])
    foreach(line, cells[2:end])
    return println(io)
end
