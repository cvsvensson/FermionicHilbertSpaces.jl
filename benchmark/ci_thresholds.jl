# Writes ci_thresholds.toml, the time threshold of each benchmark key for ci_report.jl, from the
# benchmark comments of PRs that leave the package unchanged, so each compares main with itself:
#     julia benchmark/ci_thresholds.jl <folder of comment .md files>
# A threshold is exp(3 × the root mean square of the key's log time ratios), at least 1.05×. Keys
# that land at one of a few fixed speeds per process get 1.25× the largest log ratio, at least 2×.
using TOML, Statistics

const FIXED_SPEED = r"^partial_trace_algorithms/.*/Dense/FullAlg$|^embed/precomputed/apply_inplace$|^matrix_application/lazy liouvillian/"
const UNITS = Dict("ns" => 1.0e-9, "μs" => 1.0e-6, "us" => 1.0e-6, "ms" => 1.0e-3, "s" => 1.0)

function parse_time(cell)
    m = match(r"^([\d.e+-]+) ± [\d.e+-]+ (\S+)$", strip(cell))
    return isnothing(m) ? nothing : parse(Float64, m[1]) * UNITS[m[2]]
end

ratios = Dict{String, Vector{Float64}}()
for file in readdir(ARGS[1]; join = true)
    time_table = first(split(read(file, String), "Memory benchmarks"))
    for line in eachline(IOBuffer(time_table))
        cells = split(line, '|')
        length(cells) == 6 || continue
        base, head = parse_time(cells[3]), parse_time(cells[4])
        isnothing(base) || isnothing(head) || push!(get!(ratios, strip(cells[2]), Float64[]), log(head / base))
    end
end
threshold(key, r) = occursin(FIXED_SPEED, key) ? max(2.0, exp(1.25 * maximum(abs, r))) : max(1.05, exp(3 * sqrt(mean(abs2, r))))
thresholds = Dict(key => ceil(threshold(key, r); digits = 2) for (key, r) in ratios)
open(io -> TOML.print(io, thresholds; sorted = true), joinpath(@__DIR__, "ci_thresholds.toml"), "w")
