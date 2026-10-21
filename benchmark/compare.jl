# Run the benchmark suite and compare saved results.
#
#   julia --project=<env> compare.jl run <out.json> [--suite=main|ttfx|all] [--filter=<regex>] [--seconds=<s>]
#   julia --project=<env> compare.jl compare <base.json>[,<base2.json>...] <new.json>[,<new2.json>...] [--tol=<t>] [--band=<b>] [--memtol=<t>]
#
# `run` includes the benchmark files next to this script into the
# FermionicHilbertSpaces loaded by <env>, runs them, saves the results with
# `BenchmarkTools.save` and prints a summary. `--suite=main` (the default)
# runs `SUITE` from benchmarks.jl, `--suite=ttfx` the time-to-first-use
# benchmarks from ttfx.jl under the key "ttfx", and `--suite=all` both.
# `--filter` keeps the benchmarks whose "/"-joined key matches the regex, and
# `--seconds` overrides the time budget of every benchmark.
#
# `compare` matches benchmarks by key and compares minimum times, allocation
# counts and memory. When several comma-separated files are given for one
# side, the smallest minimum over the files is used, which helps on a noisy
# machine: alternate runs of the two versions and pass all files. A time
# ratio outside 1 ± tol (default 0.1) is flagged, as is any difference in
# the number of allocations or a relative difference in memory larger than
# memtol (default 0.01). Memory is not compared exactly because it can differ
# by a few bytes between Julia sessions. The "beyond noise" column repeats the
# time verdict with the wider band 1 ± band (default 0.15), for machines where
# minimum times drift by more than tol between identical runs. The summary
# groups the benchmarks by whether their allocation count went down, stayed
# the same or went up.
using BenchmarkTools
using Printf

function parse_options(args)
    positional = String[]
    options = Dict{String,String}()
    for arg in args
        if startswith(arg, "--")
            key, value = occursin('=', arg) ? split(arg[3:end], '='; limit=2) : (arg[3:end], "")
            options[key] = value
        else
            push!(positional, arg)
        end
    end
    return positional, options
end

function check_options(options, allowed)
    for key in keys(options)
        key in allowed || error("unknown option --$key; expected one of ", join("--" .* allowed, ", "))
        isempty(options[key]) && error("option --$key needs a value: --$key=<value>")
    end
end

joinkey(k) = join(k, "/")

function filtered(group::BenchmarkGroup, pattern)
    out = BenchmarkGroup()
    for (k, b) in BenchmarkTools.leaves(group)
        occursin(pattern, joinkey(k)) && (out[k] = b)
    end
    return out
end

function prettyrow(k, t)
    @sprintf("%-80s %10s %9d %10s %6d", joinkey(k), BenchmarkTools.prettytime(minimum(t.times)), t.allocs,
        BenchmarkTools.prettymemory(t.memory), length(t.times))
end

function run_suite(out, suite; seconds=nothing)
    if seconds !== nothing
        for (_, b) in BenchmarkTools.leaves(suite)
            b.params.seconds = seconds
        end
    end
    t = @elapsed results = run(suite; verbose=true)
    BenchmarkTools.save(out, results)
    println(@sprintf("%-80s %10s %9s %10s %6s", "benchmark", "min", "allocs", "memory", "n"))
    for (k, trial) in sort!(collect(BenchmarkTools.leaves(results)); by=joinkey ∘ first)
        println(prettyrow(k, trial))
    end
    println("run took ", round(t; digits=1), " s; results saved to ", out)
end

function cmd_run(positional, options)
    length(positional) == 1 || error("usage: compare.jl run <out.json> [options]")
    check_options(options, ("suite", "filter", "seconds"))
    which = get(options, "suite", "main")
    which in ("main", "ttfx", "all") || error("--suite must be main, ttfx or all")
    suite = BenchmarkGroup()
    t = @elapsed begin
        if which in ("main", "all")
            include(joinpath(@__DIR__, "benchmarks.jl"))
            for (k, b) in BenchmarkTools.leaves(invokelatest(getglobal, Main, :SUITE))
                suite[k] = b
            end
        end
        if which in ("ttfx", "all")
            include(joinpath(@__DIR__, "ttfx.jl"))
            suite["ttfx"] = invokelatest(getglobal, Main, :TTFX_SUITE)
        end
    end
    println("suite setup took ", round(t; digits=1), " s")
    haskey(options, "filter") && (suite = filtered(suite, Regex(options["filter"])))
    seconds = haskey(options, "seconds") ? parse(Float64, options["seconds"]) : nothing
    invokelatest(run_suite, positional[1], suite; seconds)
end

# Leaf key => trial with the smallest minimum time over the given files.
function load_best(paths)
    best = Dict{String,BenchmarkTools.Trial}()
    for path in split(paths, ',')
        for (k, t) in BenchmarkTools.leaves(only(BenchmarkTools.load(path)))
            key = joinkey(k)
            if !haskey(best, key) || minimum(t.times) < minimum(best[key].times)
                best[key] = t
            end
        end
    end
    return best
end

function cmd_compare(positional, options)
    length(positional) == 2 || error("usage: compare.jl compare <base.json> <new.json> [--tol=<t>] [--band=<b>] [--memtol=<t>]")
    check_options(options, ("tol", "band", "memtol"))
    tol = parse(Float64, get(options, "tol", "0.1"))
    band = parse(Float64, get(options, "band", "0.15"))
    memtol = parse(Float64, get(options, "memtol", "0.01"))
    base, new = load_best(positional[1]), load_best(positional[2])
    keys_all = sort!(collect(union(keys(base), keys(new))))
    width = maximum(length, keys_all; init=9)
    println("| ", rpad("benchmark", width), " | base min | new min | ratio | base allocs | new allocs | base memory | new memory | flags | beyond noise |")
    println("|", "-"^(width + 2), "|---|---|---|---|---|---|---|---|---|")
    logratios = Float64[]
    nfaster = nslower = nalloc = nonly_base = nonly_new = 0
    # allocation change (-1, 0, 1) => (count, beyond-band faster, beyond-band slower, log ratios)
    groups = Dict(c => [0, 0, 0, Float64[]] for c in (-1, 0, 1))
    for key in keys_all
        if !haskey(new, key) || !haskey(base, key)
            println("| ", rpad(key, width), " | | | | | | | | only in ", haskey(base, key) ? "base" : "new", " | |")
            haskey(base, key) ? (nonly_base += 1) : (nonly_new += 1)
            continue
        end
        b, n = base[key], new[key]
        tb, tn = minimum(b.times), minimum(n.times)
        ratio = tn / tb
        push!(logratios, log(ratio))
        flags = String[]
        if ratio < 1 - tol
            push!(flags, "faster")
            nfaster += 1
        elseif ratio > 1 + tol
            push!(flags, "SLOWER")
            nslower += 1
        end
        if b.allocs != n.allocs || abs(n.memory - b.memory) > memtol * max(b.memory, 1)
            push!(flags, "allocs")
            nalloc += 1
        end
        noise = ratio < 1 - band ? "faster" : ratio > 1 + band ? "SLOWER" : ""
        grp = groups[cmp(n.allocs, b.allocs)]
        grp[1] += 1
        noise == "faster" && (grp[2] += 1)
        noise == "SLOWER" && (grp[3] += 1)
        push!(grp[4], log(ratio))
        println("| ", rpad(key, width), " | ", BenchmarkTools.prettytime(tb), " | ", BenchmarkTools.prettytime(tn), " | ",
            @sprintf("%.3f", ratio), " | ", b.allocs, " | ", n.allocs, " | ", BenchmarkTools.prettymemory(b.memory), " | ",
            BenchmarkTools.prettymemory(n.memory), " | ", join(flags, ", "), " | ", noise, " |")
    end
    println()
    println(length(logratios), " benchmarks compared (time tolerance ", tol, ", memory tolerance ", memtol, "): ", nfaster, " faster, ", nslower,
        " slower, ", nalloc, " with changed allocations or memory; ", nonly_base, " only in base, ", nonly_new, " only in new")
    isempty(logratios) || println("geometric mean of time ratios (new/base): ", @sprintf("%.3f", exp(sum(logratios) / length(logratios))))
    println("by allocation count (time verdicts with noise band ", band, "):")
    for (c, label) in ((-1, "fewer allocations"), (0, "same allocations"), (1, "more allocations"))
        count, faster, slower, lr = groups[c]
        count == 0 && continue
        println("  ", rpad(label * ":", 19), lpad(count, 3), " benchmarks, ", faster, " faster, ", slower,
            " slower, geometric mean ratio ", @sprintf("%.3f", exp(sum(lr) / length(lr))))
    end
end

positional, options = parse_options(ARGS)
isempty(positional) && error("usage: compare.jl run|compare ...")
command = popfirst!(positional)
if command == "run"
    cmd_run(positional, options)
elseif command == "compare"
    cmd_compare(positional, options)
else
    error("unknown command $command; expected run or compare")
end
