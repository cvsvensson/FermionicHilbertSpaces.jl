# Compare FermionicHilbertSpaces between two git refs on cases larger than the CI suite.
#
#     julia benchmark/local/compare.jl [target=.] [baseline=origin/main] [--full] [--repeats=3] [--threads=4]
#
# `.` is the working tree; other refs are checked out with `git worktree add --detach` into a
# temporary directory. Both sides run the cases in cases.jl, which are always taken from this
# directory, in fresh processes that alternate between the sides. `--full` runs bigger sizes.
# Each process records the minimum time and bytes of a case. A case counts as changed only when
# every process of one side is more than 1.1× (time) or 1.02× (allocated bytes, which are nearly
# deterministic) above every process of the other. The report shows the ratio of the medians.
using Statistics: median
include(joinpath(@__DIR__, "..", "report.jl"))

const TIME_RATIO, BYTES_RATIO = 1.1, 1.02

function parse_args(args)
    flags = Dict(m[1] => m[2] for m in (match(r"^--([a-z]+)=?(.*)$", a) for a in args) if !isnothing(m))
    refs = filter(!startswith("--"), args)
    target, baseline = get(refs, 1, "."), get(refs, 2, "origin/main")
    nthreads = parse(Int, get(flags, "threads", string(min(4, Sys.CPU_THREADS))))
    return (; target, baseline, full = haskey(flags, "full"), repeats = parse(Int, get(flags, "repeats", "3")), nthreads)
end

git(root, args...) = readchomp(`git -C $root $args`)

function setup_side(root, tmp, label, ref)
    path = ref == "." ? root : joinpath(tmp, label)
    ref == "." || run(pipeline(`git -C $root worktree add --detach $path $ref`; stdout = devnull))
    sha = git(path, "rev-parse", "--short", "HEAD")
    env = mkpath(joinpath(tmp, "env-$label"))
    sources = "\n[sources]\nFermionicHilbertSpaces = {path = \"$(replace(path, '\\' => '/'))\"}\n"
    write(joinpath(env, "Project.toml"), read(joinpath(@__DIR__, "Project.toml"), String) * sources)
    run(`$(Base.julia_cmd()) --project=$env -e "using Pkg; Pkg.resolve(io = devnull); Pkg.instantiate(); Pkg.precompile()"`)
    name = ref == "." ? "working tree ($sha)" : "$ref ($sha)"
    return (; ref, path, env, name, results = Dict{String, Vector{Tuple{Float64, Int}}}())
end

function run_case!(side, opts, args...)
    out = tempname()
    run(ignorestatus(`$(Base.julia_cmd()) --project=$(side.env) --startup-file=no -t$(opts.nthreads) $(joinpath(@__DIR__, "cases.jl")) $args $out`))
    isfile(out) || return
    for line in eachline(out)
        name, t, bytes = split(line, '\t')
        push!(get!(side.results, name, Tuple{Float64, Int}[]), (parse(Float64, t), parse(Int, bytes)))
    end
    return rm(out)
end

function compare(base, target, metric)
    idx, threshold, format = metric == "Time" ? (1, TIME_RATIO, format_time) : (2, BYTES_RATIO, format_bytes)
    rows = map(sort!(collect(intersect(keys(base.results), keys(target.results))))) do name
        b, t = getindex.(base.results[name], idx), getindex.(target.results[name], idx)
        changed = minimum(t) > threshold * maximum(b) || threshold * maximum(t) < minimum(b)
        (; name, baseline = median(b), target = median(t), changed)
    end
    return print_changes(stdout, metric, filter(r -> r.baseline >= 0, rows), format)
end

function report(base, target, opts)
    println("\n$(target.name) vs $(base.name): $(opts.repeats) repeats, $(opts.nthreads) threads, $(opts.full ? "full" : "default") sizes\n")
    compare(base, target, "Time")
    compare(base, target, "Memory")
    for (a, b) in ((base, target), (target, base))
        missing_names = setdiff(keys(a.results), keys(b.results))
        isempty(missing_names) || println("Only in $(a.name): ", join(sort!(collect(missing_names)), ", "))
    end
    return
end

function main(args)
    opts = parse_args(args)
    root = git(@__DIR__, "rev-parse", "--show-toplevel")
    tmp = mktempdir()
    sides = []
    try
        push!(sides, setup_side(root, tmp, "baseline", opts.baseline))
        push!(sides, setup_side(root, tmp, "target", opts.target))
        workflows = split(readchomp(`$(Base.julia_cmd()) --project=$(sides[1].env) --startup-file=no $(joinpath(@__DIR__, "cases.jl")) list`), '\n')
        t0 = time()
        for r in 1:opts.repeats
            @info "Repeat $r of $(opts.repeats)"
            foreach(side -> run_case!(side, opts, "runtime", opts.full ? "full" : "default"), isodd(r) ? sides : reverse(sides))
            foreach(((side, w),) -> run_case!(side, opts, "firstcall", w), Iterators.product(sides, workflows))
        end
        @info "Measured in $(round(Int, time() - t0)) s"
        report(sides..., opts)
    finally
        for side in sides
            side.ref == "." || run(`git -C $root worktree remove --force $(side.path)`)
        end
        rm(tmp; recursive = true, force = true)
    end
    return
end

main(ARGS)
