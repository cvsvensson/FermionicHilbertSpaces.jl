# Benchmarks

`benchmarks.jl` defines `SUITE`, a BenchmarkTools `BenchmarkGroup`. The CI
workflow `.github/workflows/benchmark.yml` runs it with AirspeedVelocity on
every pull request, comparing the pull request against `main`. The benchmark
script from the pull request is run against both versions, so benchmarks may
only use functionality that `main` also has. Benchmarks for older package
versions are in `benchmarks_v0.9.jl`; `benchmarks.jl` picks the file that
matches the package version.

`ttfx.jl` defines `TTFX_SUITE`, which times fresh Julia processes that run
`using FermionicHilbertSpaces`, and that plus a small first-use workload. It takes about a
minute and is not part of `SUITE`.

## Running locally

Set up the environment once, from the repository root:

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

If the package depends on an unregistered version of a dependency, also
`Pkg.develop` that dependency into the benchmark environment.

Then run the suite and save the results:

```
julia --project=benchmark benchmark/compare.jl run results.json
```

Options:

- `--suite=main|ttfx|all` selects `SUITE` (the default), `TTFX_SUITE`, or both.
- `--filter=<regex>` runs only the benchmarks whose `/`-joined key matches.
- `--seconds=<s>` overrides the time budget of every benchmark, for example
  `--seconds=0.1` for a quick check that everything runs.

`compare.jl run` includes the benchmark files from its own directory into
whichever FermionicHilbertSpaces the environment given by `--project`
provides, so the same benchmark files can be run against several versions of
the package, each developed in its own environment.

## Comparing two versions

```
julia --project=benchmark benchmark/compare.jl compare base.json new.json
```

prints a table of minimum times, their ratio (new/base), allocation counts
and memory for every benchmark. Time ratios outside `1 ± 0.1` are flagged
(`--tol=<t>` changes the tolerance), and so is any change in the number of
allocations or a change in memory of more than 1% (`--memtol=<t>`).
A second column, "beyond noise", repeats the time verdict with the wider
band `1 ± 0.15` (`--band=<b>`), and the summary groups the benchmarks by
whether their allocation count went down, stayed the same or went up.

Timings on a desktop machine are noisy, and minimum times are more stable
than medians. Run the two versions alternately several times and pass the
files as comma-separated lists, for example
`compare base1.json,base2.json new1.json,new2.json`; for each benchmark the
smallest minimum over the files is used. Allocation counts are
deterministic and are the most reliable signal.
