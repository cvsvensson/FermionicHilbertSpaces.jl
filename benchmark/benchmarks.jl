using FermionicHilbertSpaces

# AirspeedVelocity defines `PACKAGE_VERSION` before including this file. When
# it is not defined, use the version of the loaded package.
const BENCHMARKED_VERSION = @isdefined(PACKAGE_VERSION) ? PACKAGE_VERSION : pkgversion(FermionicHilbertSpaces)

if BENCHMARKED_VERSION < v"0.10"
    include("benchmarks_v0.9.jl")
else
    include("benchmarks_v0.10.jl")
end
