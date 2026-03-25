
if PACKAGE_VERSION < v"0.10"
    include("benchmarks_v0.9.jl")
else
    include("benchmarks_v0.10.jl")
end
